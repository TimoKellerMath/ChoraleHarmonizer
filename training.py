"""Trainieren: Modell bauen/laden, Loss/Metriken, Trainer-Konfiguration.

Das Modell ist ein kleines, von Grund auf trainiertes T5 (Encoder liest den
Sopran, Decoder erzeugt die Zielsequenz "S A T B ;" pro 16tel). Der Loss ist
eine positionsgewichtete Cross-Entropy, die das Gradientensignal auf die
eigentlichen A/T/B-Entscheidungen konzentriert.
"""

from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    GenerationConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

import config as C


def has_final_model(path):
    """Fertig trainiertes Modell (nicht nur Checkpoints) vorhanden?"""
    return (Path(path) / 'model.safetensors').exists() \
        or (Path(path) / 'pytorch_model.bin').exists()


def has_checkpoints(path):
    """Zwischenstände eines (abgebrochenen) Trainings vorhanden?"""
    return Path(path).exists() and any(Path(path).glob('checkpoint-*'))


def build_model(music_tokenizer):
    """Lädt ein fertig trainiertes Modell oder erstellt ein neues.

    Ein abgebrochenes Training (Checkpoints ohne finales Modell) führt zu
    need_to_train=True; train_and_save setzt dann am letzten Checkpoint fort.

    Returns: (model, need_to_train)
    """
    vocab_size = len(music_tokenizer)
    if has_final_model(C.MODEL_DIR) and not C.FORCE_RETRAIN:
        print(f"✓ Lade existierendes Modell aus {C.MODEL_DIR}")
        model = T5ForConditionalGeneration.from_pretrained(C.MODEL_DIR)
        need_to_train = False
    elif C.TRAIN_FROM_SCRATCH:
        print("⏳ Erstelle neues Modell (from scratch, musik-dimensioniert)...")
        model_config = T5Config(
            vocab_size=vocab_size,
            d_model=C.SCRATCH_D_MODEL,
            d_ff=C.SCRATCH_D_FF,
            d_kv=C.SCRATCH_D_MODEL // C.SCRATCH_HEADS,
            num_layers=C.SCRATCH_LAYERS,
            num_decoder_layers=C.SCRATCH_LAYERS,
            num_heads=C.SCRATCH_HEADS,
            dropout_rate=C.SCRATCH_DROPOUT,
            relative_attention_num_buckets=C.REL_ATTN_NUM_BUCKETS,
            relative_attention_max_distance=C.REL_ATTN_MAX_DISTANCE,
            decoder_start_token_id=music_tokenizer.cls_token_id,
            pad_token_id=music_tokenizer.pad_token_id,
            eos_token_id=music_tokenizer.sep_token_id,
        )
        model = T5ForConditionalGeneration(model_config)
        need_to_train = True
    else:
        print("⏳ Erstelle neues Modell (t5-small vortrainiert)...")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        need_to_train = True

    # ====== EMBEDDINGS an MusicTokenizer anpassen ======
    if model.config.vocab_size != vocab_size:
        print(f"\n✓ Resize embeddings to vocab_size={vocab_size}")
        model.resize_token_embeddings(vocab_size)  # joint resize: Encoder, Decoder und LM-Head synchron
        model.config.vocab_size = vocab_size

    model.config.decoder_start_token_id = music_tokenizer.cls_token_id
    model.config.pad_token_id = music_tokenizer.pad_token_id
    model.config.eos_token_id = music_tokenizer.sep_token_id

    model.generation_config = GenerationConfig(
        decoder_start_token_id=music_tokenizer.cls_token_id,
        eos_token_id=music_tokenizer.sep_token_id,
        pad_token_id=music_tokenizer.pad_token_id,
        # Bewusst KEINE repetition_penalty / num_beams hier: die eigentlichen
        # Sampling-Parameter stehen in generation.py (siehe GEN_* in config.py).
    )

    return model, need_to_train


def make_trainer(model, music_tokenizer, tokenized_train, tokenized_val):
    """Baut den Seq2SeqTrainer mit gewichtetem Loss und Musik-Metriken."""
    underscore_id = music_tokenizer.convert_tokens_to_ids("_")
    bar_id = music_tokenizer.convert_tokens_to_ids("|")
    semicolon_id = music_tokenizer.convert_tokens_to_ids(";")

    # ====== CE-GEWICHTE (pro Token-ID) ======
    # Strukturtoken (|, ;) bekommen Gewicht 0: ihre Positionen sind durch die
    # Sopran-Eingabe vollständig festgelegt. Alle anderen Token (Noten und _)
    # bekommen Gewicht 1.0: bei nur ~53% _ ist das Class-Imbalance-Problem
    # mild genug, dass eine Umgewichtung mehr schadet als nützt (führt zu
    # zu vielen Note-Onsets statt Haltungen).
    ce_weight = torch.ones(len(music_tokenizer))
    ce_weight[bar_id] = 0.0
    ce_weight[semicolon_id] = 0.0

    # Noten-Tokens (inkl. '_' und 'R') für das Decoder-Input-Dropout:
    # keine Specials, keine Struktur, keine Taktart-/Tonart-Token
    note_ids = torch.tensor([
        i for i, t in music_tokenizer.inverse_dict.items()
        if t not in music_tokenizer.special_tokens
        and t not in ('|', ';') and '/' not in t and ':' not in t
    ])

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = np.array(predictions)
        labels = np.array(labels)

        min_len = min(predictions.shape[1], labels.shape[1])
        predictions = predictions[:, :min_len]
        labels = labels[:, :min_len]

        valid_mask = labels != -100

        # Gesamt-Token-Accuracy
        correct = int(((predictions == labels) & valid_mask).sum())
        total = int(valid_mask.sum())
        token_acc = correct / total if total > 0 else 0.0

        # Note-Accuracy: nur echte Noten — keine Haltungen (_) und keine
        # Strukturtoken (;, |), da deren Positionen vorgegeben sind und
        # ihre Vorhersage nicht zur eigentlichen Harmonisierungsaufgabe gehört.
        structural = (labels == underscore_id) | (labels == semicolon_id) | (labels == bar_id)
        note_mask = (~structural) & valid_mask
        if C.NUM_VOICES > 2:
            # Sopran-Slots (erstes Token jeder Beat-Gruppe, d.h. nach ';' oder '|'
            # bzw. an Position 0) sind bloße Kopien der Eingabe und werden bei der
            # Generierung forciert — nicht in die Note-Accuracy einrechnen.
            prev = np.full_like(labels, semicolon_id)
            prev[:, 1:] = labels[:, :-1]
            soprano_slot = ((prev == semicolon_id) | (prev == bar_id)) & (labels != bar_id)
            note_mask &= ~soprano_slot
        note_correct = int(((predictions == labels) & note_mask).sum())
        note_total = int(note_mask.sum())
        note_acc = note_correct / note_total if note_total > 0 else 0.0

        return {"token_accuracy": token_acc, "note_accuracy": note_acc}

    def preprocess_logits_for_metrics(logits, labels):  # noqa: ARG001
        """Logits -> argmax-IDs, bevor compute_metrics sie bekommt
        (nötig bei predict_with_generate=False)."""
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")

        # Decoder-Input-Dropout (nur Training): ein Teil der A/T/B-Tokens in
        # der Teacher-Forcing-Vergangenheit wird durch Zufallsnoten ersetzt.
        # Das Modell lernt, sich von eigenen Fehlern zu erholen, statt sie zu
        # verketten (Exposure Bias: beim freien Generieren sieht es die eigene,
        # teils falsche Vergangenheit). Labels bleiben unverändert.
        if num_items_in_batch is not None and C.DECODER_INPUT_DROPOUT > 0.0:
            dec = inputs.get('decoder_input_ids')
            if dec is not None:
                prev = torch.full_like(labels, semicolon_id)
                prev[:, 1:] = labels[:, :-1]
                soprano_slot = ((prev == semicolon_id) | (prev == bar_id)) & (labels != bar_id)
                atb = ((labels != -100) & (labels != bar_id)
                       & (labels != semicolon_id) & ~soprano_slot)
                # Decoder-Input-Position t entspricht Label-Position t-1
                drop = atb[:, :-1] & (torch.rand(labels[:, :-1].shape, device=labels.device)
                                      < C.DECODER_INPUT_DROPOUT)
                if drop.any():
                    dec = dec.clone()
                    pool = note_ids.to(dec.device)
                    rand = pool[torch.randint(len(pool), (int(drop.sum()),), device=dec.device)]
                    dec[:, 1:][drop] = rand
                    inputs = {**inputs, 'decoder_input_ids': dec}

        outputs = model(**inputs)
        logits = outputs.logits

        # Per-Token-CE, Gewichtung danach: Token-ID-Gewichte (| und ; = 0) mal
        # Positionsgewichte (Sopran-Kopien klein). Ohne diese Umgewichtung machen
        # Haltungen + Sopran-Kopien + Struktur ~2/3 des Targets aus und die
        # eigentlichen A/T/B-Entscheidungen bekommen nur einen Bruchteil des
        # Gradienten (Symptom: eval_loss sinkt, note_accuracy stagniert/fällt).
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none', label_smoothing=C.LABEL_SMOOTHING)
        per_token = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        ).view(labels.shape)

        valid = labels != -100
        weights = ce_weight.to(logits.device)[labels.clamp(min=0)]
        if C.NUM_VOICES > 2:
            # Sopran-Slot = erstes Token jeder Beat-Gruppe (nach ';' oder '|', bzw. Position 0)
            prev = torch.full_like(labels, semicolon_id)
            prev[:, 1:] = labels[:, :-1]
            soprano_slot = ((prev == semicolon_id) | (prev == bar_id)) & (labels != bar_id)
            weights = torch.where(soprano_slot, weights * C.SOPRANO_SLOT_WEIGHT, weights)
        weights = weights * valid
        loss = (per_token * weights).sum() / weights.sum().clamp(min=1e-8)

        # Neuere transformers-Versionen teilen bei Modellen mit loss-kwargs (T5)
        # NICHT mehr durch gradient_accumulation_steps — bei unserem Mittelwert-
        # Loss müssen wir das selbst tun, sonst sind geloggter Loss und
        # Gradienten um den Faktor GRAD_ACCUM_STEPS zu groß.
        # num_items_in_batch ist nur im Training gesetzt, in der Evaluation None.
        if num_items_in_batch is not None:
            loss = loss / C.GRAD_ACCUM_STEPS

        return (loss, outputs) if return_outputs else loss

    training_args = Seq2SeqTrainingArguments(
        output_dir=C.MODEL_DIR,
        # predict_with_generate=False: eval uses teacher-forced forward pass (fast),
        # metrics are computed from argmax of logits (overoptimistic vs. free generation
        # but fine for checkpoint selection by note_accuracy).
        predict_with_generate=False,
        per_device_train_batch_size=C.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=C.VAL_BATCH_SIZE,
        num_train_epochs=C.NUM_EPOCHS,
        learning_rate=C.LEARNING_RATE,
        warmup_steps=C.WARMUP_STEPS,
        weight_decay=0.01,
        # label smoothing passiert im gepatchten compute_loss (LABEL_SMOOTHING)
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # Checkpoint-Auswahl nach note_accuracy statt eval_loss: eval_loss kann
        # weiter sinken (bessere Kalibrierung der leichten Tokens), während die
        # eigentlichen Notenentscheidungen schon wieder schlechter werden.
        metric_for_best_model="note_accuracy",
        greater_is_better=True,
        gradient_accumulation_steps=C.GRAD_ACCUM_STEPS,
        fp16=torch.cuda.is_available(),
        fp16_full_eval=False,  # avoid NaN with fp16 + gradient_checkpointing on T5
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=[],
        # AdamW statt Adafactor: konvergiert bei dieser Modellgröße (~10M Param.)
        # meist schneller; Speicher ist hier kein Engpass.
        optim="adamw_torch",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=music_tokenizer,
        model=model,
        padding='longest',
        pad_to_multiple_of=None,
        label_pad_token_id=-100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        processing_class=music_tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.compute_loss = compute_loss
    return trainer


def train_and_save(trainer, music_tokenizer):
    print("\n" + "=" * 80)
    print("STARTE TRAINING")
    print("=" * 80)
    # Abgebrochenes Training am letzten Checkpoint fortsetzen (Modell-,
    # Optimizer- und Scheduler-Zustand; übersprungene Batches der
    # angebrochenen Epoche werden dank seed-fixiertem Shuffle korrekt
    # ausgelassen). FORCE_RETRAIN=True startet bewusst bei null.
    resume = has_checkpoints(C.MODEL_DIR) and not C.FORCE_RETRAIN
    if resume:
        print("↻ Setze abgebrochenes Training am letzten Checkpoint fort")
    trainer.train(resume_from_checkpoint=True if resume else None)

    print(f"\n✓ Speichere Modell in {C.MODEL_DIR}")
    trainer.save_model(C.MODEL_DIR)

    vocab_file = Path(C.TOKENIZER_DIR) / "music_vocab.json"
    music_tokenizer.save_vocab(vocab_file)
