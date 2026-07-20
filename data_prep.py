"""Einlesen der Daten: Rohdatensatz laden, Splits bilden, tokenisieren.

Erwartet die von bach_chorales.py erzeugte Datendatei (Format v2.1):
  Quelle:  "4/4 C5 _ _ _ | ..."            (Sopran mit Taktart-Token und Fermaten ^)
  Ziel:    "C5 A4 F4 F3 ; _ _ _ _ ; ..."   (pro 16tel "S A T B ;", Taktstriche |)
"""

import json
from pathlib import Path

from datasets import Dataset

import config as C
from music_tokenizer import MusicTokenizer


def load_raw_dataset():
    if not Path(C.RAW_DATA_FILE).exists():
        raise FileNotFoundError(f"Daten nicht gefunden: {C.RAW_DATA_FILE} "
                                f"(erzeugen mit: python bach_chorales.py)")
    with open(C.RAW_DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_max_length(raw_dataset):
    max_length = 0
    for src, tgt in raw_dataset:
        src_len = len(src.split()) + 2  # + 2: BOS/SEP token
        tgt_len = len(tgt.split()) + 2
        max_length = max(max_length, src_len, tgt_len)
    print(f"Max length: {max_length}")
    return max_length


def split_into_phrases(src, tgt, min_steps=8):
    """Zerlegt ein (Quelle, Ziel)-Paar an den Fermaten in Phrasen.

    Globale Präfix-Token (Taktart '/', Tonart ':') werden jeder Phrase
    vorangestellt; Taktstriche werden dem vorangehenden Schritt zugeordnet,
    sodass Quelle und Ziel konsistent geschnitten sind. Phrasen kürzer als
    min_steps Sechzehntel entfallen; besteht das Stück nur aus einer Phrase,
    wird nichts zurückgegeben (wäre ein Duplikat)."""
    src_toks = src.split()
    prefix = []
    i = 0
    while i < len(src_toks) and ('/' in src_toks[i] or ':' in src_toks[i]):
        prefix.append(src_toks[i])
        i += 1

    # Quelle in 16tel-Schritte gliedern (Taktstriche an den vorigen Schritt)
    src_steps = []
    fermata = []
    for tok in src_toks[i:]:
        if tok == '|':
            if src_steps:
                src_steps[-1].append('|')
            continue
        src_steps.append([tok])
        fermata.append(tok.endswith('^'))

    # Ziel ebenso: eine Gruppe endet mit ';'
    tgt_steps = []
    cur = []
    for tok in tgt.split():
        if tok == '|':
            if tgt_steps:
                tgt_steps[-1].append('|')
            continue
        cur.append(tok)
        if tok == ';':
            tgt_steps.append(cur)
            cur = []

    if len(tgt_steps) != len(src_steps):
        return []  # inkonsistentes Paar — lieber keine Phrasen erzeugen

    # Phrasenanfänge: erster Onset nach einer Fermate
    starts = [0]
    after = False
    for s, toks in enumerate(src_steps):
        if toks[0] != '_' and after:
            starts.append(s)
            after = False
        if fermata[s]:
            after = True
    if len(starts) < 2:
        return []

    phrases = []
    bounds = starts + [len(src_steps)]
    for a, b in zip(bounds, bounds[1:]):
        if b - a < min_steps:
            continue
        src_p = ' '.join(prefix + [t for st in src_steps[a:b] for t in st])
        tgt_p = ' '.join(t for st in tgt_steps[a:b] for t in st)
        phrases.append((src_p, tgt_p))
    return phrases


def load_or_compute_dataset():
    """Lädt Rohdaten, baut Tokenizer-Vokabular, splittet und tokenisiert.

    Returns: (tokenized_train, tokenized_val, raw_dataset, test_data, music_tokenizer)
    """
    raw_dataset = load_raw_dataset()
    calculate_max_length(raw_dataset)

    # ===== MUSIC TOKENIZER ERSTELLEN =====
    music_tokenizer = MusicTokenizer()
    music_tokenizer.build_vocab(raw_dataset)

    print(f"Tokens ({len(music_tokenizer)}): "
          f"{sorted([t for t in music_tokenizer.inverse_dict.values() if t not in music_tokenizer.special_tokens])[:20]}...")

    vocab_file = Path(C.TOKENIZER_DIR) / "music_vocab.json"
    music_tokenizer.save_vocab(vocab_file)
    print(f"✓ Vokabular gespeichert in {vocab_file}")

    # ====== SPLITS ERSTELLEN ======
    # Der Datensatz enthält je Choral 12 Transpositionen als aufeinanderfolgende
    # Einträge; Val/Test werden in ganzen Chorälen (× 12) abgetrennt.
    assert len(raw_dataset) % 12 == 0

    train_with_transposed_chorales = True
    if train_with_transposed_chorales:
        multiple = 12
    else:
        multiple = 1
        raw_dataset = [raw_dataset[i] for i in range(0, len(raw_dataset), 12)]

    val_size = C.VAL_CHORALES * multiple
    test_size = C.TEST_CHORALES * multiple

    n = len(raw_dataset)
    train_size = n - val_size - test_size  # all remaining samples go to train

    train_data = raw_dataset[:train_size]
    val_data = raw_dataset[train_size:train_size + val_size]
    test_data = raw_dataset[-test_size:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Phrasen-Augmentierung: Kadenzen konzentriert lernen. Nur aus den
    # Originaltonarten des Trainingssplits (jeder 12. Eintrag, Intervall 0),
    # damit die Epochenzeit nicht explodiert. Val/Test bleiben unberührt.
    if C.AUGMENT_PHRASES and train_with_transposed_chorales:
        extra = []
        for k in range(6, len(train_data), 12):
            extra.extend(split_into_phrases(*train_data[k]))
        print(f"Phrasen-Augmentierung: +{len(extra)} Beispiele")
        train_data = train_data + extra

    # Fester Seed: bei Resume eines abgebrochenen Trainings spult der Trainer
    # die Dataloader-Reihenfolge vor, um bereits gesehene Batches zu
    # überspringen — das ist nur mit reproduzierbarem Shuffle korrekt.
    train_dataset = Dataset.from_dict({
        "source": [src for src, tgt in train_data],
        "target": [tgt for src, tgt in train_data]
    }).shuffle(seed=42)

    val_dataset = Dataset.from_dict({
        "source": [src for src, tgt in val_data],
        "target": [tgt for src, tgt in val_data]
    }).shuffle(seed=42)

    # ====== TOKENISIERUNG ======
    def tokenize_function(examples):
        """Tokenisiert mit MusicTokenizer; Padding macht später der DataCollator."""
        source_tokens = [music_tokenizer.encode(src) for src in examples["source"]]
        target_tokens = [music_tokenizer.encode(tgt) for tgt in examples["target"]]
        # Entferne erstes CLS aus Target (Decoder-Start setzt der Collator)
        target_tokens = [t[1:] for t in target_tokens]
        return {"input_ids": source_tokens, "labels": target_tokens}

    print("Tokenisiere Trainingsdaten...")
    tokenized_train = train_dataset.map(
        tokenize_function, batched=True,
        remove_columns=["source", "target"], desc="Tokenizing train",
    )

    print("Tokenisiere Validierungsdaten...")
    tokenized_val = val_dataset.map(
        tokenize_function, batched=True,
        remove_columns=["source", "target"], desc="Tokenizing val",
    )

    return tokenized_train, tokenized_val, raw_dataset, test_data, music_tokenizer


def build_voice_token_sets(raw_dataset, music_tokenizer, num_lower=C.NUM_VOICES - 1):
    """Erlaubte Tokens pro Unterstimme, aus den Daten gelernt.

    Register-Constraints wie in DeepBach (eigenes Vokabular pro Stimme):
    Alt/Tenor/Bass dürfen bei der Generierung nur Tokens erzeugen, die im
    Datensatz in genau dieser Stimme vorkommen (inkl. '_' und 'R').
    Verhindert Stimmkreuzungen und Registerfehler.
    """
    sets = [set() for _ in range(num_lower)]
    for _, tgt in raw_dataset:
        slot = 0  # 0 = Sopran-Slot, 1..3 = Alt/Tenor/Bass
        for tok in tgt.split():
            if tok == '|':
                continue
            if tok == ';':
                slot = 0
                continue
            if 1 <= slot <= num_lower:
                sets[slot - 1].add(tok)
            slot += 1
    return [music_tokenizer.convert_tokens_to_ids(sorted(s)) for s in sets]
