"""Selbst-Destillation: die Suche ins Modell zurücktrainieren.

Idee (RAFT/Best-of-N-Destillation statt vollem RL): Die Generierungspipeline
(phrasenweises Best-of-N + Reparatur + Regel-Score) produziert deutlich
bessere Harmonisierungen, als das Modell in einem einzelnen Sample liefert.
Diese Pipeline-Outputs werden als zusätzliche Trainingsziele verwendet —
das Modell lernt, von sich aus dorthin zu samplen, wo die Suche es bisher
hintragen musste. Das optimiert erstmals direkt die freie Generierung
(Teacher-Forced note_accuracy ist bei ~0.62 gesättigt und misst das nicht).

Zwei Phasen:

  python self_distill.py generate [N]   # Pipeline-Outputs für N (default alle)
                                        # Train-Choräle erzeugen; resumefähig,
                                        # Qualitätsfilter DISTILL_MAX_RULE_SCORE
  python self_distill.py train [--dry]  # Feintuning MODEL_DIR -> DISTILL_MODEL_DIR
                                        # auf Mischung aus destillierten Paaren
                                        # (x DISTILL_REPEAT) und Bach-Originalen
                                        # derselben Choräle; wenige Epochen,
                                        # kleine Lernrate, Val unverändert

Danach auswerten:
  python compare_models.py music_transformer_satb-key music_transformer_satb-key-sd

Risiken und Gegenmittel: Drift in Eigenheiten des Modells (-> Mischung mit
Bach-Originalen, kleine Lernrate, wenige Epochen), Reward-Hacking auf den
Regel-Score (-> Ziele sind komplette Pipeline-Outputs inkl. Likelihood,
nicht nur regelminimale), Qualitätsausreißer (-> Filter).
"""

import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import T5ForConditionalGeneration

import config as C
from data_prep import build_voice_token_sets, load_or_compute_dataset, load_raw_dataset
from generation import ChoraleHarmonizer, rule_violation_score
from music_tokenizer import MusicTokenizer
from training import make_trainer, train_and_save


def num_steps(src):
    return max(1, len([t for t in src.split()
                       if t != '|' and '/' not in t and ':' not in t]))


def train_sources(raw):
    """Train-Choräle in Originaltonart (Val/Test liegen am Datenende)."""
    n_hold = (C.VAL_CHORALES + C.TEST_CHORALES) * 12
    train = raw[:len(raw) - n_hold]
    return [train[i] for i in range(6, len(train), 12)]


def generate_phase(limit=0):
    raw = load_raw_dataset()
    tokenizer = MusicTokenizer()
    tokenizer.build_vocab(raw)

    print(f"Lade Modell aus {C.MODEL_DIR} ...")
    model = T5ForConditionalGeneration.from_pretrained(C.MODEL_DIR)
    model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    harmonizer = ChoraleHarmonizer(model, tokenizer, build_voice_token_sets(raw, tokenizer))

    sources = train_sources(raw)
    if limit:
        sources = sources[:limit]

    # Resume: bereits destillierte Quellen überspringen
    entries = []
    if Path(C.DISTILL_FILE).exists():
        entries = json.load(open(C.DISTILL_FILE))
        print(f"↻ {len(entries)} vorhandene Einträge geladen (Resume)")
    done = {e['src'] for e in entries}

    kept = sum(1 for e in entries if e['score'] <= C.DISTILL_MAX_RULE_SCORE)
    with torch.no_grad():
        for i, (src, _bach) in enumerate(sources, 1):
            if src in done:
                continue
            out = harmonizer.transform(src)
            score = 100 * rule_violation_score(out.split()) / num_steps(src)
            entries.append({'src': src, 'out': out, 'score': round(score, 2)})
            if score <= C.DISTILL_MAX_RULE_SCORE:
                kept += 1
            print(f"[{i}/{len(sources)}] Regel-Score {score:.1f} "
                  f"{'✓' if score <= C.DISTILL_MAX_RULE_SCORE else '✗ (wird beim Training gefiltert)'}")
            json.dump(entries, open(C.DISTILL_FILE, 'w'))
    print(f"\n{len(entries)} Outputs in {C.DISTILL_FILE}, "
          f"davon {kept} unter der Qualitätsschwelle {C.DISTILL_MAX_RULE_SCORE}")


def tokenize_pairs(pairs, tokenizer):
    return Dataset.from_dict({
        'input_ids': [tokenizer.encode(src) for src, _ in pairs],
        'labels': [tokenizer.encode(tgt)[1:] for _, tgt in pairs],
    })


def train_phase(dry=False):
    if not Path(C.DISTILL_FILE).exists():
        raise FileNotFoundError(f"{C.DISTILL_FILE} fehlt — erst: python self_distill.py generate")
    entries = json.load(open(C.DISTILL_FILE))
    distilled = [(e['src'], e['out']) for e in entries
                 if e['score'] <= C.DISTILL_MAX_RULE_SCORE]

    # Val-Split und Tokenizer exakt wie im Haupttraining
    _, tokenized_val, raw, _, tokenizer = load_or_compute_dataset()

    # Bach-Originale derselben Choräle als Anker gegen Drift
    bach_by_src = {src: tgt for src, tgt in train_sources(raw)}
    anchors = [(src, bach_by_src[src]) for src, _ in distilled if src in bach_by_src]

    mixture = distilled * C.DISTILL_REPEAT + anchors
    print(f"Feintuning-Daten: {len(distilled)} destilliert x{C.DISTILL_REPEAT} "
          f"+ {len(anchors)} Bach-Anker = {len(mixture)}")
    tokenized_mix = tokenize_pairs(mixture, tokenizer).shuffle(seed=42)

    print(f"Lade Basis-Modell aus {C.MODEL_DIR} ...")
    model = T5ForConditionalGeneration.from_pretrained(C.MODEL_DIR)

    # Trainer-Konfiguration über config umbiegen: kleines LR, wenige Epochen,
    # eigenes Ausgabeverzeichnis — das Basis-Modell bleibt unangetastet
    saved = (C.MODEL_DIR, C.NUM_EPOCHS, C.LEARNING_RATE, C.WARMUP_STEPS)
    C.MODEL_DIR = C.DISTILL_MODEL_DIR
    C.NUM_EPOCHS = C.DISTILL_EPOCHS
    C.LEARNING_RATE = C.DISTILL_LR
    C.WARMUP_STEPS = C.DISTILL_WARMUP
    try:
        trainer = make_trainer(model, tokenizer, tokenized_mix, tokenized_val)
        if dry:
            print("--dry: Trainer gebaut, kein Training gestartet")
            return
        train_and_save(trainer, tokenizer)
        print(f"\n✓ Destilliertes Modell in {C.DISTILL_MODEL_DIR}")
        print("Auswerten: python compare_models.py "
              f"{saved[0]} {C.DISTILL_MODEL_DIR}")
    finally:
        C.MODEL_DIR, C.NUM_EPOCHS, C.LEARNING_RATE, C.WARMUP_STEPS = saved


def main():
    args = sys.argv[1:]
    if not args or args[0] not in ('generate', 'train'):
        print(__doc__)
        sys.exit(1)
    if args[0] == 'generate':
        limit = int(args[1]) if len(args) > 1 else 0
        generate_phase(limit)
    else:
        train_phase(dry='--dry' in args)


if __name__ == '__main__':
    main()