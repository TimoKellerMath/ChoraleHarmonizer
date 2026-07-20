"""Messskript für Log-Ratio-Regelgewichte.

Bestimmt, wie oft das trainierte Modell (Rohproben OHNE Reranking/Reparatur)
gegen jede Satzregel verstößt, vergleicht mit Bachs eigenen Raten und schlägt
Gewichte als geclippte Log-Ratios vor:

    w_r = clip( log( (Rate_Modell_r + eps) / (Rate_Bach_r + eps) ), 0, W_MAX )

Begründung: Der Regel-Malus soll die Ausgabeverteilung des Modells Richtung
Bach schieben (Naive-Bayes-Sicht "Bach vs. Modell" über Poisson-Zählungen).
Eine Regel, die das Modell ohnehin nie verletzt, braucht kein Gewicht — egal
wie selten sie bei Bach ist; am stärksten zählt, was das Modell oft und Bach
selten tut.

Aufruf (nach dem Training):
    python measure_rule_weights.py [N_CHORALES] [SAMPLES_PER_CHORAL]

Ausgabe: Vergleichstabelle und ein fertiger RULE_WEIGHTS-Vorschlag zum
Einfügen in generation.py. Es wird nichts automatisch geändert.
"""

import math
import sys

import torch
from transformers import T5ForConditionalGeneration

import config as C
import generation
from data_prep import build_voice_token_sets, load_raw_dataset
from generation import (RULE_NAMES, RULE_WEIGHTS, ChoraleHarmonizer,
                        count_rule_violations_by_rule, rule_violation_score)
from music_tokenizer import MusicTokenizer

N_CHORALES = int(sys.argv[1]) if len(sys.argv) > 1 else 25
SAMPLES_PER = int(sys.argv[2]) if len(sys.argv) > 2 else 2
EPS = 0.02   # Glättung in "Verstöße pro 100 Sechzehntel"
W_MAX = 8.0  # Obergrenze (Bach-Rate 0 ergäbe sonst unendlich)


def steps_of(src):
    return len([t for t in src.split() if t != '|' and '/' not in t and ':' not in t])


def main():
    raw = load_raw_dataset()
    tokenizer = MusicTokenizer()
    tokenizer.build_vocab(raw)

    print(f"Lade Modell aus {C.MODEL_DIR} ...")
    model = T5ForConditionalGeneration.from_pretrained(C.MODEL_DIR)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    harmonizer = ChoraleHarmonizer(model, tokenizer, build_voice_token_sets(raw, tokenizer))

    # ==== Modellraten: Rohproben ohne Best-of-N/Reparatur ====
    # GEN_CANDIDATES=1 nimmt in transform() den Einzelpfad (kein Reranking,
    # keine Reparatur) — Constraints und Logit-Prozessoren bleiben aktiv,
    # denn sie sind Teil des Systems, das der Regel-Malus korrigieren soll.
    saved = (C.GEN_CANDIDATES, C.GEN_REPAIR_ITERS)
    C.GEN_CANDIDATES, C.GEN_REPAIR_ITERS = 1, 0

    # Trainings-Soprane in Originaltonart (Val/Test liegen am Datenende)
    n_holdout = (C.VAL_CHORALES + C.TEST_CHORALES) * 12
    train = raw[:len(raw) - n_holdout]
    sources = [train[i][0] for i in range(6, len(train), 12)][:N_CHORALES]

    model_counts = {r: 0 for r in RULE_NAMES}
    model_steps = 0
    with torch.no_grad():
        for i, src in enumerate(sources, 1):
            for _ in range(SAMPLES_PER):
                out = harmonizer.transform(src)
                for r, n in count_rule_violations_by_rule(out.split()).items():
                    model_counts[r] += n
                model_steps += steps_of(src)
            print(f"  {i}/{len(sources)} Choräle gesampelt ({SAMPLES_PER}x)", end='\r')
    print()
    C.GEN_CANDIDATES, C.GEN_REPAIR_ITERS = saved

    # ==== Bach-Raten (alle Originaltonarten) ====
    bach_counts = {r: 0 for r in RULE_NAMES}
    bach_steps = 0
    for idx in range(6, len(raw), 12):
        src, tgt = raw[idx]
        bach_steps += steps_of(src)
        for r, n in count_rule_violations_by_rule(tgt.split()).items():
            bach_counts[r] += n

    # ==== Vorschlag ====
    suggested = {}
    rows = []
    for r in RULE_NAMES:
        b = 100 * bach_counts[r] / bach_steps
        m = 100 * model_counts[r] / model_steps
        w = min(max(math.log((m + EPS) / (b + EPS)), 0.0), W_MAX)
        suggested[r] = round(w, 1)
        rows.append((r, b, m, w))

    print(f"\nModellproben: {len(sources)} Choräle x {SAMPLES_PER} = "
          f"{model_steps} Sechzehntel | Bach: {bach_steps} Sechzehntel")
    print(f"\n{'Regel':24s} {'Bach/100':>9s} {'Modell/100':>11s} {'log-Ratio':>10s} {'aktuell':>8s}")
    for r, b, m, w in sorted(rows, key=lambda x: -x[3]):
        print(f"{r:24s} {b:9.3f} {m:11.3f} {w:10.1f} {RULE_WEIGHTS.get(r, 1.0):8.1f}")

    print("\n# Vorschlag zum Einfügen in generation.py "
          "(Kommentar: Bach-Rate / Modell-Rate pro 100 Sechzehntel):")
    print("RULE_WEIGHTS = {")
    for r, b, m, w in sorted(rows, key=lambda x: -x[3]):
        print(f"    '{r}': {w:4.1f},  # Bach {b:.2f} / Modell {m:.2f}")
    print("}")

    # Einordnung: Bach-Baseline unter den vorgeschlagenen Gewichten (mit
    # Kaskaden-Dämpfung), als Anhaltspunkt für RULE_PENALTY_WEIGHT
    saved_weights = dict(RULE_WEIGHTS)
    RULE_WEIGHTS.update(suggested)
    try:
        import statistics
        per100 = []
        for idx in range(6, len(raw), 12):
            src, tgt = raw[idx]
            per100.append(100 * rule_violation_score(tgt.split()) / max(1, steps_of(src)))
        print(f"\nBach-Baseline unter den vorgeschlagenen Gewichten: "
              f"median {statistics.median(per100):.1f} pro 100 Sechzehntel")
        print("(Kalibrierungstest in tests.py nach Übernahme entsprechend anpassen;")
        print(" RULE_PENALTY_WEIGHT ggf. neu abstimmen, die Skala ändert sich.)")
    finally:
        RULE_WEIGHTS.clear()
        RULE_WEIGHTS.update(saved_weights)


if __name__ == "__main__":
    main()