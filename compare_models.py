"""A/B-Vergleich zweier trainierter Modelle auf den Val-Chorälen.

Beide Modelle durchlaufen die volle Generierungspipeline (phrasenweises
Best-of-N, Reparatur, alle Prozessoren, gleicher Seed). Verglichen werden
modellunabhängige Größen:
  - gewichteter Regel-Score pro 100 Sechzehntel (rule_violation_score)
  - Übereinstimmung der frei generierten A/T/B-Töne mit Bachs Original
    (alle Slots bzw. nur an Bachs Note-Onsets)

Aufruf:
    python compare_models.py MODELL_DIR_A [MODELL_DIR_B ...]
    python compare_models.py            # nur config.MODEL_DIR
"""

import statistics
import sys

import torch
from transformers import T5ForConditionalGeneration

import config as C
from data_prep import build_voice_token_sets, load_raw_dataset
from generation import ChoraleHarmonizer, rule_violation_score
from music_tokenizer import MusicTokenizer


def groups_of(s):
    gs, g = [], []
    for t in s.split():
        if t == '|':
            continue
        if t == ';':
            if len(g) == 4:
                gs.append(g)
            g = []
        else:
            g.append(t)
    return gs


def atb_match(out, ref):
    """Anteil übereinstimmender A/T/B-Token (alle Slots, nur Bach-Onsets)."""
    go, gr = groups_of(out), groups_of(ref)
    n = min(len(go), len(gr))
    all_m = all_t = on_m = on_t = 0
    for i in range(n):
        for v in (1, 2, 3):
            a, b = go[i][v], gr[i][v]
            all_t += 1
            all_m += (a == b)
            if b != '_':
                on_t += 1
                on_m += (a == b)
    return all_m / max(1, all_t), on_m / max(1, on_t)


def val_pairs(raw):
    """Die Val-Choräle in Originaltonart (Quelle, Bach-Original)."""
    n_hold = (C.VAL_CHORALES + C.TEST_CHORALES) * 12
    val = raw[len(raw) - n_hold: len(raw) - C.TEST_CHORALES * 12]
    return [val[i] for i in range(6, len(val), 12)]


def num_steps(src):
    return max(1, len([t for t in src.split()
                       if t != '|' and '/' not in t and ':' not in t]))


def evaluate(model_dir, pairs, tokenizer, voice_ids, device):
    torch.manual_seed(0)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device).eval()
    harmonizer = ChoraleHarmonizer(model, tokenizer, voice_ids)
    scores, matches_all, matches_on = [], [], []
    for src, ref in pairs:
        out = harmonizer.transform(src)
        scores.append(100 * rule_violation_score(out.split()) / num_steps(src))
        ma, mo = atb_match(out, ref)
        matches_all.append(ma)
        matches_on.append(mo)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return scores, matches_all, matches_on


def main():
    model_dirs = sys.argv[1:] or [C.MODEL_DIR]
    raw = load_raw_dataset()
    tokenizer = MusicTokenizer()
    tokenizer.build_vocab(raw)
    voice_ids = build_voice_token_sets(raw, tokenizer)
    pairs = val_pairs(raw)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Val-Choräle (Originaltonart): {len(pairs)}\n")

    for d in model_dirs:
        scores, m_all, m_on = evaluate(d, pairs, tokenizer, voice_ids, device)
        print(f"=== {d} ===")
        print(f"Regel-Score/100:   median {statistics.median(scores):.2f}  "
              f"mean {statistics.mean(scores):.2f}")
        print(f"A/T/B-Match alle:  {statistics.mean(m_all):.3f}")
        print(f"A/T/B-Match Onset: {statistics.mean(m_on):.3f}\n")


if __name__ == '__main__':
    main()