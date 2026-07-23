"""ChoraleHarmonizer — Einstiegspunkt.

Harmonisiert Sopranstimmen im Stil von Bach-Chorälen (Alt, Tenor, Bass).

Ablauf:
  1. Daten einlesen und tokenisieren        (data_prep.py, music_tokenizer.py)
  2. Modell laden oder trainieren           (training.py)
  3. Choräle harmonisieren + exportieren    (generation.py, bach_chorales.py)

Aufruf:
  python chorale_harmonizer.py                          # 5 Testchoräle, Standard-Stil
  python chorale_harmonizer.py --num-chorales 10        # mehr Choräle (Test + Val)
  python chorale_harmonizer.py --presets konservativ,kuehn
                                                        # je Choral eine Fassung pro Stil-Preset
Vorher einmalig den Datensatz erzeugen: python bach_chorales.py
Alle Stellschrauben: config.py und style_config.json
"""

import argparse
import os
import warnings

import torch
from datasets import Dataset

print("CUDA is available:", torch.cuda.is_available())

warnings.filterwarnings("ignore", category=UserWarning, message=".*pin_memory.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*encoder-decoder.*")

if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = ""

torch.serialization.add_safe_globals([Dataset])

import config as C
from data_prep import build_voice_token_sets, load_or_compute_dataset
from generation import ChoraleHarmonizer, write_test_outputs
from training import build_model, make_trainer, train_and_save


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Bach-Choral-Harmonisierung")
    parser.add_argument("--num-chorales", type=int, default=C.TEST_CHORALES,
                        help="Anzahl zu harmonisierender Choräle (Test-, dann "
                             f"Val-Choräle; max {C.TEST_CHORALES + C.VAL_CHORALES})")
    parser.add_argument("--presets", type=str, default="",
                        help="Kommagetrennte Stil-Presets aus style_config.json "
                             "(z.B. konservativ,ausgewogen,kuehn); je Choral "
                             "entsteht eine Fassung pro Preset")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    presets = [p.strip() for p in args.presets.split(',') if p.strip()] or None

    os.makedirs(C.DATA_DIR, exist_ok=True)
    os.makedirs(C.TOKENIZER_DIR, exist_ok=True)

    # ====== 1. DATEN ======
    tokenized_train, tokenized_val, raw_dataset, test_data, music_tokenizer = load_or_compute_dataset()
    print(f"Train: {len(tokenized_train)}, Val: {len(tokenized_val)}")

    # ====== 2. MODELL / TRAINING ======
    model, need_to_train = build_model(music_tokenizer)
    # Trainer auch ohne Training bauen: er übernimmt die Geräteplatzierung
    trainer = make_trainer(model, music_tokenizer, tokenized_train, tokenized_val)
    if need_to_train:
        train_and_save(trainer, music_tokenizer)

    # ====== 3. GENERIERUNG ======
    print("\n" + "=" * 80)
    print("TESTE")
    print("=" * 80)

    # Kandidaten-Pool in Originaltonart: erst Test-, dann Val-Choräle
    n_holdout = (C.VAL_CHORALES + C.TEST_CHORALES) * 12
    val_block = raw_dataset[len(raw_dataset) - n_holdout:
                            len(raw_dataset) - C.TEST_CHORALES * 12]
    pool = ([test_data[i] for i in range(6, len(test_data), 12)]
            + [val_block[i] for i in range(6, len(val_block), 12)])
    pairs = pool[:max(1, args.num_chorales)]

    voice_allowed_ids = (build_voice_token_sets(raw_dataset, music_tokenizer)
                         if C.NUM_VOICES > 2 else None)
    harmonizer = ChoraleHarmonizer(model, music_tokenizer, voice_allowed_ids)
    write_test_outputs(harmonizer, pairs, presets=presets)


if __name__ == "__main__":
    main()