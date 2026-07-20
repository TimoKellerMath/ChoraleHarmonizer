"""ChoraleHarmonizer — Einstiegspunkt.

Harmonisiert Sopranstimmen im Stil von Bach-Chorälen (Alt, Tenor, Bass).

Ablauf:
  1. Daten einlesen und tokenisieren        (data_prep.py, music_tokenizer.py)
  2. Modell laden oder trainieren           (training.py)
  3. Testchoräle harmonisieren + exportieren (generation.py, bach_chorales.py)

Vorher einmalig den Datensatz erzeugen: python bach_chorales.py
Alle Stellschrauben: config.py
"""

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


def main():
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

    voice_allowed_ids = (build_voice_token_sets(raw_dataset, music_tokenizer)
                         if C.NUM_VOICES > 2 else None)
    harmonizer = ChoraleHarmonizer(model, music_tokenizer, voice_allowed_ids)
    write_test_outputs(harmonizer, test_data)


if __name__ == "__main__":
    main()
