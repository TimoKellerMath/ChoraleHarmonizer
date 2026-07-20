# ChoraleHarmonizer

Four-part chorale harmonization in the style of J. S. Bach: given a soprano
melody, the system writes alto, tenor and bass. A small sequence-to-sequence
transformer (trained from scratch on the 343 four-part chorales of the
music21 corpus) is combined with hard structural constraints, targeted
sampling-time penalties, and a best-of-N search guided by a **data-calibrated
voice-leading rule engine** — every rule weight and every exception is
measured against Bach's own practice, not taken from a textbook.

Output is written as MIDI and MusicXML, side by side with Bach's original
setting of the same melody.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Build the dataset from the music21 Bach corpus (one-time, ~5 min CPU)
python bach_chorales.py

# 2. Train (~2-3 h on a consumer GPU) and harmonize the held-out test chorales
python chorale_harmonizer.py
```

Results land in `output/`: for each test chorale `<n>-Bach.{mid,mxl}`
(original) and `<n>-ChoraleHarmonizer.{mid,mxl}` (generated). If a trained
model already exists in `MODEL_DIR`, training is skipped and the script only
generates; delete the directory or set `FORCE_RETRAIN = True` in `config.py`
to retrain. An interrupted training run resumes automatically from its last
checkpoint. All knobs live in `config.py`.

Run the test suite (no GPU or model needed):

```bash
python tests.py
```

## How it works

**Data (format v2.2).** Chorales are flattened onto a sixteenth-note grid.
The source is the soprano with a time-signature and a key token in front;
the target interleaves all four voices per sixteenth step:

```
source:  4/4 G:major F4 _ _ _ | A4 _ _ _ G4 _ _ _ ... A4^ _ _ _ ...
target:  F4 C4 A3 F2 ; _ _ _ _ ; ... | A4 F4 C4 F3 ; ...
```

`_` holds the previous note, `|` marks barlines, `^` a fermata, `R` a rest.
Each chorale is used in 12 transpositions; training additionally sees each
phrase (between fermatas) as its own sample, and 10% of the decoder-input
tokens are randomly corrupted during training so the model learns to recover
from its own mistakes at generation time (exposure-bias mitigation).

**Model.** A deliberately small from-scratch T5 (d_model 256, 6+6 layers,
relative attention, ~15 M parameters) — after swapping the vocabulary for
~120 music tokens, nothing useful survives from text pretraining, and a
small model generalizes better on ~6 k samples. The loss is a
position-weighted cross-entropy: structure tokens count 0, the forced
soprano copies 0.1, so the gradient concentrates on the actual A/T/B
decisions. Checkpoints are selected by *note accuracy* (correct real note
decisions, excluding holds/structure/soprano) instead of loss.

**Generation.** Constrained sampling inside a search:

1. *Hard constraints* — soprano, barlines and group separators are forced;
   alto/tenor/bass are restricted to the registers observed for that voice
   in the corpus; at fermatas every voice must articulate the cadence chord
   (Bach does so at 99.3–99.7% of fermatas — measured).
2. *Sampling-time penalties* — a targeted processor against re-striking the
   currently sounding pitch in the middle voices, and a parallel-fifths/
   octaves check that penalizes exactly the pitches which would continue a
   perfect interval in parallel (this alone reduced the model's parallel
   rate from 5.08 to 0.22 per 100 sixteenths; Bach's own rate is 0.18).
3. *Phrase-wise best-of-N* — the piece is segmented at fermatas; per phrase,
   `GEN_CANDIDATES` continuations are sampled and the best one is fixed
   before the next phrase is generated: N^phrases combinations for the price
   of N×phrases.
4. *Repair loop* — the heaviest remaining rule violation is located and the
   piece is resampled from just before it; changes are kept only if the
   global score improves.

**Rule engine.** Twenty voice-leading rules are evaluated on the token grid,
including multi-step rules (accent parallels between consecutive beats,
leap recovery, tritone resolution, harmonic stagnation) and chord rules
(empty sonorities, doubled major thirds). The scoring is calibrated on data
rather than dogma:

- *Weights are log-ratios between the model's and Bach's violation rates*
  (`measure_rule_weights.py`). Rules the constraints already push below
  Bach's own rate get weight 0 — punishing them would trade against real
  problems. Re-measure after every retraining.
- *Exceptions are discovered, not assumed*: rules are split where Bach
  treats variants differently (resolved vs. unresolved tritone leaps,
  small vs. large voice overlaps, tenor–bass vs. upper-voice unisons).
- *Pair idioms*: combinations that co-occur in Bach far more often than
  chance (e.g. hidden fifth + similar motion, lift 39.7) are treated as one
  incident — the cheaper violation is waived instead of double-counted.
  Generic cascades are damped (heaviest violation full, rest half).

## What's new compared to the reference projects

Both reference implementations are included as submodule-style copies for
study (`DeepBach/`, `constraint-transformer-bach/`).

| | DeepBach (Hadjeres et al.) | constraint-transformer-bach | **ChoraleHarmonizer** |
|---|---|---|---|
| Generation | iterative pseudo-Gibbs resampling | causal transformer with constraints | single-pass seq2seq + phrase-wise best-of-N + repair |
| Melody context | local windows around each timestep | past context of the interleaved stream | full melody in a bidirectional encoder — the decoder "sees" upcoming fermatas and cadences |
| Metadata | separate metadata channels (beat, fermata) | token stream | meter, key and fermatas as plain tokens in one vocabulary |
| Voice leading | implicit, learned only | implicit, learned only | explicit 20-rule engine, **calibrated on measured Bach vs. model rates** (log-ratio weights, discovered exceptions, pair idioms) |
| Sampling guards | — | hard constraints | hard constraints **plus** targeted logit penalties (repeat notes, parallels) that leave holds and passing tones untouched |
| Evaluation | listening | loss/accuracy | teacher-forced note accuracy **and** free-generation metrics (rule score, agreement with Bach) via `compare_models.py` |

The central idea beyond the references: **treat the style rules as a
measured, model-relative quantity.** Instead of penalizing textbook sins,
the system measures where the *current model* actually deviates from Bach
and concentrates all penalty mass exactly there — and it learns from Bach
which "violations" are in fact idioms that must stay unpunished.

## Tools

| Command | Purpose |
|---|---|
| `python measure_rule_weights.py [N] [K]` | sample raw model outputs, compare per-rule violation rates against Bach, print a ready-to-paste `RULE_WEIGHTS` proposal (rerun after each retraining) |
| `python compare_models.py DIR_A DIR_B` | free-generation A/B of two trained models on the validation chorales (rule score + agreement with Bach) |
| `python self_distill.py generate` / `train` | RAFT-style self-distillation: run the full search pipeline over the training melodies, then fine-tune the model on its own best outputs (with Bach originals as drift anchors) |

## Tuning

The most useful knobs in `config.py`:

- `GEN_CANDIDATES` — search width per phrase (quality vs. generation time)
- `GEN_TEMPERATURE`, `GEN_TOP_P` — stricter vs. more adventurous writing
- `UNDERSCORE_BIAS` — < 0 yields more passing-tone motion
- `RULE_PENALTY_WEIGHT`, `PARALLEL_PENALTY`, `REPEAT_NOTE_PENALTY` — rule
  engine influence at reranking / sampling time
- `GEN_PHRASEWISE`, `GEN_REPAIR_ITERS`, `FERMATA_FORCE_ONSET` — pipeline
  stages, individually switchable for ablations

## Project structure

| File | Role |
|---|---|
| `bach_chorales.py` | I/O: music21 corpus → token dataset; token sequences → MusicXML/MIDI |
| `chorale_harmonizer.py` | entry point: data → training → generation |
| `config.py` | every path and hyperparameter |
| `music_tokenizer.py` | token strings ↔ ids (HuggingFace-compatible) |
| `data_prep.py` | dataset loading, splits, tokenization, phrase augmentation, per-voice registers |
| `training.py` | model construction, weighted loss, metrics, trainer, checkpoint resume |
| `generation.py` | rule engine, logit processors, constrained search, `ChoraleHarmonizer` |
| `tests.py` | 31 tests: rules, calibration against Bach, token handling, export |

## Results (reproducible)

- Teacher-forced note accuracy: **0.623** on held-out chorales (plateau —
  confirmed by capacity experiments; given a soprano there are many valid
  harmonizations, so this is close to the intrinsic ceiling).
- Parallel fifths/octaves in raw samples: **5.08 → 0.22** per 100 sixteenths
  with the sampling-time check (Bach: 0.18).
- Weighted rule score of full-pipeline output: **median ≈ 1.2–1.3** per 100
  sixteenths on validation melodies — Bach's own settings measure 1.74
  under the same rules.