"""Zentrale Konfiguration für ChoraleHarmonizer.

Alle Pfade und Hyperparameter für Datenaufbereitung, Training und Generierung.
"""

# ====== PFADE ======
MODEL_DIR = "./music_transformer_satb-key"  # v2.2-Modell (note_acc 0.623); das Kapazitäts-
                                            # Experiment ...-key-h8 war gleichauf (siehe SCRATCH_HEADS)
TOKENIZER_DIR = "./music_tokenizer"
DATA_DIR = "./bach_data_cache"
RAW_DATA_FILE = "bach-chorales/data-3-voices-False-satb-ts-key"  # Format v2.2, erzeugen mit: python bach_chorales.py
OUTPUT_DIR = "output"

# ====== DATEN / TRAINING ======
NUM_VOICES = 4
TRAIN_BATCH_SIZE = 8 # 16 for one voice, 4 for 3 voices
GRAD_ACCUM_STEPS = 2
VAL_BATCH_SIZE = 8 # 32 is too large
NUM_EPOCHS = 60  # 25->40 Epochen brachte +0.10 note_accuracy (0.49 -> 0.60); Kurve war noch nicht gesättigt
LEARNING_RATE = 3e-4  # from-scratch Modell verträgt/braucht mehr als feingetuntes t5-small
WARMUP_STEPS = 200
LABEL_SMOOTHING = 0.05    # 0.1 drückte das Gradientensignal der harten Noten-Tokens zu stark
SOPRANO_SLOT_WEIGHT = 0.1 # CE-Gewicht der Sopran-Kopien im Target: bei Generierung forciert,
                          # sollen den Loss nicht dominieren (nur als Alignment-Hilfssignal)
FORCE_RETRAIN = False  # True: Training neu starten, auch wenn Modell bereits existiert
AUGMENT_PHRASES = True # Phrasen (zwischen Fermaten) zusätzlich als eigene Trainingsbeispiele —
                       # nur aus den Originaltonarten des Trainingssplits, damit die
                       # Epochenzeit nicht explodiert; Kadenzen werden konzentrierter gelernt
DECODER_INPUT_DROPOUT = 0.1  # Anteil der A/T/B-Tokens im Decoder-Input, die beim Training
                             # durch Zufallstoken ersetzt werden (Exposure Bias: das Modell
                             # lernt, sich von eigenen Fehlern zu erholen). 0 = aus.
VAL_CHORALES = 10   # number of validation chorales (× 12 transpositions if augmented)
TEST_CHORALES = 5   # number of test chorales (× 12 transpositions if augmented)

# ====== MODELL ======
# t5-small ist auf englischen Text vortrainiert; nach dem Embedding-Resize auf
# ~60 Musik-Tokens bleibt davon wenig Nützliches, aber 60M Parameter Overfitting-
# Risiko. Ein kleines, von Grund auf trainiertes T5 mit relativer Attention
# (wie im constraint-transformer) generalisiert auf ~2500 Chorälen besser.
TRAIN_FROM_SCRATCH = True
SCRATCH_D_MODEL = 256
SCRATCH_D_FF = 1024
SCRATCH_LAYERS = 6      # je Encoder und Decoder
SCRATCH_HEADS = 4       # Kapazitäts-Experiment 8 Heads + rel-attn 512/64 brachte nichts:
                        # note_acc 0.6224 vs 0.6230, A/B auf Val-Chorälen gleichauf —
                        # Architektur ist nicht der Engpass (Modell dazu: ...-key-h8)
SCRATCH_DROPOUT = 0.15  # wichtigste Regularisierung gegen Auswendiglernen
# T5-Relativattention: Buckets/Reichweite so, dass die 5er-Periode (S A T B ;),
# ganze Takte (~80 Tokens) UND Phrasenlängen (~300-500 Tokens) aufgelöst werden
REL_ATTN_NUM_BUCKETS = 32
REL_ATTN_MAX_DISTANCE = 256

# ====== GENERIERUNG ======
# Keine repetition_penalty / no_repeat_ngram: in dieser Darstellung sind
# Wiederholungen ('_'-Haltungen, wiederkehrende Tonhöhen) musikalisch korrekt.
# Eine globale Penalty verzerrt genau die Statistik von Haltungen und
# Durchgangsnoten (=> unbachische Mittelstimmen und Bässe).
GEN_TEMPERATURE = 0.75
GEN_TOP_P = 0.92
UNDERSCORE_BIAS = 0.0   # <0: mehr Note-Onsets (Durchgangsnoten), >0: mehr Haltungen; 0: den Daten vertrauen
GEN_CANDIDATES = 32      # Best-of-N: Kandidaten samplen, nach A/T/B-Likelihood reranken (1 = aus).
                         # 32 statt 16: Modellseite ist ausgereizt (note_acc-Plateau 0.62),
                         # Qualität kommt jetzt aus der Suche — kostet nur Generierzeit.
REPEAT_NOTE_PENALTY = 1.5  # Logit-Malus, wenn Alt/Tenor den gerade klingenden Ton ERNEUT anschlagen
                           # würden (statt zu halten) — gezielt gegen repetierende Mittelstimmen;
                           # Durchgangsnoten (andere Tonhöhen) sind nicht betroffen. 0 = aus.
PARALLEL_PENALTY = 6.0     # Logit-Malus beim Sampling eines A/T/B-Slots für Tonhöhen, die mit
                           # einer bereits festgelegten Stimme der Gruppe eine parallele
                           # Quinte/Oktave bilden würden (gemessene Kernschwäche des Modells).
                           # Kein hartes Verbot: Bach erlaubt sich Parallelen selten (0.18/100),
                           # der Malus lässt sie nur gegen sehr starke Modell-Präferenz zu. 0 = aus.
RULE_PENALTY_WEIGHT = 2.0  # Musiktheorie-Malus im Best-of-N-Reranking (0 = aus): Score =
                           # mittlere A/T/B-Log-Likelihood − Gewicht × gewichtete Regelverstöße/16tel
                           # (Regelkatalog und Bach-kalibrierte Gewichte: generation.RULE_WEIGHTS).
GEN_PHRASEWISE = True      # Best-of-N pro Phrase (an Fermaten segmentiert) statt pro Gesamtstück:
                           # beste Phrase wird fixiert, dann die nächste gesampelt — effektiv
                           # N^Phrasen Kombinationen zum Preis von N×Phrasen.
GEN_REPAIR_ITERS = 2       # Reparatur-Schleife: ab der Stelle des schwersten Regelverstoßes
                           # neu samplen und nur bei Verbesserung übernehmen (0 = aus).
FERMATA_FORCE_ONSET = True # An Fermaten-Schritten kein '_' in A/T/B: alle Stimmen artikulieren
                           # den Kadenzakkord (bei Bach zu 99.3-99.7% der Fall).

# ====== SELBST-DESTILLATION (self_distill.py) ======
DISTILL_MODEL_DIR = "./music_transformer_satb-key-sd"
DISTILL_FILE = "bach-chorales/distill-best-of-n.json"
DISTILL_MAX_RULE_SCORE = 4.0  # Qualitätsfilter: nur Pipeline-Outputs unter ~2x Bach-Median (1.8)
DISTILL_REPEAT = 2            # Gewicht der destillierten Paare gegenüber den Bach-Ankern
DISTILL_EPOCHS = 5            # kurzes Feintuning — gegen Drift in Modell-Eigenheiten
DISTILL_LR = 5e-5             # kleine Lernrate, Basis ist das fertige v2.2-Modell
DISTILL_WARMUP = 50
