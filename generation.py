"""Erzeugen: Constrained Decoding, Best-of-N-Reranking, Musiktheorie-Checks.

Kernstück ist die Klasse ChoraleHarmonizer: sie erzeugt zu einer
Sopranstimme (Format v2.1, siehe bach_chorales.py) die Unterstimmen.
Sopran-Slots, ';' und '|' werden hart forciert, A/T/B-Slots auf ihr
Register beschränkt; aus GEN_CANDIDATES Sampling-Kandidaten wählt ein
Reranking aus Modell-Likelihood und Satzregel-Malus den besten.
"""

import torch
from transformers import LogitsProcessor, LogitsProcessorList

import config as C


# ====== MUSIKTHEORIE-CHECKS (für das Best-of-N-Reranking) ======

NOTE_BASE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def token_to_midi(tok):
    """Notenname (music21-Stil, z.B. 'B-3', 'C#4', 'E-4^') -> MIDI-Nummer;
    'R'/'_' werden unverändert zurückgegeben, Unparsebares -> None."""
    tok = tok.rstrip('^')
    if tok in ('R', '_'):
        return tok
    try:
        letter = tok[0]
        acc = 0
        j = 1
        while j < len(tok) and tok[j] in '#-':
            acc += 1 if tok[j] == '#' else -1
            j += 1
        octave = int(tok[j:])
        return (octave + 1) * 12 + NOTE_BASE[letter] + acc
    except (KeyError, ValueError, IndexError):
        return None


LETTER_ORDER = 'CDEFGAB'


# Gewichte pro Regel als Log-Ratios zwischen Modell- und Bach-Rate:
#   w_r = clip(log((Rate_Modell + eps) / (Rate_Bach + eps)), 0, 8)
# Gemessen mit measure_rule_weights.py am v2.2-Modell (25 Train-Choräle x 2
# Rohproben ohne Reranking/Reparatur) gegen alle 343 Bach-Choräle in
# Originaltonart. Kommentar je Regel: Bach-Rate / Modell-Rate pro 100
# Sechzehntel. Begründung (Naive-Bayes "Bach vs. Modell"): Regeln, die das
# Modell von sich aus seltener verletzt als Bach, brauchen keinen Malus —
# die harten Constraints (Register, Fermaten) und der Repeat-Prozessor haben
# Kreuzung, Abstand, Tonwiederholung etc. bereits gelöst. Der Malus
# konzentriert sich auf die gemessene Kernschwäche: Parallelen (28x Bach).
# Ausnahme von der Formel: 'stagnation' bleibt als Wächter-Regel hart
# (Bach-Rate exakt 0; feuert nie, außer etwas geht wirklich schief).
# Nach jedem Retraining neu messen: python measure_rule_weights.py
# Stand: gemessen MIT aktivem ParallelPenaltyProcessor — der drückt die
# Parallelen-Rate des Modells von 5.08 auf 0.22 (Bach: 0.18); der Rest-Malus
# verteilt sich auf die verbleibenden kleinen Abweichungen.
RULE_WEIGHTS = {
    'stagnation':          100.0,  # Bach 0.00 / Modell 0.00 — Wächter
    'verdeckte':             0.9,  # Bach 0.19 / Modell 0.48
    'tritonus_unaufgeloest': 0.8,  # Bach 0.10 / Modell 0.25
    'uebermaessige_sekunde': 0.8,  # Bach 0.14 / Modell 0.33
    'septsprung':            0.7,  # Bach 0.10 / Modell 0.22
    'alle_gleiche_richtung': 0.6,  # Bach 1.23 / Modell 2.33
    'leerer_klang':          0.4,  # Bach 0.32 / Modell 0.49
    'sprung_ueber_oktave':   0.3,  # Bach 0.05 / Modell 0.07
    'sprung_ohne_ausgleich': 0.2,  # Bach 0.97 / Modell 1.24
    'ueberschneidung_klein': 0.2,  # Bach 1.06 / Modell 1.34
    'parallelen':            0.2,  # Bach 0.18 / Modell 0.22 — Sampling-Check erledigt das
    'terzverdopplung_dur':   0.1,  # Bach 1.70 / Modell 1.80
    'akzent_parallelen':     0.0,  # Bach 0.48 / Modell 0.51
    'kreuzung':              0.0,  # Bach 0.80 / Modell 0.25 — Constraints erledigen das
    'ueberschneidung_gross': 0.0,  # Bach 0.91 / Modell 0.47
    'abstand':               0.0,  # Bach 0.54 / Modell 0.18
    'unisono_tb':            0.0,  # Bach 1.16 / Modell 1.08
    'unisono_hoch':          0.0,  # Bach 1.20 / Modell 0.78
    'tritonus_aufgeloest':   0.0,  # Bach 0.27 / Modell 0.13
    'ton_wiederholung':      0.0,  # Bach 2.94 / Modell 1.72 — Repeat-Prozessor erledigt das
}


# Bach-Idiome aus Regel-Kombinationen: Paare, die in Bachs Sätzen in derselben
# 16tel-Gruppe ÜBERZUFÄLLIG oft gemeinsam auftreten (Lift = beobachtet /
# unabhängig-erwartet, gemessen über alle 343 Choräle; aufgenommen ab
# Lift >= 5 und n >= 20). Interpretation: die zweite "Verletzung" ist Teil
# DESSELBEN Vorfalls (verdeckte Quinte setzt Gleichbewegung voraus, der leere
# Klang entsteht durch das Unisono, ...) — sie bekommt im Score keinen
# eigenen Malus (statt der generischen 0.5-Dämpfung). Zahl = gemessener Lift.
PAIR_IDIOMS = {
    frozenset(p) for p, _lift in [
        (('kreuzung', 'ueberschneidung_gross'), 75.6),
        (('alle_gleiche_richtung', 'verdeckte'), 39.7),
        (('kreuzung', 'ueberschneidung_klein'), 31.3),
        (('alle_gleiche_richtung', 'septsprung'), 28.5),
        (('septsprung', 'ueberschneidung_klein'), 26.0),
        (('abstand', 'kreuzung'), 17.2),
        (('ueberschneidung_gross', 'verdeckte'), 17.2),
        (('abstand', 'ueberschneidung_gross'), 15.3),
        (('leerer_klang', 'unisono_hoch'), 14.3),
        (('ueberschneidung_klein', 'unisono_tb'), 13.3),
        (('leerer_klang', 'unisono_tb'), 10.3),
        (('alle_gleiche_richtung', 'ueberschneidung_gross'), 9.3),
        (('terzverdopplung_dur', 'unisono_hoch'), 7.7),
        (('akzent_parallelen', 'alle_gleiche_richtung'), 7.4),
        (('abstand', 'ueberschneidung_klein'), 6.6),
        (('ueberschneidung_klein', 'unisono_hoch'), 6.6),
        (('ton_wiederholung', 'unisono_tb'), 5.9),
        (('akzent_parallelen', 'unisono_tb'), 5.3),
        (('sprung_ohne_ausgleich', 'ton_wiederholung'), 5.3),
        (('ueberschneidung_gross', 'unisono_tb'), 5.2),
    ]
}


# Alle geprüften Regeln (Beschreibung siehe rule_violation_events)
RULE_NAMES = (
    # Onset-zu-Onset
    'parallelen',             # Quint-/Oktavparallelen (inkl. Antiparallelen)
    'verdeckte',              # verdeckte Quinten/Oktaven der Außenstimmen
    'kreuzung',               # Stimmkreuzung bei Onset
    'ueberschneidung_klein',  # Overlap <= Ganzton über/unter dem Nachbarton
    'ueberschneidung_gross',  # Overlap > Ganzton
    'abstand',                # S-A bzw. A-T weiter als eine Oktave
    'unisono_tb',             # Tenor und Bass im Einklang (bei Bach üblich)
    'unisono_hoch',           # S-A oder A-T im Einklang
    'alle_gleiche_richtung',  # alle vier Stimmen gleichgerichtet
    # Melodik der Unterstimmen
    'uebermaessige_sekunde',  # z.B. F-G# in Moll
    'tritonus_aufgeloest',    # Tritonus-Sprung mit Schritt-Auflösung danach
    'tritonus_unaufgeloest',  # Tritonus-Sprung ohne Auflösung
    'septsprung',             # Sprung um eine Septime (10/11 HT)
    'sprung_ueber_oktave',    # Sprung > 12 HT
    'sprung_ohne_ausgleich',  # Sprung >= kl. Sexte ohne Schritt zurück
    # Mehrschrittig / mehrstimmig
    'akzent_parallelen',      # Quinten/Oktaven auf aufeinanderfolgenden
                              # Schlägen trotz Zwischenbewegung
    'leerer_klang',           # Schlag-Akkord mit nur 1-2 Tonklassen
                              # (Ausnahme: Fermate, Schlussakkord)
    'terzverdopplung_dur',    # verdoppelte Terz im vollständigen Durakkord
    'ton_wiederholung',       # >= 3 gleiche Onsets in Folge in Alt/Tenor
    'stagnation',             # A/T/B > 1 Takt ohne jeden Onset
                              # (Ausnahme: Fermate im Fenster)
)


def rule_violation_events(tokens):
    """Prüft eine Zielsequenz im Format v2.1 ("S A T B ;" pro 16tel, '|'
    Taktstriche) gegen die Satzregeln des Bach-Chorsatzes und liefert jede
    Verletzung als (Gruppenindex, Regelname) — der Index erlaubt gezielte
    Reparatur an der betroffenen Stelle.

    Neben Onset-zu-Onset-Regeln werden auch mehrschrittige Regeln geprüft
    (Akzentparallelen zwischen aufeinanderfolgenden Schlägen, Sprung- und
    Tritonus-Auflösung, Tonwiederholungsketten, harmonische Stagnation) und
    Akkordregeln (leerer Klang, Terzverdopplung im Durakkord). Einige Regeln
    sind in Varianten aufgespalten, weil Bach die eine Ausprägung meidet und
    die andere zulässt (z.B. aufgelöster vs. unaufgelöster Tritonus-Sprung) —
    die Gewichte dazu stehen in RULE_WEIGHTS.

    Haltungen ('_') lösen keine Prüfung aus; Pausen unterbrechen die Verfolgung.
    Der Sopran ist vorgegeben und wird melodisch nicht geprüft."""
    events = []

    # Gruppen mit Metrik parsen: (Tokens, ist_schlag, hat_fermate)
    groups = []
    g = []
    idx_in_measure = 0
    for tok in tokens:
        if tok == '|':
            idx_in_measure = 0
            continue
        if tok == ';':
            if len(g) == 4:
                groups.append((g, idx_in_measure % 4 == 0, '^' in g[0]))
                idx_in_measure += 1
            g = []
        else:
            g.append(tok)

    # letzter Onset im Stück (für die Schlussakkord-Ausnahme)
    last_onset_gi = -1
    for gi, (g, _, _) in enumerate(groups):
        if any(t != '_' for t in g):
            last_onset_gi = gi

    cur = [None, None, None, None]         # aktuell klingende MIDI-Tonhöhe je Stimme
    cur_letter = [None, None, None, None]  # Stammton (0..6) für Umdeutung #/b
    pending_leap = [0, 0, 0, 0]            # Richtung eines auszugleichenden Sprungs
    pending_tritone = [0, 0, 0, 0]         # Richtung eines aufzulösenden Tritonus
    prev_beat_chord = None                 # klingender Akkord am vorigen Schlag
    intermediate_onset = [False] * 4       # Onset seit dem letzten Schlag?
    repeat_pitch = [None, None, None, None]  # Tonwiederholungs-Ketten (Alt/Tenor)
    repeat_run = [0, 0, 0, 0]
    static_run = 0                         # 16tel ohne Onset in A/T/B
    fermata_in_static = False

    for gi, (g, is_beat, has_fermata) in enumerate(groups):
        new = cur.copy()
        new_letter = cur_letter.copy()
        moved = [False] * 4
        onset = [False] * 4
        for v, tok in enumerate(g):
            m = token_to_midi(tok)
            if m == '_' or m is None:
                continue
            onset[v] = True
            if m == 'R':
                new[v] = None
                new_letter[v] = None
                pending_leap[v] = 0
                pending_tritone[v] = 0
                repeat_run[v] = 0
                repeat_pitch[v] = None
                continue
            moved[v] = (cur[v] != m)
            new[v] = m
            new_letter[v] = LETTER_ORDER.index(tok.rstrip('^')[0])

        # Quint-/Oktavparallelen (inkl. Antiparallelen)
        for i in range(4):
            for j in range(i + 1, 4):
                if None in (cur[i], cur[j], new[i], new[j]):
                    continue
                if not (moved[i] and moved[j]):
                    continue
                if ((cur[i] - cur[j]) % 12 == (new[i] - new[j]) % 12
                        and (new[i] - new[j]) % 12 in (0, 7)):
                    events.append((gi, 'parallelen'))

        # Verdeckte Quinten/Oktaven in den Außenstimmen: gleiche Richtung in
        # eine reine Quinte/Oktave hinein, Sopran springt (>1 Ganzton).
        # Echte Parallelen (gleiche Intervallklasse vorher) zählt schon oben.
        if None not in (cur[0], cur[3], new[0], new[3]) and moved[0] and moved[3]:
            ds, db = new[0] - cur[0], new[3] - cur[3]
            if ds * db > 0 and (new[0] - new[3]) % 12 in (0, 7) \
                    and (cur[0] - cur[3]) % 12 != (new[0] - new[3]) % 12 \
                    and abs(ds) > 2:
                events.append((gi, 'verdeckte'))

        # Stimmkreuzung bei Onset
        for v in range(3):
            if new[v] is not None and new[v + 1] is not None \
                    and new[v] < new[v + 1] and (moved[v] or moved[v + 1]):
                events.append((gi, 'kreuzung'))

        # Stimmüberschneidung (Overlap), nach Ausmaß getrennt:
        # kleine Überschneidungen (<= Ganzton) erlaubt sich Bach häufig
        for v in range(3):
            over = 0
            if moved[v + 1] and cur[v] is not None and new[v + 1] is not None \
                    and new[v + 1] > cur[v]:
                over = max(over, new[v + 1] - cur[v])
            if moved[v] and cur[v + 1] is not None and new[v] is not None \
                    and new[v] < cur[v + 1]:
                over = max(over, cur[v + 1] - new[v])
            if over > 2:
                events.append((gi, 'ueberschneidung_gross'))
            elif over > 0:
                events.append((gi, 'ueberschneidung_klein'))

        # Stimmabstand: S–A und A–T höchstens eine Oktave
        for hi, lo in ((0, 1), (1, 2)):
            if new[hi] is not None and new[lo] is not None \
                    and (moved[hi] or moved[lo]) and new[hi] - new[lo] > 12:
                events.append((gi, 'abstand'))

        # Benachbarte Stimmen im Einklang (T-B separat: bei Bach üblich)
        for v in range(3):
            if new[v] is not None and new[v + 1] is not None \
                    and (moved[v] or moved[v + 1]) and new[v] == new[v + 1]:
                events.append((gi, 'unisono_tb' if v == 2 else 'unisono_hoch'))

        # Alle vier Stimmen in derselben Richtung
        if all(moved) and None not in cur and None not in new:
            dirs = [new[v] - cur[v] for v in range(4)]
            if all(d > 0 for d in dirs) or all(d < 0 for d in dirs):
                events.append((gi, 'alle_gleiche_richtung'))

        # Melodik der Unterstimmen (der Sopran ist vorgegeben)
        for v in (1, 2, 3):
            if moved[v] and cur[v] is not None and new[v] is not None:
                diff = abs(new[v] - cur[v])
                direction = 1 if new[v] > cur[v] else -1

                if diff in (10, 11):
                    events.append((gi, 'septsprung'))
                if diff > 12:
                    events.append((gi, 'sprung_ueber_oktave'))
                if diff == 3 and cur_letter[v] is not None and new_letter[v] is not None \
                        and (new_letter[v] - cur_letter[v]) % 7 in (1, 6):
                    events.append((gi, 'uebermaessige_sekunde'))

                # Tritonus-Sprung: Bewertung hängt von der Fortsetzung ab —
                # Schritt in Gegenrichtung = aufgelöst (bei Bach akzeptiert)
                if pending_tritone[v] != 0:
                    if direction * pending_tritone[v] < 0 and diff <= 2:
                        events.append((gi, 'tritonus_aufgeloest'))
                    else:
                        events.append((gi, 'tritonus_unaufgeloest'))
                    pending_tritone[v] = 0
                if diff == 6:
                    pending_tritone[v] = direction

                # Sprungausgleich: nach einem Sprung >= kleine Sexte soll die
                # Stimme schrittweise (<= Ganzton) in Gegenrichtung weitergehen
                if pending_leap[v] != 0:
                    if not (direction * pending_leap[v] < 0 and diff <= 2):
                        events.append((gi, 'sprung_ohne_ausgleich'))
                    pending_leap[v] = 0
                if diff >= 8:
                    pending_leap[v] = direction

        # Tonwiederholungsketten in den Mittelstimmen (>= 3 gleiche Onsets)
        for v in (1, 2):
            if onset[v] and new[v] is not None:
                if new[v] == repeat_pitch[v]:
                    repeat_run[v] += 1
                    if repeat_run[v] >= 3:
                        events.append((gi, 'ton_wiederholung'))
                else:
                    repeat_pitch[v] = new[v]
                    repeat_run[v] = 1

        # Harmonische Stagnation: kein einziger Onset in A/T/B über mehr als
        # einen 4/4-Takt (16 Sechzehntel) — außer eine Fermate liegt im Fenster
        if any(onset[1:]):
            if static_run > 16 and not fermata_in_static:
                events.append((gi, 'stagnation'))
            static_run = 0
            fermata_in_static = False
        else:
            static_run += 1
        if has_fermata:
            fermata_in_static = True

        # ==== Schlag-Ebene (Akkordregeln, Akzentparallelen) ====
        if is_beat:
            # Akzentparallelen: Quinte/Oktave zwischen denselben Stimmen auf
            # zwei aufeinanderfolgenden Schlägen, obwohl sich (mindestens)
            # eine der Stimmen dazwischen wegbewegt hatte. Direkte Parallelen
            # am Schlag selbst zählt schon die Onset-Regel.
            if prev_beat_chord is not None:
                for i in range(4):
                    for j in range(i + 1, 4):
                        if None in (prev_beat_chord[i], prev_beat_chord[j],
                                    new[i], new[j]):
                            continue
                        if new[i] == prev_beat_chord[i] or new[j] == prev_beat_chord[j]:
                            continue
                        if moved[i] and moved[j] and cur[i] is not None and cur[j] is not None \
                                and (cur[i] - cur[j]) % 12 == (new[i] - new[j]) % 12:
                            continue  # als direkte Parallele bereits gezählt
                        if not (intermediate_onset[i] or intermediate_onset[j]):
                            continue
                        if ((prev_beat_chord[i] - prev_beat_chord[j]) % 12
                                == (new[i] - new[j]) % 12
                                and (new[i] - new[j]) % 12 in (0, 7)):
                            events.append((gi, 'akzent_parallelen'))

            if any(onset) and None not in new:
                pcs = [p % 12 for p in new]
                distinct = set(pcs)
                # Leerer Klang (nur Grundton/Quinte bzw. Oktaven) auf dem
                # Schlag — Ausnahmen: Fermate und Schlussakkord
                if len(distinct) <= 2 and not has_fermata and gi != last_onset_gi:
                    events.append((gi, 'leerer_klang'))
                # Terzverdopplung im vollständigen Durakkord
                if len(distinct) == 3:
                    for root in distinct:
                        if {(root) % 12, (root + 4) % 12, (root + 7) % 12} == distinct:
                            if pcs.count((root + 4) % 12) >= 2:
                                events.append((gi, 'terzverdopplung_dur'))
                            break

            prev_beat_chord = new.copy()
            intermediate_onset = [False] * 4
        else:
            for v in range(4):
                if onset[v]:
                    intermediate_onset[v] = True

        cur = new
        cur_letter = new_letter

    # offene Tritonus-Sprünge am Stückende gelten als unaufgelöst
    for p in pending_tritone:
        if p != 0:
            events.append((len(groups) - 1, 'tritonus_unaufgeloest'))
    return events


def count_rule_violations_by_rule(tokens):
    """Aggregiert rule_violation_events zu dict Regelname -> Anzahl."""
    counts = {name: 0 for name in RULE_NAMES}
    for _, name in rule_violation_events(tokens):
        counts[name] += 1
    return counts


def count_rule_violations(tokens):
    """Ungewichtete Gesamtzahl aller Satzregel-Verstöße."""
    return len(rule_violation_events(tokens))


def rule_violation_score(tokens):
    """Gewichtete Verstoß-Summe für das Reranking mit Kaskaden-Dämpfung und
    Idiom-Erlass.

    Gewichte: RULE_WEIGHTS (Log-Ratios Modell vs. Bach). Pro 16tel-Gruppe:
      - der teuerste Verstoß zählt voll;
      - jeder weitere zählt 0 (!), wenn er mit einem bereits gezählten
        Verstoß der Gruppe ein gemessenes Bach-Idiom bildet (PAIR_IDIOMS —
        z.B. verdeckte Quinte + Gleichbewegung: EIN Vorfall, keine zwei);
      - sonst zählt er zur Hälfte (generische Kaskaden-Dämpfung: 44% der
        Bach-Ereignisse liegen in Gruppen mit mehreren gleichzeitigen
        Ereignissen, mechanische Kopplungen würden sonst doppelt bestraft)."""
    by_group = {}
    for gi, name in rule_violation_events(tokens):
        by_group.setdefault(gi, []).append(name)
    total = 0.0
    for names in by_group.values():
        names = sorted(names, key=lambda n: -RULE_WEIGHTS.get(n, 1.0))
        counted = []
        for idx, name in enumerate(names):
            w = RULE_WEIGHTS.get(name, 1.0)
            if idx == 0:
                total += w
            elif any(frozenset((name, c)) in PAIR_IDIOMS for c in counted):
                pass  # Teil eines bereits gezählten Idioms
            else:
                total += 0.5 * w
            counted.append(name)
    return total


def analyze_source(sequence: str):
    """Strukturanalyse einer Sopran-Quellsequenz (Format v2.1).

    Returns dict:
      step_tokens:        Sopran-Token je 16tel-Schritt (inkl. '^' und '_')
      group_starts:       Zielposition (Decoder) des Gruppenanfangs je Schritt
      bar_positions:      Zielpositionen der Taktstriche
      fermata_steps:      Schrittindizes mit Fermate
      phrase_start_steps: Schrittindizes, an denen eine neue Phrase beginnt
                          (Stückanfang und jeder erste Onset nach einer Fermate)
      total_len:          Gesamtlänge der Zielsequenz in Tokens
    """
    step_size = 5 if C.NUM_VOICES > 2 else 1
    step_tokens = []
    group_starts = []
    bar_positions = []
    fermata_steps = set()
    phrase_start_steps = [0]
    pos = 0
    after_fermata = False
    for tok in sequence.split():
        if '/' in tok or ':' in tok:
            continue  # Taktart-/Tonart-Token: nur Encoder-Info, kein 16tel-Schritt
        if tok == '|':
            bar_positions.append(pos)
            pos += 1
            continue
        if tok != '_' and after_fermata:
            phrase_start_steps.append(len(step_tokens))
            after_fermata = False
        step_tokens.append(tok)
        group_starts.append(pos)
        if tok.endswith('^'):
            fermata_steps.add(len(step_tokens) - 1)
            after_fermata = True
        pos += step_size
    return {
        'step_tokens': step_tokens,
        'group_starts': group_starts,
        'bar_positions': bar_positions,
        'fermata_steps': fermata_steps,
        'phrase_start_steps': phrase_start_steps,
        'total_len': pos,
    }


def strip_soprano_slots(seq: str) -> str:
    """Entfernt die Sopran-Slots aus einer Zielsequenz im Format v2
    ("S A T B ;" pro 16tel) => "A T B ;" für die Ausgabe."""
    out = []
    expect_soprano = True
    for tok in seq.split():
        if tok == '|':
            out.append(tok)
            continue
        if expect_soprano:
            expect_soprano = False
            continue
        out.append(tok)
        if tok == ';':
            expect_soprano = True
    return ' '.join(out)


# ====== LOGITS-PROZESSOREN ======

class TokenBiasLogitsProcessor(LogitsProcessor):
    """Konstanter Logit-Bias auf ein einzelnes Token (z.B. '_')."""

    def __init__(self, token_id: int, bias: float):
        self.token_id = token_id
        self.bias = bias

    def __call__(self, input_ids, scores):  # noqa: ARG002
        scores = scores.clone()
        scores[..., self.token_id] = scores[..., self.token_id] + self.bias
        return scores


class RepeatNotePenaltyProcessor(LogitsProcessor):
    """Malus für sofortige Tonwiederholung: denselben (noch klingenden) Ton
    erneut anschlagen statt zu halten. Wirkt nur auf die angegebenen
    Stimmen (default Alt+Tenor) und nur auf genau die eine Tonhöhe —
    Haltungen ('_') und Durchgangsnoten zu anderen Tönen sind unberührt."""

    def __init__(self, slot_map, penalty, underscore_id, rest_id, voices=(0, 1)):
        self.penalty = penalty
        self.underscore_id = underscore_id
        self.rest_id = rest_id
        # Positionen je Stimme in Reihenfolge + Lookup Position -> (Stimme, Index)
        self.voice_positions = {v: [] for v in voices}
        self.pos_index = {}
        for pos in sorted(slot_map):
            v = slot_map[pos]
            if v in self.voice_positions:
                self.pos_index[pos] = (v, len(self.voice_positions[v]))
                self.voice_positions[v].append(pos)

    def __call__(self, input_ids, scores):
        pos = input_ids.shape[1] - 1  # Zielposition, die gerade erzeugt wird
        if pos not in self.pos_index:
            return scores
        v, idx = self.pos_index[pos]
        positions = self.voice_positions[v]
        scores = scores.clone()
        for b in range(input_ids.shape[0]):
            # aktuell klingenden Ton der Stimme suchen (letzter Onset)
            for j in range(idx - 1, -1, -1):
                tok = int(input_ids[b, positions[j] + 1])  # +1: Decoder-Starttoken
                if tok != self.underscore_id:
                    if tok != self.rest_id:
                        scores[b, tok] = scores[b, tok] - self.penalty
                    break
        return scores


class ParallelPenaltyProcessor(LogitsProcessor):
    """Parallelen-Check direkt im Sampling: beim Erzeugen eines A/T/B-Slots
    werden alle Tonhöhen bestraft, die mit einer in dieser Gruppe bereits
    festgelegten Stimme (Sopran ist immer festgelegt, frühere Slots ebenso)
    eine parallele Quinte/Oktave bilden würden — beide Stimmen bewegt,
    Intervallklasse bleibt rein. Haltungen ('_') und Pausen sind nie
    betroffen; die Strafe ist endlich (kein Verbot), damit seltene, bewusst
    gesetzte Parallelen wie bei Bach möglich bleiben."""

    def __init__(self, group_starts, music_tokenizer, penalty):
        self.penalty = penalty
        self.group_starts = group_starts
        # Zielposition -> (Schritt, SATB-Stimme 1..3); Sopran (0) wird forciert
        self.pos_info = {}
        for s, gs in enumerate(group_starts):
            for v in (1, 2, 3):
                self.pos_info[gs + v] = (s, v)
        # Token-ID -> MIDI; Sentinels: -1 Haltung ('_'), -2 kein Ton (R, Struktur)
        self.midi_of = []
        for i in range(len(music_tokenizer)):
            m = token_to_midi(music_tokenizer.convert_ids_to_tokens(i))
            self.midi_of.append(-1 if m == '_' else m if isinstance(m, int) else -2)
        self.midi_vec = torch.tensor([m if m >= 0 else -1000 for m in self.midi_of])

    def _sounding(self, row, step, voice):
        """Aktuell klingende MIDI-Tonhöhe der Stimme vor Gruppe `step`."""
        for t in range(step - 1, -1, -1):
            m = self.midi_of[int(row[self.group_starts[t] + voice + 1])]  # +1: Decoder-Start
            if m == -1:
                continue  # Haltung: weiter zurück
            return m if m >= 0 else None  # Pause unterbricht
        return None

    def __call__(self, input_ids, scores):
        pos = input_ids.shape[1] - 1  # Zielposition, die gerade erzeugt wird
        info = self.pos_info.get(pos)
        if info is None:
            return scores
        step, v = info
        scores = scores.clone()
        mv = self.midi_vec.to(scores.device)
        for b in range(input_ids.shape[0]):
            row = input_ids[b]
            p_v = self._sounding(row, step, v)
            if p_v is None:
                continue
            for u in range(4):
                u_pos = self.group_starts[step] + u
                if u == v or u_pos >= pos:
                    continue  # Slot noch nicht erzeugt
                n_u = self.midi_of[int(row[u_pos + 1])]
                if n_u < 0:
                    continue  # Haltung (nicht bewegt) oder Pause
                p_u = self._sounding(row, step, u)
                if p_u is None or n_u == p_u:
                    continue  # Stimme u hat sich nicht bewegt
                # Intervallklasse konsistent in der Orientierung (oben - unten)
                if v < u:
                    d0 = (p_v - p_u) % 12
                    if d0 not in (0, 7):
                        continue
                    forbidden = ((mv - n_u) % 12 == d0) & (mv >= 0) & (mv != p_v)
                else:
                    d0 = (p_u - p_v) % 12
                    if d0 not in (0, 7):
                        continue
                    forbidden = ((n_u - mv) % 12 == d0) & (mv >= 0) & (mv != p_v)
                scores[b, forbidden] -= self.penalty
        return scores


# ====== HARMONISIERER ======

class ChoraleHarmonizer:
    """Harmonisiert eine Sopranstimme mit Alt, Tenor und Bass."""

    def __init__(self, model, music_tokenizer, voice_allowed_ids):
        self.model = model
        self.tokenizer = music_tokenizer
        self.voice_allowed_ids = voice_allowed_ids

        self.underscore_id = music_tokenizer.convert_tokens_to_ids('_')
        self.bar_id = music_tokenizer.convert_tokens_to_ids('|')
        self.semicolon_id = music_tokenizer.convert_tokens_to_ids(';')
        self.rest_id = music_tokenizer.convert_tokens_to_ids('R')
        # Fallback: alle Tokens außer Strukturzeichen (nur für den 1-Stimmen-Fall)
        self.note_only_ids = [i for i in range(len(music_tokenizer))
                              if i not in {self.bar_id, self.semicolon_id}]

    def build_decoder_constraints(self, sequence: str):
        """
        Constrained Decoding (Idee aus constraint-transformer-bach):
        Ziel-Struktur pro 16tel ist [S A T B ;], an Taktgrenzen [|].
        - Sopran-Slots werden auf das bekannte Sopran-Token forciert
          (inkl. Fermaten-Token! => Kadenz-Rhythmus ist lokal sichtbar)
        - ';' und '|' werden forciert
        - A/T/B-Slots werden auf das Register der jeweiligen Stimme beschränkt
        Returns dict: forced (pos->token_id), slots (pos->Stimmindex 0..2),
        total_len, group_starts, fermata_slots (A/T/B-Positionen an Fermaten),
        phrase_boundaries (Zielpositionen der Phrasenanfänge).
        """
        info = analyze_source(sequence)
        forced = {}
        slots = {}
        for p in info['bar_positions']:
            forced[p] = self.bar_id
        if C.NUM_VOICES > 2:
            for step, tok in enumerate(info['step_tokens']):
                pos = info['group_starts'][step]
                forced[pos] = self.tokenizer.convert_tokens_to_ids(tok)
                slots[pos + 1], slots[pos + 2], slots[pos + 3] = 0, 1, 2
                forced[pos + 4] = self.semicolon_id
        fermata_slots = {info['group_starts'][s] + k
                         for s in info['fermata_steps'] for k in (1, 2, 3)}
        phrase_boundaries = [info['group_starts'][s] for s in info['phrase_start_steps']]
        return {
            'forced': forced,
            'slots': slots,
            'total_len': info['total_len'],
            'group_starts': info['group_starts'],
            'fermata_slots': fermata_slots,
            'phrase_boundaries': phrase_boundaries,
        }

    @torch.no_grad()
    def transform(self, sequence: str) -> str:
        """Generiert eine Harmonisierung in drei Stufen:
        1. Phrasenweises Best-of-N: an den Fermaten segmentiert; pro Phrase
           werden GEN_CANDIDATES Fortsetzungen gesampelt, die beste (Modell-
           Likelihood auf den A/T/B-Slots minus Bach-kalibrierter Regel-Malus)
           wird fixiert — effektiv N^Phrasen Kombinationen zum Preis von
           N×Phrasen.
        2. Reparatur: ab der Stelle des schwersten verbliebenen Regelverstoßes
           wird neu gesampelt; übernommen wird nur, was den Gesamt-Score
           verbessert (GEN_REPAIR_ITERS Versuche).
        3. Constrained Sampling überall: Sopran/Struktur forciert, Register
           je Stimme beschränkt, an Fermaten artikulieren alle Stimmen."""
        model, tokenizer = self.model, self.tokenizer

        encoded_input = tokenizer.encode(sequence)
        input_ids = torch.tensor([encoded_input]).to(model.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        cons = self.build_decoder_constraints(sequence)
        forced_pos = cons['forced']
        slot_map = cons['slots']
        total_len = cons['total_len']
        fermata_slots = cons['fermata_slots'] if C.FERMATA_FORCE_ONSET else set()

        allowed_no_hold = None
        if self.voice_allowed_ids is not None:
            # An Fermaten kein '_': alle Stimmen schlagen den Kadenzakkord an
            # (bei Bach zu >99% der Fall)
            allowed_no_hold = [[i for i in ids if i != self.underscore_id]
                               for ids in self.voice_allowed_ids]

        def prefix_allowed_tokens_fn(batch_id, input_ids):  # noqa: ARG001
            pos = len(input_ids) - 1  # 0-indexed position being generated
            if pos in forced_pos:
                return [forced_pos[pos]]
            if slot_map and pos in slot_map:
                v = slot_map[pos]
                if pos in fermata_slots:
                    return allowed_no_hold[v]
                return self.voice_allowed_ids[v]
            return self.note_only_ids

        processors = []
        if C.UNDERSCORE_BIAS != 0.0:
            processors.append(TokenBiasLogitsProcessor(self.underscore_id, bias=C.UNDERSCORE_BIAS))
        if C.REPEAT_NOTE_PENALTY > 0.0 and slot_map:
            processors.append(RepeatNotePenaltyProcessor(
                slot_map, C.REPEAT_NOTE_PENALTY, self.underscore_id, self.rest_id))
        if C.PARALLEL_PENALTY > 0.0 and slot_map:
            processors.append(ParallelPenaltyProcessor(
                cons['group_starts'], tokenizer, C.PARALLEL_PENALTY))
        logits_processor = LogitsProcessorList(processors) if processors else None

        num_steps = max(1, len(cons['group_starts']))
        atb_positions_all = sorted(slot_map)
        start_id = tokenizer.cls_token_id

        # Top-p-Sampling ohne globale Penalties: repetition_penalty würde
        # musikalisch korrekte Wiederholungen ('_', gleiche Tonhöhen) bestrafen.
        # Da Sopran, Struktur und Register hart forciert sind, bleibt das
        # Sampling im gültigen Raum.
        def gen(prefix, n_new, n_seq):
            """Sampelt n_seq Fortsetzungen von genau n_new Tokens ab prefix."""
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=prefix.unsqueeze(0),
                max_new_tokens=n_new,
                min_new_tokens=n_new,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                logits_processor=logits_processor,
                do_sample=True,
                num_beams=1,
                top_p=C.GEN_TOP_P,
                top_k=0,
                temperature=C.GEN_TEMPERATURE,
                num_return_sequences=n_seq,
            )

        # Stil-Terme (Harmonik-Bigramm, Textur): Abweichung vom Bach-typischen
        # Überraschungsniveau, konfiguriert/abschaltbar in style_config.json
        try:
            from style_model import get_style_scorer
            style = get_style_scorer()
            if not style.active:
                style = None
        except (FileNotFoundError, KeyError):
            style = None  # kein Datensatz / keine Statistik: Stil-Terme aus

        def score(cands):
            """Globalziel je Kandidat über den bereits erzeugten Teil:
            mittlere A/T/B-Log-Likelihood − RULE_PENALTY_WEIGHT × gewichtete
            Regelverstöße pro 16tel − Stil-Abweichung (style_model).
            Gemeinsame Präfixe kürzen sich beim Vergleich heraus."""
            n = cands.size(0)
            out = model(
                input_ids=input_ids.expand(n, -1),
                attention_mask=attention_mask.expand(n, -1),
                decoder_input_ids=cands[:, :-1],
            )
            logprobs = torch.log_softmax(out.logits.float(), dim=-1)
            token_lp = logprobs.gather(-1, cands[:, 1:].unsqueeze(-1)).squeeze(-1)
            length = cands.size(1) - 1
            atb = [p for p in atb_positions_all if p < length]
            lik = token_lp[:, atb].mean(dim=1) if atb else token_lp.mean(dim=1)
            token_lists = [tokenizer.decode(c, skip_special_tokens=True).split()
                           for c in cands]
            pens = torch.tensor(
                [rule_violation_score(t) for t in token_lists],
                dtype=lik.dtype, device=lik.device) / num_steps
            total = lik
            if C.RULE_PENALTY_WEIGHT > 0.0:
                total = total - C.RULE_PENALTY_WEIGHT * pens
            if style is not None:
                stil = torch.tensor(
                    [style.penalty(t, sequence) for t in token_lists],
                    dtype=lik.dtype, device=lik.device)
                total = total - stil
            return total, pens

        if C.GEN_CANDIDATES <= 1 or not slot_map or total_len == 0:
            best = gen(torch.tensor([start_id], device=model.device), total_len, 1)[0]
            return tokenizer.decode(best, skip_special_tokens=True).strip()

        # ==== 1. Phrasenweises Best-of-N ====
        if C.GEN_PHRASEWISE:
            seg_starts = sorted({0} | {b for b in cons['phrase_boundaries'] if 0 < b < total_len})
        else:
            seg_starts = [0]
        seg_ends = seg_starts[1:] + [total_len]

        prefix = torch.tensor([start_id], device=model.device)
        for s, e in zip(seg_starts, seg_ends):
            cands = gen(prefix, e - s, C.GEN_CANDIDATES)
            scores, pens = score(cands)
            k = int(scores.argmax())
            prefix = cands[k]
        best, best_score, best_pen = prefix, scores[k], pens[k]

        # ==== 2. Reparatur an der Stelle des schwersten Verstoßes ====
        repairs = 0
        for _ in range(C.GEN_REPAIR_ITERS):
            events = rule_violation_events(
                tokenizer.decode(best, skip_special_tokens=True).split())
            # nur substanzielle Verstöße reparieren (Gewicht >= 1.0)
            events = [(gi, r) for gi, r in events if RULE_WEIGHTS.get(r, 1.0) >= 1.0]
            if not events:
                break
            gi = max(events, key=lambda ev: (RULE_WEIGHTS.get(ev[1], 1.0), -ev[0]))[0]
            step = max(0, gi - 1)  # eine Gruppe Kontext vor dem Verstoß neu würfeln
            start_pos = cons['group_starts'][step]
            if start_pos >= total_len:
                break
            cands = gen(best[: start_pos + 1], total_len - start_pos, C.GEN_CANDIDATES)
            scores, pens = score(cands)
            k = int(scores.argmax())
            if scores[k] > best_score:
                best, best_score, best_pen = cands[k], scores[k], pens[k]
                repairs += 1
            else:
                break  # keine Verbesserung gefunden

        print(f"  Regel-Score {float(best_pen) * num_steps:.1f} "
              f"({len(seg_starts)} Phrasen, {repairs} Reparaturen)")

        return tokenizer.decode(best, skip_special_tokens=True).strip()


def _export(src, seq, path):
    from bach_chorales import output_chorale
    if C.NUM_VOICES > 2:
        # Format v2: Sopran-Slot aus dem Target entfernen (er steckt
        # schon in src und würde sonst als eigene Stimme exportiert)
        seq = strip_soprano_slots(seq)
    elif seq.find(';') == -1:
        seq = seq.replace(' ', ' ; ')
    output_chorale(src, seq, path)


def write_test_outputs(harmonizer, pairs, output_dir=C.OUTPUT_DIR, presets=None):
    """Harmonisiert Choräle (Originaltonart-Paare) und exportiert Bach-Original
    und ChoraleHarmonizer-Fassung(en) als MIDI + MusicXML.

    presets: Liste von Stil-Preset-Namen aus style_config.json (z.B.
    ['konservativ', 'kuehn']) — je Choral entsteht eine Fassung pro Preset,
    mit Preset-Suffix im Dateinamen. None/[] = eine Fassung ohne Preset."""
    from pathlib import Path

    print(f"\n{'#' * 80}")
    print("TESTMATERIAL")
    print(f"{'#' * 80}\n")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    preset_list = list(presets) if presets else [None]

    for i, (src, tgt) in enumerate(pairs, 1):
        print(f"─── BEISPIEL {i} ───")
        print(f"Input:          {src}")
        try:
            _export(src, tgt, output_dir + f"/{i}-Bach")
        except Exception as e:
            print(e)

        for preset in preset_list:
            suffix = ''
            if preset is not None:
                from style_model import set_preset
                set_preset(preset)
                suffix = '-' + preset
                print(f"  [Preset: {preset}]")
            new_output = harmonizer.transform(src)
            try:
                _export(src, new_output, output_dir + f"/{i}-ChoraleHarmonizer{suffix}")
            except Exception as e:
                print(e)
        print()
