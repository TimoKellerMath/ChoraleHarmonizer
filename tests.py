"""Tests für ChoraleHarmonizer.

Aufruf:  python tests.py        (oder: pytest tests.py, falls pytest installiert)

Getestet werden die reinen Logik-Bausteine (Satzregeln, Tokenverarbeitung,
Export) — kein Training, kein Modell nötig.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bach_chorales import output_chorale, trim_trailing_rests
from generation import (RULE_WEIGHTS, analyze_source, count_rule_violations,
                        count_rule_violations_by_rule, rule_violation_events,
                        rule_violation_score, strip_soprano_slots,
                        token_to_midi)


# ====== token_to_midi ======

def test_token_to_midi():
    assert token_to_midi('C4') == 60
    assert token_to_midi('C#4') == 61
    assert token_to_midi('B-3') == 58
    assert token_to_midi('E-4^') == 63   # Fermate wird ignoriert
    assert token_to_midi('R') == 'R'
    assert token_to_midi('_') == '_'
    assert token_to_midi('4/4') is None  # Taktart-Token ist keine Note


# ====== Satzregeln (count_rule_violations_by_rule) ======

def test_parallels():
    # Oktavparallele S-B + Quintparallele T-B; alle Stimmen gleichgerichtet
    r = count_rule_violations_by_rule("C5 E4 G3 C3 ; D5 F4 A3 D3 ;".split())
    assert r['parallelen'] == 2
    assert r['alle_gleiche_richtung'] == 1
    # Gegenbewegung, sauber
    assert count_rule_violations("C5 E4 G3 C3 ; B4 F4 G3 G2 ;".split()) == 0
    # Liegenbleibende Quinte (nur eine Stimme bewegt sich): keine Parallele
    assert count_rule_violations("C5 E4 G3 C3 ; D5 _ _ _ ;".split()) == 0


def test_holds_are_neutral():
    assert count_rule_violations("C5 E4 G3 C3 ; _ _ _ _ ; _ _ _ _ ;".split()) == 0


def test_voice_crossing():
    # Alt über Sopran
    r = count_rule_violations_by_rule("C4 E4 G3 C3 ;".split())
    assert r['kreuzung'] == 1 and count_rule_violations("C4 E4 G3 C3 ;".split()) == 1


def test_augmented_second():
    # Tenor G#3 -> F3: 3 Halbtöne auf Nachbar-Stammtönen => übermäßige Sekunde
    r = count_rule_violations_by_rule("C5 E4 G#3 C3 ; _ _ F3 _ ;".split())
    assert r['uebermaessige_sekunde'] == 1
    # Kleine Terz G3 -> E3: gleiche Halbtonzahl, aber keine Nachbar-Stammtöne
    assert count_rule_violations("C5 E4 G3 C3 ; _ _ E3 _ ;".split()) == 0


def test_tritone_leap():
    # Tenor F3 -> B3 ohne Fortsetzung: unaufgelöst (streng)
    r = count_rule_violations_by_rule("C5 E4 F3 C3 ; _ _ B3 _ ;".split())
    assert r['tritonus_unaufgeloest'] == 1 and r['tritonus_aufgeloest'] == 0
    # Mit Schritt-Auflösung in Gegenrichtung (B3 -> A3): milde Variante
    r = count_rule_violations_by_rule("C5 E4 F3 C3 ; _ _ B3 _ ; _ _ A3 _ ;".split())
    assert r['tritonus_aufgeloest'] == 1 and r['tritonus_unaufgeloest'] == 0


def test_large_leaps():
    # Bass-Sprung > Oktave (C3 -> A1)
    r = count_rule_violations_by_rule("C5 E4 G3 C3 ; _ _ _ A1 ;".split())
    assert r['sprung_ueber_oktave'] == 1 and r['septsprung'] == 0
    # Bass-Oktavsprung (genau 12 Halbtöne): erlaubt
    assert count_rule_violations("C5 E4 G3 C3 ; _ _ _ C2 ;".split()) == 0


def test_seventh_leap():
    # Tenor-Septsprung G3 -> F4 (10 Halbtöne)
    r = count_rule_violations_by_rule("C5 A4 G3 C3 ; _ _ F4 _ ;".split())
    assert r['septsprung'] == 1 and r['sprung_ueber_oktave'] == 0


def test_leap_recovery():
    # Bass springt eine kleine Sexte aufwärts (C3 -> A3) und geht danach
    # weiter aufwärts statt schrittweise zurück => Sprung ohne Ausgleich
    r = count_rule_violations_by_rule("C5 A4 E4 C3 ; _ _ _ A3 ; _ _ _ B3 ;".split())
    assert r['sprung_ohne_ausgleich'] == 1
    # Schrittweise Gegenbewegung nach dem Sprung: in Ordnung
    r = count_rule_violations_by_rule("C5 A4 E4 C3 ; _ _ _ A3 ; _ _ _ G3 ;".split())
    assert r['sprung_ohne_ausgleich'] == 0


def test_adjacent_unison():
    # Tenor und Bass im Einklang (bei Bach üblich => milde Variante)
    r = count_rule_violations_by_rule("C5 A4 E4 E4 ;".split())
    assert r['unisono_tb'] == 1 and r['unisono_hoch'] == 0 and r['kreuzung'] == 0
    # Sopran und Alt im Einklang
    r = count_rule_violations_by_rule("C5 C5 F4 F3 ;".split())
    assert r['unisono_hoch'] == 1 and r['unisono_tb'] == 0


def test_voice_spacing():
    # Abstand S-A > Oktave (C5 - A3 = 15 Halbtöne)
    r = count_rule_violations_by_rule("C5 A3 G3 C3 ;".split())
    assert r['abstand'] == 1


def test_overlap():
    # Tenor springt weit über den zuvor klingenden Altton (G3 -> G4 > E4):
    # große Überschneidung + Kreuzung mit dem gehaltenen Alt
    r = count_rule_violations_by_rule("C5 E4 G3 C3 ; _ _ G4 _ ;".split())
    assert r['ueberschneidung_gross'] == 1 and r['ueberschneidung_klein'] == 0
    assert r['kreuzung'] == 1
    # Überschneidung um nur einen Halbton (D4 -> F4 > E4): milde Variante
    r = count_rule_violations_by_rule("C5 E4 D4 G3 ; _ _ F4 _ ;".split())
    assert r['ueberschneidung_klein'] == 1 and r['ueberschneidung_gross'] == 0


def test_pair_idiom_discount():
    """Verdeckte Quinte in Gleichbewegung aller Stimmen = EIN Vorfall:
    nur der teuerste Verstoß zählt (Bach-Idiom, Lift 39.7)."""
    from generation import RULE_WEIGHTS
    # Alle vier Stimmen aufwärts, Außenstimmen in eine Quinte (Sopran springt)
    seq = "C5 A4 E4 G2 ; E5 B4 F4 A2 ;".split()
    r = count_rule_violations_by_rule(seq)
    assert r['verdeckte'] == 1 and r['alle_gleiche_richtung'] == 1
    expected = max(RULE_WEIGHTS['verdeckte'], RULE_WEIGHTS['alle_gleiche_richtung'])
    assert abs(rule_violation_score(seq) - expected) < 1e-9


def test_hidden_fifths():
    # S springt C5 -> E5, Bass G2 -> A2 gleiche Richtung in die Quinte hinein
    r = count_rule_violations_by_rule("C5 A4 E4 G2 ; E5 _ _ A2 ;".split())
    assert r['verdeckte'] == 1
    # Gleiches Zielintervall in Gegenbewegung: keine verdeckte Quinte
    r = count_rule_violations_by_rule("C5 A4 E4 E3 ; E5 _ _ A2 ;".split())
    assert r['verdeckte'] == 0


def test_rule_weights_complete():
    """Jede gezählte Regel hat ein Gewicht (und umgekehrt)."""
    counted = set(count_rule_violations_by_rule([]).keys())
    assert counted == set(RULE_WEIGHTS.keys())


def test_cascade_damping():
    """Mechanisch gekoppelte Verstöße in derselben Gruppe (hier: Kreuzung +
    große Überschneidung desselben Vorfalls) werden gedämpft: der teuerste
    zählt voll, weitere zur Hälfte."""
    seq = "C5 E4 G3 C3 ; _ _ G4 _ ;".split()
    r = count_rule_violations_by_rule(seq)
    assert r['kreuzung'] == 1 and r['ueberschneidung_gross'] == 1
    w_k, w_u = RULE_WEIGHTS['kreuzung'], RULE_WEIGHTS['ueberschneidung_gross']
    expected = max(w_k, w_u) + 0.5 * min(w_k, w_u)
    assert abs(rule_violation_score(seq) - expected) < 1e-9


# ====== Mehrschrittige / mehrstimmige Regeln ======

def test_akzent_parallelen():
    # Quinte S-B auf zwei aufeinanderfolgenden Schlägen (D5/G3 -> C5/F3),
    # dazwischen Durchgangston im Bass (A3): direkte Regel greift nicht,
    # die Schlag-Ebene erkennt die Akzentparallele
    seq = ("D5 A4 F#4 G3 ; _ _ _ _ ; _ _ _ A3 ; _ _ _ _ ; "
           "C5 _ _ F3 ; _ _ _ _ ;")
    r = count_rule_violations_by_rule(seq.split())
    assert r['akzent_parallelen'] == 1 and r['parallelen'] == 0


def test_leerer_klang():
    # Leerer Klang (nur C und G) auf einem Schlag mitten im Stück
    seq = ("C5 E4 G3 C3 ; _ _ _ _ ; _ _ _ _ ; _ _ _ _ ; "
           "C5 G4 G3 C3 ; _ _ _ _ ; _ _ _ _ ; _ _ _ _ ; "
           "C5 E4 G3 C3 ; _ _ _ _ ;")
    r = count_rule_violations_by_rule(seq.split())
    assert r['leerer_klang'] == 1
    # Schlussakkord-Ausnahme: leerer Klang als letzter Akkord ist erlaubt
    seq = ("C5 E4 G3 C3 ; _ _ _ _ ; _ _ _ _ ; _ _ _ _ ; "
           "C5 G4 G3 C3 ; _ _ _ _ ;")
    r = count_rule_violations_by_rule(seq.split())
    assert r['leerer_klang'] == 0


def test_terzverdopplung_dur():
    # C-Dur mit verdoppelter Terz (zwei E)
    r = count_rule_violations_by_rule("E5 E4 G3 C3 ;".split())
    assert r['terzverdopplung_dur'] == 1
    # c-Moll mit verdoppelter Terz: von der Regel nicht erfasst
    r = count_rule_violations_by_rule("E-5 E-4 G3 C3 ;".split())
    assert r['terzverdopplung_dur'] == 0


def test_ton_wiederholung():
    # Alt schlägt denselben Ton dreimal in Folge an
    r = count_rule_violations_by_rule("C5 E4 G3 C3 ; _ E4 _ _ ; _ E4 _ _ ;".split())
    assert r['ton_wiederholung'] == 1
    # Zwei Anschläge: noch keine Kette
    r = count_rule_violations_by_rule("C5 E4 G3 C3 ; _ E4 _ _ ;".split())
    assert r['ton_wiederholung'] == 0


def test_stagnation():
    # Über einen ganzen 4/4-Takt hinaus kein einziger Onset in A/T/B
    seq = "C5 E4 G3 C3 ; " + "_ _ _ _ ; " * 17 + "_ D4 A3 F3 ;"
    r = count_rule_violations_by_rule(seq.split())
    assert r['stagnation'] == 1, r
    # Fermaten-Ausnahme: liegt eine Fermate im Fenster, ist Stillstand erlaubt
    seq = "C5^ E4 G3 C3 ; " + "_ _ _ _ ; " * 17 + "_ D4 A3 F3 ;"
    r = count_rule_violations_by_rule(seq.split())
    assert r['stagnation'] == 0


# ====== ParallelPenaltyProcessor (Parallelen-Check im Sampling) ======

def test_parallel_penalty_processor():
    import torch
    from generation import ParallelPenaltyProcessor
    from music_tokenizer import MusicTokenizer

    tok = MusicTokenizer()
    tok.build_vocab([("C5 F4 G4 B4 D5 G3 C3 ; _ |", "R")])

    # Historie: Gruppe 1 = C5/F4 (Quinte S-A), Gruppe 2: Sopran ist auf D5
    # gewechselt, der Alt-Slot wird gerade erzeugt.
    history = "C5 F4 G3 C3 ; D5".split()
    ids = [tok.cls_token_id] + [tok.convert_tokens_to_ids(t) for t in history]
    input_ids = torch.tensor([ids])
    proc = ParallelPenaltyProcessor(group_starts=[0, 5], music_tokenizer=tok, penalty=6.0)
    scores = proc(input_ids, torch.zeros(1, len(tok)))

    g4 = tok.convert_tokens_to_ids('G4')   # D5-G4 = wieder Quinte => Parallele
    b4 = tok.convert_tokens_to_ids('B4')   # D5-B4 = Terz => erlaubt
    f4 = tok.convert_tokens_to_ids('F4')   # Ton halten (kein 'bewegt') => erlaubt
    us = tok.convert_tokens_to_ids('_')
    assert scores[0, g4] < 0, "parallele Quinte muss bestraft werden"
    assert scores[0, b4] == 0 and scores[0, f4] == 0 and scores[0, us] == 0

    # Gegenprobe: hält der Sopran (D5 -> '_'), gibt es keine Parallele
    history2 = "C5 F4 G3 C3 ; _".split()
    ids2 = [tok.cls_token_id] + [tok.convert_tokens_to_ids(t) for t in history2]
    scores2 = proc(torch.tensor([ids2]), torch.zeros(1, len(tok)))
    assert torch.all(scores2 == 0)


# ====== analyze_source (Phrasen, Fermaten, Zielpositionen) ======

def test_split_into_phrases():
    from data_prep import split_into_phrases
    src = "4/4 G:major C5 _ _ _ E5^ _ _ _ | G5 _ _ _ C5 _ _ _"
    tgt = ("C5 G4 E4 C3 ; _ _ _ _ ; _ _ _ _ ; _ _ _ _ ; "
           "E5^ G4 C4 C3 ; _ _ _ _ ; _ _ _ _ ; _ _ _ _ ; | "
           "G5 G4 E4 C3 ; _ _ _ _ ; _ _ _ _ ; _ _ _ _ ; "
           "C5 G4 E4 C3 ; _ _ _ _ ; _ _ _ _ ; _ _ _ _ ;")
    phrases = split_into_phrases(src, tgt, min_steps=4)
    assert len(phrases) == 2
    # Präfix (Taktart + Tonart) steht vor jeder Phrase
    assert phrases[0][0].startswith("4/4 G:major") and phrases[1][0].startswith("4/4 G:major")
    assert '^' in phrases[0][0] and '^' not in phrases[1][0]
    # Ziel-Phrasen ergeben zusammengesetzt wieder das Original
    assert ' '.join(p[1] for p in phrases).split() == tgt.split()
    # Einphrasige Stücke liefern nichts (wären Duplikate)
    assert split_into_phrases("4/4 C5 _ _ _", "C5 G4 E4 C3 ; _ _ _ _ ; _ _ _ _ ; _ _ _ _ ;") == []


def test_analyze_source():
    # Tonart-Token wird wie das Taktart-Token übersprungen
    info = analyze_source("4/4 G:major C5 _ _ _")
    assert len(info['step_tokens']) == 4 and info['group_starts'][0] == 0

    info = analyze_source("4/4 C5 _ _ _ E5^ _ _ _ | G5 _ R _")
    # 12 Schritte à 5 Zieltokens + 1 Taktstrich
    assert len(info['step_tokens']) == 12
    assert info['total_len'] == 12 * 5 + 1
    assert info['bar_positions'] == [40]
    assert info['group_starts'][8] == 41       # erste Gruppe nach dem Taktstrich
    assert info['fermata_steps'] == {4}
    # Phrasen: Stückanfang + erster Onset nach der Fermate (G5, Schritt 8)
    assert info['phrase_start_steps'] == [0, 8]


def test_rule_events_match_counts():
    seq = "C5 E4 G3 C3 ; D5 F4 A3 D3 ; _ _ G4 _ ;".split()
    agg = {}
    for gi, rule in rule_violation_events(seq):
        assert 0 <= gi < 3
        agg[rule] = agg.get(rule, 0) + 1
    counts = count_rule_violations_by_rule(seq)
    assert all(counts[k] == agg.get(k, 0) for k in counts)
    assert sum(counts.values()) == count_rule_violations(seq)


# ====== strip_soprano_slots ======

def test_strip_soprano_slots():
    assert strip_soprano_slots("C5 A4 F4 F3 ; _ _ _ _ ; | D5 B4 G4 G3 ;") == \
        "A4 F4 F3 ; _ _ _ ; | B4 G4 G3 ;"


# ====== trim_trailing_rests ======

def test_key_token_filtered_in_export():
    from bach_chorales import parse_soprano_string
    part = parse_soprano_string("3/4 F#:minor F#4 _ _ _")
    elems = list(part.notesAndRests)
    assert len(elems) == 1 and elems[0].nameWithOctave == 'F#4'


def test_trim_trailing_rests():
    assert trim_trailing_rests(['C4', '_', 'R', '_', '_']) == ['C4', '_']
    assert trim_trailing_rests(['C4', '_', 'R', '_', 'R', '_']) == ['C4', '_']
    assert trim_trailing_rests(['C4', '_', '_', '_']) == ['C4', '_', '_', '_']
    assert trim_trailing_rests(['R', '_', 'C4', '_']) == ['R', '_', 'C4', '_']
    assert trim_trailing_rests(['R', '_']) == []
    assert trim_trailing_rests([]) == []


# ====== Export (MusicXML) ======

def test_export_no_trailing_rests():
    """Stimmen enden unterschiedlich lang mit Schlusspausen -> Export hat
    keinerlei Pausen, korrekte Taktart, gemeinsames Ende an der Taktgrenze."""
    from music21 import converter

    src = "3/4 C5 _ _ _ E5 _ _ _ G5^ _ _ _ | R _ _ _"
    tgt = ("G4 E4 C3 ; _ _ _ ; E4 _ C3 ; _ _ _ ; E4 C4 C3 ; _ _ _ ; | "
           "R R R ; _ _ _ ; R R R ; _ _ _ ; R R R ; _ _ _ ; R R R ; _ _ _ ;")
    with tempfile.TemporaryDirectory() as d:
        output_chorale(src, tgt, os.path.join(d, 't'))
        score = converter.parse(os.path.join(d, 't.mxl'))
        assert len(score.parts) == 4
        for part in score.parts:
            elems = list(part.flatten().notesAndRests)
            assert elems and not any(e.isRest for e in elems)
            assert part.flatten().getTimeSignatures()[0].ratioString == '3/4'
            assert abs(part.flatten().highestTime - 3.0) < 1e-6


def test_export_keeps_inner_rests():
    """Pausen mitten im Stück (Atempausen) bleiben erhalten."""
    from music21 import converter

    src = "4/4 C5 _ _ _ R _ _ _ E5 _ _ _ G5 _ _ _"
    tgt = ("G4 E4 C3 ; _ _ _ ; _ _ _ ; _ _ _ ; R R R ; _ _ _ ; _ _ _ ; _ _ _ ; "
           "G4 E4 C3 ; _ _ _ ; _ _ _ ; _ _ _ ; G4 E4 C3 ; _ _ _ ; _ _ _ ; _ _ _ ;")
    with tempfile.TemporaryDirectory() as d:
        output_chorale(src, tgt, os.path.join(d, 't2'))
        score = converter.parse(os.path.join(d, 't2.mxl'))
        for part in score.parts:
            elems = list(part.flatten().notesAndRests)
            assert sum(1 for e in elems if e.isRest) == 1
            assert not elems[-1].isRest


# ====== Kalibrierung an echtem Bach (nur wenn Datensatz vorhanden) ======

def test_rules_calibrated_on_bach():
    """Echte Bach-Sätze müssen unter den (gewichteten) Satzregeln niedrig
    scoren, sonst würde das Reranking normalen Bach bestrafen. Zusätzlich:
    streng gewichtete Regeln (3.0) müssen bei Bach tatsächlich selten sein —
    das rechtfertigt die inverse Gewichtung.

    Messung auf dem vollen Korpus (343 Choräle, Originaltonarten):
    Verstöße pro 100 Sechzehntel, nach Häufigkeit sortiert —
      stagnation 0.00, sprung_ueber_oktave 0.05, tritonus_unaufgeloest 0.10,
      septsprung 0.10, uebermaessige_sekunde 0.14, parallelen 0.18,
      verdeckte 0.19, tritonus_aufgeloest 0.27, leerer_klang 0.32,
      akzent_parallelen 0.48, abstand 0.54, kreuzung 0.80,
      ueberschneidung_gross 0.91, sprung_ohne_ausgleich 0.97,
      ueberschneidung_klein 1.06, unisono_tb 1.17, unisono_hoch 1.20,
      alle_gleiche_richtung 1.23, terzverdopplung_dur 1.70,
      ton_wiederholung 2.94."""
    import json
    import statistics

    import config as C
    if not os.path.exists(C.RAW_DATA_FILE):
        print("  (übersprungen: Datensatz fehlt)")
        return
    data = json.load(open(C.RAW_DATA_FILE))
    per_100_weighted = []
    strict_totals = {name: 0 for name, w in RULE_WEIGHTS.items() if w >= 3.0}
    total_steps = 0
    for idx in range(6, min(len(data), 720), 12):  # Originaltonarten (Stichprobe)
        src, tgt = data[idx]
        steps = max(1, len([t for t in src.split()
                            if t != '|' and '/' not in t and ':' not in t]))
        total_steps += steps
        per_100_weighted.append(100 * rule_violation_score(tgt.split()) / steps)
        for name, n in count_rule_violations_by_rule(tgt.split()).items():
            if name in strict_totals:
                strict_totals[name] += n
    # Log-Ratio-Gewichte (mit Parallelen-Check im Sampling gemessen)
    # + Kaskaden-Dämpfung + Idiom-Erlass (PAIR_IDIOMS): median 1.74 (voller Korpus)
    median = statistics.median(per_100_weighted)
    assert median < 4, f"Bach-Median {median:.1f} gewichtete Verstöße/100 — Gewichte zu streng?"
    for name, n in strict_totals.items():
        rate = 100 * n / total_steps
        assert rate < 0.5, f"Regel '{name}' ist streng gewichtet, aber Bach verstößt {rate:.2f}/100 dagegen"


# ====== Runner ======

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"✓ {name}")
        except AssertionError as e:
            failed += 1
            print(f"✗ {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} Tests bestanden")
    sys.exit(1 if failed else 0)