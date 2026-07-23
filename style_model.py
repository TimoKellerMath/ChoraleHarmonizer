"""Stilmodelle jenseits der Einzelregeln: Harmonik-Bigramme und Textur.

Zwei Bewertungs-Terme für das Best-of-N-Reranking (kein Neutraining nötig):

  Harmonik:  Bigramm-Modell über die Schlag-Akkorde (Pitch-Class-Mengen
             relativ zur Tonika, Dur/Moll getrennt) — bestraft Fortschreitungen,
             die Bach nie macht, inklusive unplausiblem harmonischem Rhythmus.
  Textur:    Onset-Wahrscheinlichkeit je Stimme und Taktposition (je Taktart) —
             bestraft zu statische wie zu geschäftige Sätze.

Wichtig gegen "Durchschnitts-Bach": Beide Terme bewerten die ABWEICHUNG vom
Bach-typischen Überraschungsniveau, |NLL − Ziel|, nicht die NLL selbst.
Reines NLL-Minimieren würde immer die häufigsten Fortschreitungen bevorzugen;
mit einem Ziel auf Bachs eigenem Median wird genau Bachs Kühnheitsniveau
angestrebt — und über style_config.json (target_delta, Presets
konservativ..kuehn) lässt es sich gezielt verschieben.

Statistiken werden einmalig aus C.RAW_DATA_FILE gezählt (Originaltonarten)
und in C.STYLE_STATS_FILE gecacht. Alle Parameter: style_config.json;
enabled=false oder weight=0 schaltet einen Term ab, fehlende Config-Datei
schaltet beide ab.
"""

import json
import math
import statistics
from pathlib import Path

import config as C

NOTE_BASE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


# ====== TOKEN-/QUELL-ANALYSE (bewusst ohne Import aus generation.py) ======

def _pc_of(tok):
    """Token -> Pitch-Class 0..11; '_'/'R' unverändert; sonst None."""
    tok = tok.rstrip('^')
    if tok in ('R', '_'):
        return tok
    if not tok or tok[0] not in NOTE_BASE:
        return None
    pc = NOTE_BASE[tok[0]]
    for ch in tok[1:]:
        if ch == '#':
            pc += 1
        elif ch == '-':
            pc -= 1
        else:
            break
    return pc % 12


def parse_key(source):
    """Tonart-Token ("G:major") -> (Tonika-PC, Modus) bzw. (None, None)."""
    tok = next((t for t in source.split() if ':' in t), None)
    if tok is None:
        return None, None
    name, _, mode = tok.partition(':')
    if not name or name[0] not in NOTE_BASE:
        return None, None
    pc = NOTE_BASE[name[0]]
    for ch in name[1:]:
        pc += 1 if ch == '#' else -1 if ch == '-' else 0
    return pc % 12, mode


def parse_meter(source):
    return next((t for t in source.split() if '/' in t), '4/4')


def steps_per_measure(meter):
    try:
        num, den = meter.split('/')
        return max(1, round(16 * int(num) / int(den)))
    except (ValueError, ZeroDivisionError):
        return 16


def _grid(tokens):
    """Zielsequenz -> Liste (Gruppe, 16tel-Index im Takt)."""
    groups, g, idx = [], [], 0
    for tok in tokens:
        if tok == '|':
            idx = 0
            continue
        if tok == ';':
            if len(g) == 4:
                groups.append((g, idx))
                idx += 1
            g = []
        else:
            g.append(tok)
    return groups


def beat_chords(tokens, tonic_pc):
    """Klingender Akkord an jedem Schlag als Signatur "0.4.7" (PCs relativ
    zur Tonika, aufsteigend); nur Schläge mit >= 3 klingenden Stimmen."""
    chords = []
    cur = [None, None, None, None]
    for g, idx in _grid(tokens):
        for v, tok in enumerate(g):
            pc = _pc_of(tok)
            if pc == '_' or pc is None:
                continue
            cur[v] = None if pc == 'R' else pc
        if idx % 4 == 0:
            sounding = [p for p in cur if p is not None]
            if len(sounding) >= 3:
                rel = sorted({(p - tonic_pc) % 12 for p in sounding})
                chords.append('.'.join(str(p) for p in rel))
    return chords


def texture_events(tokens, spm):
    """(Stimme 1..3, Taktposition, ist_Onset) für jeden A/T/B-Slot."""
    events = []
    for g, idx in _grid(tokens):
        pos = idx % spm
        for v in (1, 2, 3):
            events.append((v, pos, g[v] != '_'))
    return events


# ====== NLL unter den Bach-Statistiken ======

def chord_nll(chords, bigrams, unigrams):
    """Mittlere NLL der Akkordübergänge; Interpolation Bigramm/Unigramm."""
    if len(chords) < 2:
        return None
    total_uni = sum(unigrams.values())
    vocab = len(unigrams) + 1
    s = n = 0
    for a, b in zip(chords, chords[1:]):
        row = bigrams.get(a, {})
        row_total = sum(row.values())
        p_bi = row.get(b, 0) / row_total if row_total else 0.0
        p_uni = (unigrams.get(b, 0) + 1) / (total_uni + vocab)
        p = 0.7 * p_bi + 0.3 * p_uni
        s += -math.log(p)
        n += 1
    return s / n


def texture_nll(events, table, spm):
    """Mittlere NLL der Onset-/Halte-Entscheidungen unter der Bach-Tabelle."""
    s = n = 0
    for v, pos, onset in events:
        counts = table.get(str(v))
        if counts is None or pos >= len(counts):
            continue
        on, tot = counts[pos]
        p_on = (on + 1) / (tot + 2)
        p = p_on if onset else 1.0 - p_on
        s += -math.log(max(p, 1e-9))
        n += 1
    return s / n if n else None


# ====== STATISTIK AUFBAUEN / CACHEN ======

STATS_VERSION = 1


def build_stats():
    """Zählt Harmonik- und Textur-Statistiken über alle Originaltonarten
    und legt zusätzlich Bachs Ziel-NLLs (Mediane) ab."""
    raw = json.load(open(C.RAW_DATA_FILE, encoding='utf-8'))
    entries = [raw[i] for i in range(6, len(raw), 12)]

    bigrams = {}   # mode -> prev -> next -> count
    unigrams = {}  # mode -> sig -> count
    texture = {}   # meter -> voice(str) -> [[onsets, total] je Position]

    per_piece = []  # (mode, meter, chords, events, spm) für den 2. Durchgang
    for src, tgt in entries:
        tonic, mode = parse_key(src)
        meter = parse_meter(src)
        spm = steps_per_measure(meter)
        toks = tgt.split()
        chords = beat_chords(toks, tonic) if tonic is not None else []
        events = texture_events(toks, spm)
        per_piece.append((mode, meter, chords, events, spm))

        if mode is not None:
            uni = unigrams.setdefault(mode, {})
            bi = bigrams.setdefault(mode, {})
            for c in chords:
                uni[c] = uni.get(c, 0) + 1
            for a, b in zip(chords, chords[1:]):
                bi.setdefault(a, {})[b] = bi.get(a, {}).get(b, 0) + 1
        table = texture.setdefault(meter, {str(v): [[0, 0] for _ in range(spm)]
                                           for v in (1, 2, 3)})
        for v, pos, onset in events:
            if pos < len(table[str(v)]):
                cell = table[str(v)][pos]
                cell[0] += int(onset)
                cell[1] += 1

    # 2. Durchgang: Bachs eigene NLL-Verteilung -> Ziele (Mediane)
    chord_nlls, texture_nlls = [], []
    for mode, meter, chords, events, spm in per_piece:
        if mode in bigrams:
            nll = chord_nll(chords, bigrams[mode], unigrams[mode])
            if nll is not None:
                chord_nlls.append(nll)
        table = texture.get(meter)
        if table:
            nll = texture_nll(events, table, spm)
            if nll is not None:
                texture_nlls.append(nll)

    stats = {
        'version': STATS_VERSION,
        'chord_bigrams': bigrams,
        'chord_unigrams': unigrams,
        'texture': texture,
        'chord_target_nll': statistics.median(chord_nlls),
        'chord_nll_p90': sorted(chord_nlls)[int(0.9 * len(chord_nlls))],
        'texture_target_nll': statistics.median(texture_nlls),
        'texture_nll_p90': sorted(texture_nlls)[int(0.9 * len(texture_nlls))],
    }
    Path(C.STYLE_STATS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(C.STYLE_STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f)
    return stats


def load_stats():
    p = Path(C.STYLE_STATS_FILE)
    if p.exists() and p.stat().st_mtime >= Path(C.RAW_DATA_FILE).stat().st_mtime:
        stats = json.load(open(p, encoding='utf-8'))
        if stats.get('version') == STATS_VERSION:
            return stats
    return build_stats()


# ====== SCORER ======

class StyleScorer:
    """Bewertet |NLL − Ziel| für Harmonik und Textur, gewichtet laut Config.

    cfg-Format: siehe style_config.json. Ziele: target_nll (null = Bachs
    Median aus den Statistiken) + target_delta des aktiven Presets —
    positive Deltas erlauben/bevorzugen kühnere Sätze als der Bach-Median.
    """

    def __init__(self, stats, cfg):
        self.stats = stats
        self.cfg = cfg or {}
        self.preset = {}
        self.preset_name = None

    def set_preset(self, name):
        presets = self.cfg.get('presets', {})
        if name not in presets:
            raise KeyError(f"Unbekanntes Preset '{name}' — verfügbar: {sorted(presets)}")
        self.preset = presets[name]
        self.preset_name = name

    def _term(self, key):
        term = self.cfg.get(key, {})
        if not term.get('enabled', False):
            return None
        weight = float(term.get('weight', 0.0))
        if weight <= 0.0:
            return None
        return term, weight

    @property
    def active(self):
        return self._term('chord_lm') is not None or self._term('texture') is not None

    def _deviation(self, nll, term, stats_target, delta_key):
        target = term.get('target_nll')
        if target is None:
            target = self.stats[stats_target]
        target += float(self.preset.get(delta_key, 0.0))
        tolerance = float(term.get('tolerance', 0.0))
        return max(0.0, abs(nll - target) - tolerance)

    def penalty(self, tokens, source):
        """Gewichtete Stil-Abweichung einer (Teil-)Zielsequenz."""
        total = 0.0
        chord_term = self._term('chord_lm')
        if chord_term is not None:
            term, weight = chord_term
            tonic, mode = parse_key(source)
            if tonic is not None and mode in self.stats['chord_bigrams']:
                nll = chord_nll(beat_chords(tokens, tonic),
                                self.stats['chord_bigrams'][mode],
                                self.stats['chord_unigrams'][mode])
                if nll is not None:
                    total += weight * self._deviation(
                        nll, term, 'chord_target_nll', 'chord_target_delta')
        texture_term = self._term('texture')
        if texture_term is not None:
            term, weight = texture_term
            meter = parse_meter(source)
            table = self.stats['texture'].get(meter)
            if table:
                spm = steps_per_measure(meter)
                nll = texture_nll(texture_events(tokens, spm), table, spm)
                if nll is not None:
                    total += weight * self._deviation(
                        nll, term, 'texture_target_nll', 'texture_target_delta')
        return total


# ====== SINGLETON + PRESET-VERWALTUNG ======

_scorer = None
_gen_defaults = None


def get_style_scorer():
    """Lazy Singleton; ohne style_config.json sind die Stil-Terme aus."""
    global _scorer
    if _scorer is None:
        cfg = None
        if Path(C.STYLE_CONFIG_FILE).exists():
            cfg = json.load(open(C.STYLE_CONFIG_FILE, encoding='utf-8'))
        stats = load_stats() if cfg else {}
        _scorer = StyleScorer(stats, cfg)
    return _scorer


def set_preset(name):
    """Aktiviert ein Preset aus style_config.json: verschiebt die Stil-Ziele
    und überschreibt optional Sampling-Parameter (temperature, top_p,
    underscore_bias). Presets wirken bis zum nächsten set_preset."""
    global _gen_defaults
    scorer = get_style_scorer()
    scorer.set_preset(name)
    if _gen_defaults is None:
        _gen_defaults = (C.GEN_TEMPERATURE, C.GEN_TOP_P, C.UNDERSCORE_BIAS)
    C.GEN_TEMPERATURE = float(scorer.preset.get('temperature', _gen_defaults[0]))
    C.GEN_TOP_P = float(scorer.preset.get('top_p', _gen_defaults[1]))
    C.UNDERSCORE_BIAS = float(scorer.preset.get('underscore_bias', _gen_defaults[2]))