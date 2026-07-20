"""Ein-/Ausgabe für ChoraleHarmonizer.

Zwei Richtungen:
  music21 -> data:  Bach-Choräle aus dem music21-Korpus in die textuelle
                    Token-Darstellung (Format v2.1) exportieren; 12
                    Transpositionen je Choral (BachChorales, to_string).
  data -> XML/MIDI: Token-Sequenzen zurück in Noten (parse_*, output_chorale).

Datensatz erzeugen: python bach_chorales.py
"""

import os
import time
from music21 import corpus, note, chord, expressions, metadata, meter, midi, stream, environment
import json
from pathlib import Path

# 16-tel = 4
# 8-tel = 2
min_note_length = 4

# Funktion, um die Oberstimme im gewünschten Format zu konvertieren
def soprano_to_string(n, is_repeat=False):
    fermata = True if n.expressions and any(isinstance(exp, expressions.Fermata) for exp in n.expressions) else False
    if is_repeat:
        pitch = '_'
    else:
        pitch = n.nameWithOctave if isinstance(n, note.Note) else 'R'
        if fermata:
            pitch += '^'  # augmentiere Note mit Fermate beim ersten (nicht wiederholten) Auftreten
        #pitch += '(' + n.beatStr + ')'
    return (pitch, fermata)

def note_to_string(n, is_repeat=False):
    if is_repeat:
        pitch = '_'
    else:
        pitch = n.nameWithOctave if isinstance(n, note.Note) else 'R'
    return pitch

def voice_to_string(voice, note_to_string_fn, cut_off_at_fermata, max_length=0):
    voice_notes = []
    length = 0
    for n in voice:
        duration_in_sixteenths = int(n.quarterLength * min_note_length)
        if n.quarterLength < 1.0/min_note_length - 1e-6:
            raise ValueError(f"Note/Pause kürzer als {min_note_length*8}-tel gefunden: {n.quarterLength} bei {n}")
        voice_notes.append(note_to_string_fn(n))
        for _ in range(1, duration_in_sixteenths):
            voice_notes.append('_')
        length += duration_in_sixteenths
        if cut_off_at_fermata and (max_length > 0) and (length >= max_length):
            break
    return voice_notes

def to_string(soprano, lower_voices_parts, cut_off_at_fermata, time_signature=None, key_token=None):
    # Sopran auf ein 16tel-Raster legen: genau ein Token pro 16tel
    # (Note, mit ^ bei Fermate am Onset; '_' für Haltungen)
    soprano_grid = []
    barline_positions = set()  # 16tel-Positionen, an denen ein | im Sopran steht
    length = 0
    prev_beat = 0.0  # damit wir den Taktwechsel erkennen können
    for n in soprano:
        duration_in_sixteenths = int(n.quarterLength * min_note_length)
        if n.quarterLength < 1.0/min_note_length - 1e-6:
            raise ValueError(f"Note/Pause kürzer als {min_note_length*8}-tel gefunden: {n.quarterLength} bei {n}")

        current_beat = float(n.beat)
        if (current_beat == 1.0 and prev_beat > 1.0):
            barline_positions.add(length)  # gleiche Position für Unterstimmen merken

        soprano_note, fermata = soprano_to_string(n)
        soprano_grid.append(soprano_note)
        for _ in range(1, duration_in_sixteenths):
            soprano_grid.append('_')
        length += duration_in_sixteenths

        prev_beat = current_beat + n.quarterLength

        if cut_off_at_fermata and fermata:
            break

    soprano_notes = []
    if time_signature is not None:
        # Taktart als globales Token an den Anfang der Quellsequenz
        # (z.B. "4/4", "3/4", "2/2"): 4/4 und alla breve sind aus den
        # Taktstrichen allein nicht unterscheidbar, prägen aber
        # harmonischen Rhythmus und Betonung.
        soprano_notes.append(time_signature)
    if key_token is not None:
        # Tonart als globales Token (z.B. "G:major"): Dur/Moll und Tonika
        # muss das Modell sonst aus den Noten erraten — gerade an Kadenzen
        # die härteste Teilaufgabe.
        soprano_notes.append(key_token)
    for pos, token in enumerate(soprano_grid):
        if pos in barline_positions:
            soprano_notes.append('|')
        soprano_notes.append(token)

    lower_voices = [voice_to_string(voice, note_to_string, cut_off_at_fermata, max_length=length)
                    for voice in lower_voices_parts]
    if len(lower_voices_parts) == 1:
        lower_voices3 = []
        for pos, token in enumerate(lower_voices[0]):
            if pos in barline_positions:
                lower_voices3.append('|')
            lower_voices3.append(token)
    else:
        # Ziel-Format pro 16tel: "S A T B ;" — der Sopran wird in die
        # Zielsequenz eingewoben (Idee aus DeepBach/constraint-transformer:
        # der Decoder sieht Melodie, Metrum und Fermaten LOKAL an jeder
        # Position, statt sie per Cross-Attention abzählen zu müssen).
        # Bei der Generierung werden die Sopran-Slots forciert, sie kosten
        # also nichts — sie liefern nur Kontext.
        lower_voices2 = list(map(list, zip(*lower_voices)))  # Liste transponieren
        lower_voices2 = lower_voices2[:len(soprano_grid)]
        lower_voices3 = []
        for pos, beat in enumerate(lower_voices2):
            if pos in barline_positions:
                lower_voices3.append('|')
            lower_voices3.append(soprano_grid[pos] + ' ' + " ".join(beat) + ' ;')
    return (" ".join(soprano_notes), " ".join(lower_voices3))

"""
def to_string(soprano, lower_voices_parts, cut_off_at_fermata):
    soprano_notes = []
    length = 0
    last_measure_number = -1
    
    for n in soprano:
        duration_in_sixteenths = int(n.quarterLength * min_note_length + 1e-10)
        
        if n.quarterLength < 1.0/min_note_length - 1e-6:
            raise ValueError(f"Note/Pause kürzer als {min_note_length*8}-tel: {n.quarterLength}")
        
        # Taktwechsel erkennen über measureNumber
        if hasattr(n, 'measureNumber') and n.measureNumber is not None:
            if (n.measureNumber != last_measure_number) and (last_measure_number != -1):
                soprano_notes.append('|')
                last_measure_number = n.measureNumber
        
        soprano_note, fermata = soprano_to_string(n)
        soprano_notes.append(soprano_note)
        
        for _ in range(1, duration_in_sixteenths):
            soprano_notes.append('_')
        
        length += duration_in_sixteenths
        
        if cut_off_at_fermata and fermata:
            break
    
    lower_voices = [voice_to_string(voice, note_to_string, cut_off_at_fermata, max_length=length)
                    for voice in lower_voices_parts]
    if len(lower_voices_parts) == 1:
        lower_voices3 = lower_voices[0]
    else:
        lower_voices2 = list(map(list, zip(*lower_voices)))  # Liste transponieren
        lower_voices3 = [" ".join(beat) + ' ;' for beat in lower_voices2]  # Noten auf Schlag zusammenfassen
    return (" ".join(soprano_notes), " ".join(lower_voices3))
"""

def parse_voice(notes):
    part = stream.Part()
    # eine 32-tel als Grundeinheit
    min_duration = 1.0/min_note_length
    new_note_duration = 0.0
    old_note = None
    for item in notes:
        if item == '_':
            # Wiederholungszeichen: verlängere die aktuelle Note
            new_note_duration += min_duration
            continue
        if '^' in item:
            # Abschluss des vorherigen offenen Tons
            if old_note is not None and new_note_duration > 0:
                old_note.quarterLength = new_note_duration
                part.append(old_note)
            pitch = item.replace('^', '')
            old_note = note.Note(pitch)
            new_note_duration = min_duration
            old_note.expressions.append(expressions.Fermata())
        else:
            if old_note is not None and new_note_duration > 0:
                old_note.quarterLength = new_note_duration
                part.append(old_note)
            if item == 'R':
                old_note = note.Rest()
            else:
                old_note = note.Note(item)
            new_note_duration = min_duration
    # letzten offenen Ton anhängen (falls vorhanden)
    if old_note is not None and new_note_duration > 0:
        old_note.quarterLength = new_note_duration
        part.append(old_note)
    return part


def trim_trailing_rests(tokens):
    """Schneidet Pausen am Stückende ab: die Stimme endet mit dem letzten
    klingenden Ton (inkl. seiner Haltungen); nachfolgende R/_ entfallen."""
    end = len(tokens)
    while end > 0:
        i = end - 1
        while i >= 0 and tokens[i] == '_':
            i -= 1
        if i >= 0 and tokens[i].startswith('R'):
            end = i  # Pause samt zugehöriger Haltungen abschneiden
        else:
            break
    return tokens[:end]


# Funktion zum Parsen und Umwandeln des Soprans in music21-Objekte
def parse_soprano_string(soprano_string):
    # Taktstriche sowie Taktart- ('/') und Tonart-Token (':') herausfiltern
    soprano_notes = [t for t in soprano_string.split()
                     if t != '|' and '/' not in t and ':' not in t]
    return parse_voice(trim_trailing_rests(soprano_notes))


# Funktion zum Parsen und Umwandeln der Unterstimmen in music21-Objekte
def parse_lower_voices_string(lower_voices_string):
    lower_voices_string = lower_voices_string.replace('|', '')
    # robustes Splitten, ignore empty groups
    lower_voices_parts = [p.strip() for p in lower_voices_string.split(';') if p.strip()]
    if not lower_voices_parts:
        return []
    num_lower_voices = len(lower_voices_parts[0].split())
    lower_voices = [[] for _ in range(num_lower_voices)]

    for voice_group in lower_voices_parts:
        pitches = voice_group.split()
        for i, pitch in enumerate(pitches[:num_lower_voices]):
            lower_voices[i].append(pitch)

    return [parse_voice(trim_trailing_rests(voice)) for voice in lower_voices]


def output_chorale(soprano_input, lower_voices_input, filename):
    soprano_part = parse_soprano_string(soprano_input)
    lower_voices_part = parse_lower_voices_string(lower_voices_input)
    parts = [soprano_part] + lower_voices_part

    # Taktart aus dem Quell-String (z.B. "4/4") für korrekte Taktstriche im Export
    ts_token = next((t for t in soprano_input.split() if '/' in t), None)
    measure_len = 4.0
    if ts_token:
        try:
            measure_len = meter.TimeSignature(ts_token).barDuration.quarterLength
        except Exception:
            ts_token = None

    # Gemeinsames Ende auf voller Taktgrenze: der Schlusston jeder Stimme wird
    # bis dorthin verlängert. Ohne das füllt music21 unvollständige Schlusstakte
    # beim Export mit (grau dargestellten) Pausen auf.
    import math
    end = max((p.highestTime for p in parts if len(p) > 0), default=0.0)
    end = math.ceil(end / measure_len - 1e-6) * measure_len
    for part in parts:
        elems = list(part.notesAndRests)
        if elems and not elems[-1].isRest:
            gap = end - part.highestTime
            if gap > 1e-6:
                elems[-1].quarterLength += gap
        if ts_token:
            part.insert(0, meter.TimeSignature(ts_token))

    choral = stream.Score()
    # WICHTIG: insert(0, ...) statt append(...) — append hängt die Parts
    # zeitlich HINTEREINANDER; der MusicXML-Export füllt dann jede Stimme
    # bis zur (vervielfachten) Gesamtlänge mit Pausen auf.
    for voice in parts:
        choral.insert(0, voice)

    midi_filename = os.path.join(filename + '.mid')
    choral.write('midi', fp=midi_filename)

    musicxml_filename = os.path.join(filename + '.mxl')
    choral.write('musicxml', fp=musicxml_filename)

    """
    midi_filename = os.path.join(filename + '.mid')
    midi_file = midi.translate.music21ObjectToMidiFile(choral)
    midi_file.open(midi_filename, 'wb')
    midi_file.write()
    midi_file.close()
    """


class VocabTokenizer:
    def __init__(self, unk_token='[UNK]'):
        self.unk_token = unk_token
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.dict = {t: i for i, t in enumerate(self.special_tokens)}
        self.inverse_dict = {i: t for i, t in enumerate(self.special_tokens)}

    def get_vocab_size(self):
        return len(self.dict)

    def token_to_id(self, token):
        return self.dict.get(token, self.dict.get(self.unk_token))

    def id_to_token(self, key):
        return self.inverse_dict.get(key, self.unk_token)

    def encode(self, voice):
        notes = voice.replace('^', ' ^').split()
        return [self.token_to_id(n) for n in notes]

    def decode(self, ids):
        return ' '.join(self.id_to_token(i) for i in ids)


class SopranoTokenizer(VocabTokenizer):
    def __init__(self, dataset):
        super().__init__()
        notes = set()
        for voice in dataset:
            soprano = voice[0]
            soprano_notes = soprano.replace('^', ' ^').split()
            notes.update(soprano_notes)

        notes = sorted(notes)
        soprano_tokens = notes + ['R', '_', '^']
        tokens = self.special_tokens + soprano_tokens
        self.dict = {token: i for i, token in enumerate(tokens)}
        self.inverse_dict = {i: token for i, token in enumerate(tokens)}

    def encode(self, voice):
        notes = voice.replace('^', ' ^').split()
        return [self.token_to_id(note) for note in notes]

    def decode(self, voice, skip_special_tokens=False):
        tokens = [self.id_to_token(value) for value in voice]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        result = ' '.join(tokens).replace(' ^', '^')
        return result


class LowersTokenizer(VocabTokenizer):
    def __init__(self, dataset, num_lower_voices):
        super().__init__()
        self.num_lower_voices = num_lower_voices
        notes = set()
        for voice in dataset:
            lowers = voice[1]
            lowers = lowers.replace(';', ' ;').split()
            notes.update(lowers)

        notes = sorted(notes)
        lowers_tokens = notes + ['R', '_']
        tokens = self.special_tokens + lowers_tokens
        self.dict = {token: i for i, token in enumerate(tokens)}
        self.inverse_dict = {i: token for i, token in enumerate(tokens)}

    def encode(self, voice):
        if self.num_lower_voices > 1:
            notes = voice.replace(';', ' ;').split()
        else:
            notes = voice.replace(';', '').split()
        return [self.token_to_id(note) for note in notes]

    def decode(self, voice, skip_special_tokens=False):
        tokens = [self.id_to_token(value) for value in voice]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        result = ''
        for idx, token in enumerate(tokens):
            result += token
            if self.num_lower_voices > 0 and (idx + 1) % self.num_lower_voices == 0:
                result += '; '
            else:
                result += ' '
        return result


class BachChorales:
    """Erzeugt (und cached) den Trainingsdatensatz aus dem music21-Korpus."""

    def __init__(self, output_dir='bach-chorales', num_lower_voices=1, cut_off_at_fermata=True, max_chorales=0):
        file = output_dir + '/' + 'data-' + str(num_lower_voices) + '-voices-' + str(cut_off_at_fermata)
        if num_lower_voices > 1:
            file += '-satb-ts-key'  # Format v2.2: Sopran in der Zielsequenz + Taktart- und Tonart-Token in der Quelle
        path = Path(file)
        if path.is_file():
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    self.raw_dataset = json.load(f)
                    print('loaded data')
            except Exception:
                os.makedirs(output_dir, exist_ok=True)
                self.raw_dataset = self.compute_raw_dataset(num_lower_voices, cut_off_at_fermata, max_chorales)
        else:
            os.makedirs(output_dir, exist_ok=True)
            self.raw_dataset = self.compute_raw_dataset(num_lower_voices, cut_off_at_fermata, max_chorales)

        self.tokenizer1 = SopranoTokenizer(self.raw_dataset)
        self.tokenizer2 = LowersTokenizer(self.raw_dataset, num_lower_voices)
        print(f'source vocab size: {self.tokenizer1.get_vocab_size()}')
        print(f'target vocab size: {self.tokenizer2.get_vocab_size()}')

        self.max_seq_len = self.compute_max_len(cut_off_at_fermata)
        print(f'max length = {self.max_seq_len}')

        with open(file, 'w', encoding='utf-8') as f:
            json.dump(self.raw_dataset, f, ensure_ascii=False, indent=2)

    def __len__(self):
        return len(self.raw_dataset)

    def compute_raw_dataset(self, num_lower_voices, cut_off_at_fermata, max_chorales):
        chorales = corpus.chorales.Iterator()
        print('iterating chorales')

        t = time.process_time()
        data = []
        i = 0
        for chorale in chorales:
            if len(chorale.parts) != 4:
                continue

            if (max_chorales > 0) and (i >= max_chorales):
                break
            if (i % 50 == 0) and (i != 0):
                t = time.process_time() - t
                print(f"{i} ({t:.2f} sec)\n", end='')
            else:
                print('.', end='')
            i += 1

            # Taktart aus dem Original lesen (Transposition ändert sie nicht);
            # bei mehreren Taktwechseln wird die erste verwendet
            ts_list = chorale.parts[0].recurse().getElementsByClass(meter.TimeSignature)
            time_signature = ts_list[0].ratioString if ts_list else '4/4'

            # Tonart einmal am Original bestimmen (Krumhansl-Analyse) und je
            # Transposition mitverschieben — Token z.B. "G:major" / "F#:minor"
            try:
                orig_key = chorale.analyze('key')
            except Exception:
                orig_key = None

            for interval in range(-6, 6):
                try:
                    transposed_chorale = chorale.transpose(interval)
                except Exception:
                    continue
                key_token = None
                if orig_key is not None:
                    k = orig_key.transpose(interval)
                    key_token = f"{k.tonic.name}:{k.mode}"
                soprano = transposed_chorale.parts[0].flatten().notesAndRests
                lower_voices_parts = [part.flatten().notesAndRests for part in transposed_chorale.parts[1:num_lower_voices+1]]
                try:
                    data.append(to_string(soprano, lower_voices_parts, cut_off_at_fermata,
                                          time_signature, key_token))
                except ValueError:
                    print('!', end = '')
                    continue

        print(f"({time.process_time() - t:.2f} sec)")
        print(f"soprano data:     {sum([len(chorale[0]) for chorale in data])} characters")
        print(f"lower voice data: {sum([len(chorale[1]) for chorale in data])} characters")

        return data

    def compute_max_len(self, cut_off_at_fermata=True):
        if not self.raw_dataset:
            return 0
        seq_lens_source = [len(self.tokenizer1.encode(d[0])) for d in self.raw_dataset]
        seq_lens_target = [len(self.tokenizer2.encode(d[1])) for d in self.raw_dataset]

        print(f'max_seqlen_source before filtering: {len(seq_lens_source)}')
        print(f'max_seqlen_target before filtering: {len(seq_lens_target)}')

        avg_source_length = sum(seq_lens_source) / len(seq_lens_source)
        avg_target_length = sum(seq_lens_target) / len(seq_lens_target)

        if cut_off_at_fermata:
            max_len_source = min(2.5 * avg_source_length, 500)
            max_len_target = min(2.5 * avg_target_length, 500)
        else:
            max_len_source = 2.5 * avg_source_length
            max_len_target = 2.5 * avg_target_length

        filtered_indices = [i for i, (s_len, t_len) in enumerate(zip(seq_lens_source, seq_lens_target))
                            if (s_len <= max_len_source) and (t_len <= max_len_target)]

        self.raw_dataset = [self.raw_dataset[i] for i in filtered_indices]
        print(f"{len(self.raw_dataset)} chorales after filtering")

        if not filtered_indices:
            return 0

        max_seq_len_source = max(seq_lens_source[i] for i in filtered_indices)
        max_seq_len_target = max(seq_lens_target[i] for i in filtered_indices)
        print(f'max_seqlen_source after filtering: {max_seq_len_source}')
        print(f'max_seqlen_target after filtering: {max_seq_len_target}')

        return max(max_seq_len_source, max_seq_len_target) + 5

def main():
    # Data
    #dataset = BachChorales(num_lower_voices = 3, cut_off_at_fermata = True)
    #dataset = BachChorales(num_lower_voices = 1, cut_off_at_fermata = False)
    dataset = BachChorales(num_lower_voices = 3, cut_off_at_fermata = False, max_chorales=0)

if __name__ == "__main__":
    main()