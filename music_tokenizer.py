"""Serialisierung/Deserialisierung: Token-Strings <-> IDs.

MusicTokenizer ist der einzige Übergang zwischen der textuellen
Datendarstellung (siehe bach_chorales.to_string) und den Tensor-IDs des
Modells — mit gerade genug HuggingFace-API, dass DataCollatorForSeq2Seq und
der Trainer damit arbeiten können.
"""

import json
from pathlib import Path


class MusicTokenizer:
    """
    Funktional identisch mit den alten Tokenizern (VoiceTokenizer, etc.)
    MIT vollständiger HuggingFace-Kompatibilität für DataCollator
    """

    def __init__(self, vocab_file=None):
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.dict = {t: i for i, t in enumerate(self.special_tokens)}
        self.inverse_dict = {i: t for i, t in enumerate(self.special_tokens)}

        # HuggingFace Kompatibilität
        self.padding_side = "right"
        self.truncation_side = "right"
        self.model_input_names = ["input_ids", "attention_mask"]

        if vocab_file and Path(vocab_file).exists():
            self.load_vocab(vocab_file)

    def load_vocab(self, vocab_file):
        """Lädt Vokabular aus JSON-Datei"""
        with open(vocab_file, 'r') as f:
            data = json.load(f)
        self.dict = {k: int(v) for k, v in data["dict"].items()}
        self.inverse_dict = {int(k): v for k, v in data["inverse_dict"].items()}

    def save_vocab(self, vocab_file):
        """Speichert Vokabular in JSON-Datei"""
        data = {
            "dict": {k: v for k, v in self.dict.items()},
            "inverse_dict": {k: v for k, v in self.inverse_dict.items()}
        }
        with open(vocab_file, 'w') as f:
            json.dump(data, f, indent=2)

    def save_pretrained(self, save_directory):
        """Speichert Tokenizer im HuggingFace-Format (wird vom Trainer aufgerufen)"""
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        vocab_file = Path(save_directory) / "vocab.json"
        self.save_vocab(str(vocab_file))

    def build_vocab(self, dataset):
        """Erstellt Vokabular aus Trainingsdaten"""
        notes = {'_'}
        for src, tgt in dataset:
            notes.update(src.split())
            notes.update(tgt.split())

        notes = sorted(notes)
        tokens = self.special_tokens + notes
        self.dict = {token: i for i, token in enumerate(tokens)}
        self.inverse_dict = {i: token for i, token in enumerate(tokens)}

    def encode(self, text):
        """Text -> IDs, gerahmt von [CLS] ... [SEP]"""
        notes = text.split()
        return [self.cls_token_id] + [self.dict.get(note, self.dict.get('[UNK]')) for note in notes] + [self.sep_token_id]

    def decode(self, ids, skip_special_tokens=False):
        """IDs -> Text"""
        tokens = [self.inverse_dict.get(int(i), '[UNK]') for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        else:
            # Entferne BOS/EOS manuell für saubere Outputs
            if tokens and tokens[0] == self.cls_token:
                tokens = tokens[1:]
            if tokens and tokens[-1] == self.sep_token:
                tokens = tokens[:-1]
        return ' '.join(tokens)

    def get_vocab_size(self):
        return len(self.dict)

    def __len__(self):
        return self.get_vocab_size()

    # ===== HuggingFace PROPERTIES =====

    @property
    def pad_token_id(self):
        return self.dict.get("[PAD]", 0)

    @property
    def eos_token_id(self):
        return self.dict.get("[SEP]", 3)

    @property
    def bos_token_id(self):
        return self.dict.get("[CLS]", 2)

    @property
    def cls_token_id(self):
        return self.dict.get("[CLS]", 2)

    @property
    def sep_token_id(self):
        return self.dict.get("[SEP]", 3)

    @property
    def unk_token_id(self):
        return self.dict.get("[UNK]", 1)

    @property
    def pad_token(self):
        return "[PAD]"

    @property
    def eos_token(self):
        return "[SEP]"

    @property
    def bos_token(self):
        return "[CLS]"

    @property
    def cls_token(self):
        return "[CLS]"

    @property
    def sep_token(self):
        return "[SEP]"

    @property
    def unk_token(self):
        return "[UNK]"

    # ===== ERFORDERLICHE METHODEN FÜR DataCollatorForSeq2Seq =====

    def pad(self,
            encoded_inputs,
            padding='longest',
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors='pt',
            return_attention_mask=True):
        """
        HuggingFace-kompatible Padding-Methode für DataCollator.
        """
        import torch

        # ===== INPUT NORMALISIEREN =====
        if isinstance(encoded_inputs, dict) and isinstance(encoded_inputs.get('input_ids', None), list):
            # Dict mit Listen (batch-form)
            batch_input_ids = encoded_inputs.get('input_ids', [])
        elif isinstance(encoded_inputs, list) and len(encoded_inputs) > 0 and isinstance(encoded_inputs[0], dict):
            # Liste von dicts (einzelne Beispiele)
            batch_input_ids = [ex.get('input_ids', []) for ex in encoded_inputs]
        else:
            raise ValueError(f"Unerwartetes Format für encoded_inputs: {type(encoded_inputs)}")

        # ===== MAX_LENGTH BESTIMMEN =====
        if max_length is None:
            if batch_input_ids:
                max_length = max(len(ids) for ids in batch_input_ids)
            else:
                max_length = 512

        # ===== PADDING DURCHFÜHREN =====
        padded_input_ids = []
        padded_attention_mask = []

        for input_ids in batch_input_ids:
            input_ids = list(input_ids) if not isinstance(input_ids, list) else input_ids

            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                padded_ids = input_ids + [self.pad_token_id] * pad_length
                mask = [1] * len(input_ids) + [0] * pad_length
            else:
                padded_ids = input_ids[:max_length]
                mask = [1] * min(len(input_ids), max_length)

            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(mask)

        # ===== RÜCKGABE =====
        result = {}
        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor(padded_input_ids, dtype=torch.long)
            if return_attention_mask:
                result['attention_mask'] = torch.tensor(padded_attention_mask, dtype=torch.long)
        else:
            result['input_ids'] = padded_input_ids
            if return_attention_mask:
                result['attention_mask'] = padded_attention_mask

        return result

    def convert_tokens_to_ids(self, tokens):
        """Konvertiert Tokens zu IDs"""
        if isinstance(tokens, str):
            return self.dict.get(tokens, self.dict.get('[UNK]'))
        return [self.dict.get(t, self.dict.get('[UNK]')) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        """Konvertiert IDs zu Tokens"""
        if isinstance(ids, int):
            return self.inverse_dict.get(ids, '[UNK]')
        return [self.inverse_dict.get(i, '[UNK]') for i in ids]

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
        """
        HuggingFace-kompatible __call__ Methode
        """
        import torch

        if isinstance(text, str):
            text = [text]

        input_ids = [self.encode(t) for t in text]

        if truncation and max_length:
            input_ids = [ids[:max_length] for ids in input_ids]

        if padding:
            if max_length is None:
                max_length = max(len(ids) for ids in input_ids) if input_ids else 512

            padded_ids = []
            attention_masks = []
            for ids in input_ids:
                padded = ids + [self.pad_token_id] * (max_length - len(ids))
                mask = [1] * len(ids) + [0] * (max_length - len(ids))
                padded_ids.append(padded)
                attention_masks.append(mask)

            input_ids = padded_ids
            attention_mask = attention_masks
        else:
            attention_mask = [[1] * len(ids) for ids in input_ids]

        result = {'input_ids': input_ids, 'attention_mask': attention_mask}

        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor(result['input_ids'], dtype=torch.long)
            result['attention_mask'] = torch.tensor(result['attention_mask'], dtype=torch.long)

        return result
