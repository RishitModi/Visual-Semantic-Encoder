import json
import re
from collections import Counter


class Vocabulary:
    """Token-to-index and index-to-token mapping.

    Special tokens (hard-coded to match the trained checkpoint):
        <PAD> = 0, <SOS> = 1, <EOS> = 2, <UNK> = 3
    """

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self, freqthreshold=2):
        self.freq_threshold = freqthreshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def buildvocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized]

    # ------------------------------------------------------------------
    # Serialisation helpers so inference doesn't need the raw annotations
    # ------------------------------------------------------------------

    def save(self, path):
        """Save vocabulary to a JSON file."""
        data = {
            "freq_threshold": self.freq_threshold,
            "itos": {str(k): v for k, v in self.itos.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        """Load vocabulary from a JSON file previously written by ``save``."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(freqthreshold=data["freq_threshold"])
        vocab.itos = {int(k): v for k, v in data["itos"].items()}
        vocab.stoi = {v: k for k, v in vocab.itos.items()}
        return vocab
