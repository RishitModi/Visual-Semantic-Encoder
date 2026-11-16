import re
from collections import Counter


class Vocabulary:
    def __init__(self, freqthreshold=2):
        self.freq_threshold = freqthreshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return re.findall(r"\b\w+\b", text.lower())

    def buildvocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            frequencies.update(self.tokenizer(sentence))
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in self.tokenizer(text)]

    def save(self, path):
        """Write the vocabulary as one 'index:word' line per entry."""
        with open(path, "w", encoding="utf-8") as f:
            for idx in sorted(self.itos):
                f.write(f"{idx}:{self.itos[idx]}\n")

    @classmethod
    def from_itos(cls, itos):
        vocab = cls(freqthreshold=1)
        vocab.itos = {int(k): w for k, w in itos.items()}
        vocab.stoi = {w: k for k, w in vocab.itos.items()}
        return vocab

    @classmethod
    def load(cls, path):
        """Read a vocabulary written by save()."""
        with open(path, "r", encoding="utf-8") as f:
            entries = [line.rstrip("\n").split(":", 1) for line in f if line.strip()]
        return cls.from_itos({int(k): w for k, w in entries})
