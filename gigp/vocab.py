from typing import Dict, List
from .utils import PAD, BOS, EOS, UNK

class Vocab:
    def __init__(self, min_freq=1, specials=(PAD, BOS, EOS, UNK)):
        self.min_freq = min_freq
        self.specials = list(specials)
        self.freq: Dict[str, int] = {}
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []

    def add_sentence(self, tokens: List[str]):
        for t in tokens:
            self.freq[t] = self.freq.get(t, 0) + 1

    def build(self):
        self.itos = []
        self.stoi = {}
        for sp in self.specials:
            self.stoi[sp] = len(self.itos)
            self.itos.append(sp)
        for tok, f in sorted(self.freq.items(), key=lambda x: (-x[1], x[0])):
            if f >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        out = []
        for i in ids:
            i = int(i)
            out.append(self.itos[i] if 0 <= i < len(self.itos) else UNK)
        return out

    @property
    def pad_id(self): return self.stoi[PAD]
    @property
    def bos_id(self): return self.stoi[BOS]
    @property
    def eos_id(self): return self.stoi[EOS]
    @property
    def unk_id(self): return self.stoi[UNK]

    def __len__(self): return len(self.itos)
