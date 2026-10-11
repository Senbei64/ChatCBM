#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Minimal BPE (Byte Pair Encoding) tokenizer.

Start from the single characters of the corpus and repeatedly merge the most
frequent adjacent pair, until the vocabulary reaches the requested size.
Merges never cross word boundaries: each word (with the space before it) is an
independent "chunk". The 6502 tokenizer (src/tokenizer.s) does the same.
"""
import json
import re
from collections import Counter

PAD, EOS, UNK, USER, BOT = "<pad>", "<eos>", "<unk>", "<user>", "<bot>"
SPECIALS = [PAD, EOS, UNK, USER, BOT]

_SPECIAL_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")
# " word" (with an optional trailing apostrophe), numbers, punctuation, spaces
_CHUNK_RE = re.compile(r" ?[^\W\d_]+'?| ?\d+| ?[^\s\w]+|\s+")


def normalize(text):
    return re.sub(r"\s+", " ", text.lower())


def _chunks(text):
    return _CHUNK_RE.findall(normalize(text))


class Tokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab                      # id -> string
        self.merges = merges                    # list of pairs (id_a, id_b), in order
        self.stoi = {s: i for i, s in enumerate(vocab)}
        self.ranks = {tuple(m): r for r, m in enumerate(merges)}
        self.merged = {tuple(m): self.stoi[vocab[m[0]] + vocab[m[1]]] for m in merges}
        self._cache = {}

    # ---- training -----------------------------------------------------------
    @classmethod
    def train(cls, text, vocab_size):
        counts = Counter()
        for part in _SPECIAL_RE.split(text):
            if part and part not in SPECIALS:
                counts.update(_chunks(part))

        chars = sorted({c for chunk in counts for c in chunk})
        vocab = SPECIALS + chars
        stoi = {s: i for i, s in enumerate(vocab)}
        words = {tuple(stoi[c] for c in chunk): n for chunk, n in counts.items()}
        merges = []

        while len(vocab) < vocab_size:
            pairs = Counter()
            for w, n in words.items():
                for pair in zip(w, w[1:]):
                    pairs[pair] += n
            if not pairs:
                break
            best, n = pairs.most_common(1)[0]
            if n < 2:
                break
            new_id = len(vocab)
            vocab.append(vocab[best[0]] + vocab[best[1]])
            merges.append(best)
            words = {_merge(w, best, new_id): c for w, c in words.items()}

        return cls(vocab, merges)

    # ---- encoding / decoding --------------------------------------------------
    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        unk = self.stoi[UNK]
        ids = [self.stoi.get(c, unk) for c in chunk]
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge(ids, pair, self.merged[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        ids = []
        for part in _SPECIAL_RE.split(text):
            if not part:
                continue
            if part in SPECIALS:
                ids.append(self.stoi[part])
            else:
                for chunk in _chunks(part):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids, skip_special=True):
        return "".join(self.vocab[i] for i in ids
                       if not (skip_special and self.vocab[i] in SPECIALS))

    def id(self, token):
        return self.stoi[token]

    def __len__(self):
        return len(self.vocab)

    # ---- saving ---------------------------------------------------------------
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["vocab"], [tuple(m) for m in d["merges"]])


def _merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out) if isinstance(ids, tuple) else out

#-----------------------------------------------------------------------------
