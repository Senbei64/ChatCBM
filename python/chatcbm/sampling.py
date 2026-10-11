#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""From logits to the next token: softmax, temperature, top-p (float reference)."""
import numpy as np


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sample(logits, rng, temperature=0.8, top_p=0.9, banned=()):
    logits = logits.astype(np.float64)
    if banned:
        logits[list(banned)] = -np.inf
    if temperature <= 0:
        return int(np.argmax(logits))  # greedy: always the most likely token

    probs = softmax(logits / temperature)
    if top_p < 1.0:
        # keep only the most likely tokens whose sum reaches top_p
        order = np.argsort(-probs)
        cut = int(np.searchsorted(np.cumsum(probs[order]), top_p)) + 1
        kept = np.zeros_like(probs)
        kept[order[:cut]] = probs[order[:cut]]
        probs = kept / kept.sum()
    return int(rng.choice(len(probs), p=probs))

#-----------------------------------------------------------------------------
