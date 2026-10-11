#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Autoregressive loop: prompt -> token -> transformer -> token -> ... until <eos>."""
import numpy as np

from . import model
from .sampling import sample
from .tokenizer import BOT, EOS, PAD, UNK, USER


def generate(p, cfg, ids, rng, max_new_tokens=60, temperature=0.8, top_p=0.9,
             stop_ids=(), banned=()):
    """Generator that yields one token at a time (streaming), float model."""
    ids = list(ids)
    for _ in range(max_new_tokens):
        window = np.array(ids[-cfg.context:])[None]  # last `context` tokens
        logits, _ = model.forward(p, cfg, window)
        nxt = sample(logits[0, -1], rng, temperature, top_p, banned)
        if nxt in stop_ids:
            return
        ids.append(nxt)
        yield nxt


def chat_prompt(tok, user_text):
    """A turn is <user> question<bot> answer<eos>: the prompt ends with <bot>."""
    return [tok.id(USER)] + tok.encode(" " + user_text.strip()) + [tok.id(BOT)]


def reply(p, cfg, tok, user_text, rng, **kw):
    """Yield the text pieces of the answer, one per token."""
    ids = chat_prompt(tok, user_text)
    stop = (tok.id(EOS), tok.id(USER))
    banned = (tok.id(PAD), tok.id(UNK), tok.id(BOT))
    for t in generate(p, cfg, ids, rng, stop_ids=stop, banned=banned, **kw):
        yield tok.decode([t])

#-----------------------------------------------------------------------------
