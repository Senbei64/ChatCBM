#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Measure the float model and the integer engine on the corpus world.

    python python/evaluate.py           # a few minutes

For both engines: validation loss, exact greedy answers to every question with a single
correct answer (by category, see corpus.facts()) and sampled stories that must be one of
the correct ones for the request (about one animal or about two).
"""
import argparse
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

import corpus
from chatcbm import intengine, model
from chatcbm.generate import chat_prompt, reply
from chatcbm.tokenizer import BOT, EOS, PAD, UNK, USER, Tokenizer
from train import encode_lines

STORY_SEEDS = 3


def val_loss(logits_fn, stream, T, windows=64):
    starts = np.arange(windows) * T
    x = stream[starts[:, None] + np.arange(T)]
    y = stream[starts[:, None] + np.arange(T) + 1]
    lg = logits_fn(x)
    lg = lg.reshape(-1, lg.shape[-1]).astype(np.float64)
    lg -= lg.max(1, keepdims=True)
    return float(-(lg[np.arange(len(lg)), y.reshape(-1)] - np.log(np.exp(lg).sum(1))).mean())


def fact_accuracy(answer):
    """Exact greedy answers by category, over every question of corpus.facts()."""
    ok, tot = Counter(), Counter()
    for category, q, a in corpus.facts():
        tot[category] += 1
        ok[category] += answer(q) == a
    return ok, tot


def valid_stories(a, b=None):
    """The correct stories about animal a (meeting b, if given): animal met x meetings x endings."""
    return {corpus.story_text(a, c, meeting, ending)
            for c in map(corpus.Animal, corpus.ANIMALS) if c.name != a.name and (b is None or c.name == b.name)
            for meeting, ending in product(corpus.meetings(a, c), corpus.endings(a, c))}


def story_accuracy(tell):
    """Sampled stories (temperature 0.6), STORY_SEEDS seeds per animal, about one animal and
    about a pair. A story counts only if it is one of the correct ones for the request."""
    animals = [corpus.Animal(r) for r in corpus.ANIMALS]
    ok, tot = Counter(), Counter()
    for i, a in enumerate(animals):
        for s in range(STORY_SEEDS):
            b = animals[(i + 1 + s) % len(animals)]
            for category, q, valid in (("one animal", corpus.story_request(a), valid_stories(a)),
                                       ("two animals", corpus.story_request(a, b), valid_stories(a, b))):
                tot[category] += 1
                ok[category] += tell(q, 1 + 100 * i + s) in valid
    return ok, tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="models", help="directory with tokenizer.json, model.npz, model_int.npz")
    ap.add_argument("--corpus", default="data/corpus.txt")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    ref, cfg = model.load(f"{args.dir}/model.npz")
    m = intengine.IntModel(intengine.load(f"{args.dir}/model_int.npz"))
    tok = Tokenizer.load(f"{args.dir}/tokenizer.json")
    lines = Path(args.corpus).read_text(encoding="utf-8").splitlines()
    n_val = max(1, len(lines) // 20)
    val = encode_lines(tok, lines[:n_val])[0]
    stop, banned = (tok.id(EOS), tok.id(USER)), (tok.id(PAD), tok.id(UNK), tok.id(BOT))

    def float_answer(q, seed=None, n=60):
        g = np.random.default_rng(seed)
        kw = dict(temperature=0) if seed is None else dict(temperature=0.6, top_p=0.9)
        return "".join(reply(ref, cfg, tok, q, g, max_new_tokens=n, **kw)).strip()

    def int_answer(q, seed=None, n=60):
        sampler = None if seed is None else intengine.IntSampler(0.6, 0.9, seed=seed)
        return tok.decode(list(m.generate(chat_prompt(tok, q), n, stop, banned, sampler))).strip()

    runs = [("float32", lambda x: model.forward(ref, cfg, x)[0], float_answer),
            ("integer", lambda x: m.logits(m.run(x)), int_answer)]
    results = {}
    for label, logits_fn, answer in runs:
        loss = val_loss(logits_fn, val, cfg.context)
        facts = fact_accuracy(answer)
        stories = story_accuracy(lambda q, seed: answer(q, seed, 90))
        results[label] = loss, facts, stories
    print(f"\n{'':26s}" + "".join(f"{label:>14s}" for label in results))
    print(f"{'validation loss':26s}" + "".join(f"{r[0]:14.3f}" for r in results.values()))
    for i, title in ((1, "answers"), (2, "stories")):
        for category in next(iter(results.values()))[i][1]:
            print(f"{title + ': ' + category:26s}" + "".join(f"{r[i][0][category]:9d}/{r[i][1][category]:<4d}"
                                                             for r in results.values()))
        print(f"{title + ': total':26s}" + "".join(
            f"{sum(r[i][0].values()):9d}/{sum(r[i][1].values()):<4d}" for r in results.values()))


if __name__ == "__main__":
    main()

#-----------------------------------------------------------------------------
