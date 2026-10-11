#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Train the BPE tokenizer and the transformer on the corpus (float32).

    python python/train.py                  # models/tokenizer.json, models/model.npz
    python python/train.py --n-layer 3      # another geometry (see chatcbm/config.py)

About 5 minutes on a CPU for the default 4-layer model and 10,000 steps.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from chatcbm import model
from chatcbm.config import Config
from chatcbm.generate import reply
from chatcbm.tokenizer import Tokenizer


class Adam:
    def __init__(self, params, betas=(0.9, 0.99), eps=1e-8):
        self.b1, self.b2, self.eps, self.t = *betas, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, g in grads.items():
            m, v = self.m[k], self.v[k]
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            params[k] -= lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def clip_grads(grads, max_norm):
    norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if norm > max_norm:
        for g in grads.values():
            g *= max_norm / norm
    return norm


def lr_at(step, total, lr, warmup=100):
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def encode_lines(tok, lines):
    stream, starts = [], []
    for line in lines:
        starts.append(len(stream))
        stream.extend(tok.encode(line))
    return np.array(stream, dtype=np.int64), np.array(starts)


def get_batch(stream, starts, batch, T, rng):
    # half of the windows start at a turn (as in a chat), half at random positions
    limit = len(stream) - T - 1
    s = np.concatenate([rng.choice(starts[starts < limit], batch // 2),
                        rng.integers(0, limit, batch - batch // 2)])
    pos = s[:, None] + np.arange(T)
    return stream[pos], stream[pos + 1]


def evaluate(p, cfg, stream, starts, rng, batches=10, batch=64):
    losses = []
    for _ in range(batches):
        x, y = get_batch(stream, starts, batch, cfg.context, rng)
        logits, _ = model.forward(p, cfg, x)
        lg = logits.reshape(-1, logits.shape[-1]).astype(np.float64)
        lg -= lg.max(1, keepdims=True)
        logp = lg[np.arange(len(lg)), y.reshape(-1)] - np.log(np.exp(lg).sum(1))
        losses.append(-logp.mean())
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--out", default="models")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-layer", type=int, default=Config.n_layer)
    ap.add_argument("--context", type=int, default=Config.context)
    ap.add_argument("--d-model", type=int, default=Config.d_model)
    ap.add_argument("--d-ff", type=int, default=Config.d_ff)
    ap.add_argument("--n-head", type=int, default=Config.n_head)
    ap.add_argument("--vocab", type=int, default=Config.vocab_size)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    lines = Path(args.corpus).read_text(encoding="utf-8").splitlines()

    cfg = Config(vocab_size=args.vocab, context=args.context, d_model=args.d_model,
                 n_layer=args.n_layer, n_head=args.n_head, d_ff=args.d_ff)
    t0 = time.time()
    tok = Tokenizer.train("\n".join(lines), cfg.vocab_size)
    tok.save(out / "tokenizer.json")
    print(f"tokenizer: {len(tok)} tokens, {len(tok.merges)} merges ({time.time() - t0:.1f}s)")

    n_val = max(1, len(lines) // 20)
    train_s, train_st = encode_lines(tok, lines[n_val:])
    val_s, val_st = encode_lines(tok, lines[:n_val])
    print(f"dataset: {len(train_s)} training tokens, {len(val_s)} validation tokens, "
          f"{sum(len(line) for line in lines) / (len(train_s) + len(val_s)):.2f} characters/token")

    p = model.init_params(cfg, args.seed)
    print(f"model: {model.count_params(p)} parameters, "
          f"{model.count_params(p) * 4 / 1024:.0f} KB in float32")

    opt = Adam(p)
    t0, avg = time.time(), None
    for step in range(args.steps):
        x, y = get_batch(train_s, train_st, args.batch, cfg.context, rng)
        loss, grads = model.loss_and_grads(p, cfg, x, y)
        clip_grads(grads, 1.0)
        opt.step(p, grads, lr_at(step, args.steps, args.lr))
        avg = loss if avg is None else 0.98 * avg + 0.02 * loss
        if step % 250 == 0 or step == args.steps - 1:
            val = evaluate(p, cfg, val_s, val_st, rng)
            print(f"step {step:5d}  loss {avg:.3f}  val {val:.3f}  ({time.time() - t0:.0f}s)")

    model.save(out / "model.npz", p, cfg)
    print(f"saved in {out}/")

    for q in ["hi", "what color is the frog?", "what does the mouse eat?", "tell me a story"]:
        print(f"> {q}\n  {''.join(reply(p, cfg, tok, q, rng, temperature=0.5)).strip()}")


if __name__ == "__main__":
    main()

#-----------------------------------------------------------------------------
