#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Quantization-aware training (QAT), starting from the float model.

    python python/qat.py                # 3000 steps, writes models/model_qat.npz

The forward pass uses the weights already rounded to nf4 (as the integer engine will see
them); the gradient computed with respect to the quantized weights updates the float
"master" weights (straight-through estimator). The network learns to work with its 4-bit
levels.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from chatcbm import intengine, model
from chatcbm.tokenizer import Tokenizer
from train import Adam, clip_grads, encode_lines, evaluate, get_batch, lr_at


def fake_quant(p):
    """Copy of the parameters with matrices and embeddings rounded to the nf4 levels."""
    q = dict(p)
    for name in p:
        if name.endswith(intengine.LIN) or name in intengine.EMBEDDINGS:
            codes, k = intengine.quantize_nf4(p[name], 1 if name in intengine.EMBEDDINGS else 0)
            q[name] = (intengine.NF4[codes] * 2.0 ** (k / 16)).astype(np.float32)
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/model.npz")
    ap.add_argument("--out", default="models/model_qat.npz")
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    p, cfg = model.load(args.model)
    tok = Tokenizer.load(Path(args.model).parent / "tokenizer.json")
    lines = Path(args.corpus).read_text(encoding="utf-8").splitlines()
    n_val = max(1, len(lines) // 20)
    train_s, train_st = encode_lines(tok, lines[n_val:])
    val_s, val_st = encode_lines(tok, lines[:n_val])
    rng = np.random.default_rng(args.seed)

    print(f"start: val float {evaluate(p, cfg, val_s, val_st, rng):.3f}, "
          f"val nf4 {evaluate(fake_quant(p), cfg, val_s, val_st, rng):.3f}")
    opt = Adam(p)
    t0, avg = time.time(), None
    for step in range(args.steps):
        x, y = get_batch(train_s, train_st, args.batch, cfg.context, rng)
        loss, grads = model.loss_and_grads(fake_quant(p), cfg, x, y)  # STE: gradient on nf4 weights
        clip_grads(grads, 1.0)
        opt.step(p, grads, lr_at(step, args.steps, args.lr))
        avg = loss if avg is None else 0.98 * avg + 0.02 * loss
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  loss nf4 {avg:.3f}  val nf4 {evaluate(fake_quant(p), cfg, val_s, val_st, rng):.3f}"
                  f"  val float {evaluate(p, cfg, val_s, val_st, rng):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    model.save(args.out, p, cfg)
    print(f"saved in {args.out}")


if __name__ == "__main__":
    main()

#-----------------------------------------------------------------------------
