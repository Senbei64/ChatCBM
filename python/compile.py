#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Compile the QAT model into the all-integer image (chatcbm/intengine.py).

    python python/compile.py            # models/model_qat.npz -> models/model_int.npz

Calibrates the scales on training windows, quantizes (nf4, log8, int16 biases), builds the
tables and saves the image. evaluate.py measures it against the float model.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from chatcbm import intengine, model
from chatcbm.tokenizer import Tokenizer
from train import encode_lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="models", help="directory with tokenizer.json")
    ap.add_argument("--model", help="model to compile (default: DIR/model_qat.npz)")
    ap.add_argument("--out", help="default: DIR/model_int.npz")
    ap.add_argument("--corpus", default="data/corpus.txt")
    args = ap.parse_args()
    args.model = args.model or f"{args.dir}/model_qat.npz"
    args.out = args.out or f"{args.dir}/model_int.npz"
    sys.stdout.reconfigure(encoding="utf-8")

    p, cfg = model.load(args.model)
    tok = Tokenizer.load(f"{args.dir}/tokenizer.json")
    lines = Path(args.corpus).read_text(encoding="utf-8").splitlines()
    n_val = max(1, len(lines) // 20)  # the same 5% validation split as train.py
    train = encode_lines(tok, lines[n_val:])[0]
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(train) - cfg.context, 512)
    calib = train[starts[:, None] + np.arange(cfg.context)]

    img = intengine.compile(p, cfg, calib)
    intengine.save(args.out, img)
    m = intengine.IntModel(intengine.load(args.out))
    sizes = intengine.image_bytes(m.img)
    print(f"{args.out}: {sum(sizes.values())} bytes on an 8-bit machine")
    for k, v in sizes.items():
        print(f"  {k:12s}{v:7d}")
    print(f"residual with {m.R} fractional bits")


if __name__ == "__main__":
    main()

#-----------------------------------------------------------------------------
