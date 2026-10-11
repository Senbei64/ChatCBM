#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Tests of the float model: hand-written backward pass and tokenizer."""
import unittest

import numpy as np

from chatcbm import model
from chatcbm.config import Config
from chatcbm.tokenizer import Tokenizer


class GradientCheck(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        cfg = Config(vocab_size=11, context=6, d_model=8, n_layer=2, n_head=2, d_ff=8)
        p = {k: v.astype(np.float64) for k, v in model.init_params(cfg, seed=1).items()}
        rng = np.random.default_rng(0)
        for v in p.values():  # non-trivial LayerNorm and biases
            v += rng.standard_normal(v.shape) * 0.1
        idx = rng.integers(0, cfg.vocab_size, (2, 5))
        targets = rng.integers(0, cfg.vocab_size, (2, 5))
        targets[0, 1] = -1  # ignored target

        _, grads = model.loss_and_grads(p, cfg, idx, targets)
        eps = 1e-6
        for name, value in p.items():
            for _ in range(3):
                i = tuple(rng.integers(0, s) for s in value.shape)
                old = value[i]
                value[i] = old + eps
                lp, _ = model.loss_and_grads(p, cfg, idx, targets)
                value[i] = old - eps
                lm, _ = model.loss_and_grads(p, cfg, idx, targets)
                value[i] = old
                num = (lp - lm) / (2 * eps)
                self.assertAlmostEqual(grads[name][i], num, delta=1e-6 + 1e-4 * abs(num), msg=name)


class TokenizerRoundTrip(unittest.TestCase):
    def test_encode_decode(self):
        text = "<user> hi, how are you?<bot> the cat is black. once a dog!<eos>"
        tok = Tokenizer.train(text * 5, 60)
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids, skip_special=False), text)


if __name__ == "__main__":
    unittest.main()

#-----------------------------------------------------------------------------
