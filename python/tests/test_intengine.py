#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Tests of the all-integer engine: close to the float model, integers only."""
import unittest

import numpy as np

from chatcbm import intengine, model
from chatcbm.config import Config


class IntegerEngine(unittest.TestCase):
    def test_close_to_float_and_integer_only(self):
        cfg = Config(n_layer=2)
        p = model.init_params(cfg, seed=8)
        rng = np.random.default_rng(8)
        for k in p:  # weights of a "trained" model: non-trivial LayerNorm and biases
            if k.endswith((".b", ".bq", ".bk", ".bv", ".bo", ".b1", ".b2")):
                p[k] = p[k] + 0.1 * rng.standard_normal(p[k].shape).astype(np.float32)
        p["wte"] *= 4  # wider logits
        ids = rng.integers(0, cfg.vocab_size, (8, 16))
        m = intengine.IntModel(intengine.compile(p, cfg, ids))
        z = m.run(ids[:2])
        self.assertEqual(z.dtype.kind, "i")
        ref, _ = model.forward(p, cfg, ids[:2])
        corr = np.corrcoef(m.logits(z).reshape(-1), ref.reshape(-1))[0, 1]
        self.assertGreater(corr, 0.95)

    def test_integer_primitives(self):
        t = intengine.Tables(intengine.make_tables(8, 8, 5, 4))
        v = np.array([1, 2, 3, 100, 1000, 32767])
        np.testing.assert_allclose(t.ilog16(v), 16 * np.log2(v), atol=0.6)
        E = np.arange(0, 200)
        np.testing.assert_allclose(t.pow2_16(E), 2.0 ** (E / 16), rtol=0.01, atol=1)


if __name__ == "__main__":
    unittest.main()

#-----------------------------------------------------------------------------
