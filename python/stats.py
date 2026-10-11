#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Count the parameters and the operations of a model geometry.

    python python/stats.py                 # default configuration
    python python/stats.py --n-layer 3     # try other geometries
"""
import argparse

from chatcbm.config import Config


def param_table(c):
    C, F, V, T, L = c.d_model, c.d_ff, c.vocab_size, c.context, c.n_layer
    # (name, multiplicative weights, biases, LayerNorm parameters, lookup tables)
    return [
        ("token embedding (shared with the output)", 0, 0, 0, V * C),
        ("positional embedding", 0, 0, 0, T * C),
        (f"attention Q,K,V,O  x{L}", L * 4 * C * C, L * 4 * C, 0, 0),
        (f"FFN {C}->{F}->{C}  x{L}", L * 2 * C * F, L * (F + C), 0, 0),
        (f"LayerNorm  x{2 * L + 1}", 0, 0, (2 * L + 1) * 2 * C, 0),
        ("output bias", 0, V, 0, 0),
    ]


def products_per_token(c, T):
    """Multiplications to generate one token at position T, with a KV cache."""
    C, F, V, H, L = c.d_model, c.d_ff, c.vocab_size, c.n_head, c.n_layer
    hs = C // H
    return (L * (4 * C * C + 2 * C * F)    # linear layers, one product per weight
            + L * H * T * hs * 2           # Q.K scores and the weighted sum of V
            + V * C)                       # logits (weight tying with the embedding)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("vocab_size", "context", "d_model", "n_layer", "n_head", "d_ff"):
        ap.add_argument("--" + f.replace("_", "-"), type=int, default=getattr(Config, f))
    c = Config(**vars(ap.parse_args()))

    print(f"configuration: {c}\n")
    rows = param_table(c)
    print(f"{'':42s}{'weights':>8s}{'biases':>8s}{'norm':>8s}{'lookup':>8s}")
    for name, *vals in rows:
        print(f"{name:42s}" + "".join(f"{v:8d}" for v in vals))
    tot = [sum(r[i] for r in rows) for i in range(1, 5)]
    print(f"{'total':42s}" + "".join(f"{v:8d}" for v in tot))
    n = sum(tot)
    print(f"\nparameters: {n}  ->  {n * 4 / 1024:.1f} KB float32, {n / 2048:.1f} KB with 4-bit weights")

    print("\nproducts to generate ONE token with a KV cache:")
    for T in (1, 16, c.context):
        print(f"  position {T:3d}: {products_per_token(c, T):>7,d}")


if __name__ == "__main__":
    main()

#-----------------------------------------------------------------------------
