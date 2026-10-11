#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Export the integer image and the tokenizer for the 6502 engine.

    python python/export.py --out build/gen

Writes:
  model.bin      the model image: packed nf4 weights, scales, biases, LayerNorm, tables
                 (included by src/model.s, must start on a page boundary)
  config.inc     constants and image offsets for the assembler
  config.h       the same constants for the C test programs
  desc.inc       layer descriptors: pointers into the image and the KV cache (src/model.s)
  tables.inc     tables of the primitives (src/prim.s)
  tokenizer.inc  BPE tokenizer: characters, merges, vocabulary strings (src/tokenizer.s)
  sampler.inc    temperature table and top-p threshold (src/sampler.s)

Weight layout (src/matmul.s): packed nf4, two weights per byte. For each matrix W
(in=32, out=32) column j takes 16 bytes: byte k holds row 2k in the low nibble and row
2k+1 in the high nibble, so a matrix is one 512-byte block. Matrices, wte and wpe come
first in the image: every block is page aligned and the kernel only patches the high
bytes of its operands. wte is both the embedding (row = token, 16 bytes) and the output
weights (column = token): same layout, 8 blocks.
"""
import argparse
from pathlib import Path

import numpy as np

from chatcbm import intengine
from chatcbm.tokenizer import BOT, EOS, PAD, UNK, USER, Tokenizer

ZERO = intengine.ZERO
TEMPERATURE, TOP_P = 0.6, 0.9   # as chat.py
MAX_NEW = 80                     # longest answer, in tokens
MATRICES = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "ffn.w1", "ffn.w2")


def pack(codes):
    """(N, 32) nf4 codes -> (N, 16) bytes: element 2k in the low nibble, 2k+1 in the high one."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)


def log8_bytes(codes):
    """nf4 codes -> log8 bytes (sign << 7 | e), zero = 127."""
    e = intengine.NF4_EXP[codes]
    s = intengine.NF4_SIGN[codes].astype(np.int64)
    return np.where(e == ZERO, ZERO, e | (s << 7)).astype(np.uint8)


def prep_bias(q, s, F, t):
    """Bias of a linear layer for lin_post: 2 (ilog16|q| + 16 (F - s) + 1024) + sign, 0 -> $7FFF."""
    q = np.asarray(q, np.int64)
    L = t.ilog16(np.where(q == 0, 1, np.abs(q))) + 16 * (F - s) + 1024
    out = np.where(q == 0, 0x7FFF, (L << 1) | (q < 0))
    assert out.min() >= 0 and out.max() <= 0x7FFF
    return out


class Image:
    """The model image, built field by field: bytes and offsets."""

    def __init__(self):
        self.data, self.offsets = bytearray(), {}

    def add(self, name, arr, dtype):
        arr = np.asarray(arr).astype(dtype).reshape(-1)
        if dtype == np.int16:
            assert arr.min() >= -32768 and arr.max() <= 32767, name
        self.offsets[name] = len(self.data)
        self.data += arr.astype("<" + np.dtype(dtype).str[1:]).tobytes()


def build_image(img, cfg, t):
    b = Image()
    for i in range(cfg.n_layer):                                 # 512-byte blocks, page aligned
        for n in MATRICES:
            b.add(f"W_{i}_{n[-2:]}", pack(img[f"h{i}.{n}.q"].T), np.uint8)   # (out, in): columns
    b.add("WTE", pack(img["wte.q"]), np.uint8)                 # (V, 16): row = token
    b.add("WPE", pack(img["wpe.q"]), np.uint8)                 # (T, 16)
    assert len(b.data) % 256 == 0
    b.add("NF4LOG", log8_bytes(np.arange(16)), np.uint8)        # nf4 code -> log8 byte
    for i in range(cfg.n_layer):
        for n in MATRICES:
            name = f"h{i}.{n}"
            b.add(f"K_{i}_{n[-2:]}", img[name + ".k"], np.int8)
            bias = f"h{i}.{intengine.BIAS[n]}"
            b.add(f"B_{i}_{n[-2:]}", prep_bias(img[bias + ".i"], img[bias + ".s"], img["F"], t), np.uint16)
        for n in ("Kq", "Kk", "Kv", "Zs"):
            b.add(f"{n}_{i}", img[f"h{i}.{n}"], np.int16)
    b.add("K_WTE", img["wte.k"], np.int8)
    b.add("B_OUT", prep_bias(img["b_out.i"], img["b_out.s"], img["F"], t), np.uint16)
    b.add("K_WPE", img["wpe.k"], np.int8)
    for idx, pre in enumerate(layernorms(cfg)):
        neg, e = img[pre + "g.neg"], img[pre + "g.e"]
        b.add(f"LNG_{idx}", np.where(e == ZERO, ZERO, e | (neg << 7)), np.uint8)
        shifted = intengine.shift(img[pre + "b.i"], img["F_ln"] - img[pre + "b.s"])   # pre-shifted
        assert np.abs(shifted).max() <= 32767, f"{pre}b does not fit int16 once shifted"
        b.add(f"LNB_{idx}", shifted, np.int16)
    b.add("T_PROD", img["t.prod"], np.uint8)
    b.add("T_PRODA", img["t.prod_attn"], np.uint8)
    b.add("T_GELU", img["t.gelu"], np.int16)
    for k, v in img.items():                                   # the k scales must fit int8
        if k.endswith(".k") and np.ndim(v):
            assert -128 <= v.min() and v.max() <= 127, k
    return b


def layernorms(cfg):
    return [f"h{i}.ln{j}." for i in range(cfg.n_layer) for j in (1, 2)] + ["lnf."]


def rows(vals, op=".byte"):
    vals = [v if isinstance(v, str) else str(int(v)) for v in vals]
    return "\n".join(f"        {op} " + ",".join(vals[i:i + 16]) for i in range(0, len(vals), 16))


def write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="build/gen")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    img = intengine.load(Path(args.models) / "model_int.npz")
    m = intengine.IntModel(img)
    cfg, t = m.cfg, m.t
    assert (cfg.d_model, cfg.d_ff, cfg.n_head, cfg.context) == (32, 32, 4, 64), "the 6502 engine assumes 32/32/4/64"
    tok = Tokenizer.load(Path(args.models) / "tokenizer.json")

    b = build_image(img, cfg, t)
    (out / "model.bin").write_bytes(bytes(b.data))

    # constants for the assembler (config.inc) and for C (config.h)
    consts = {"IMAGE_SIZE": len(b.data), "N_LAYER": cfg.n_layer, "CONTEXT": cfg.context,
              "VOCAB": cfg.vocab_size, "F_LIN": img["F"], "F_ATT": img["F_attn"], "F_LN": img["F_ln"],
              "G16": 16 * img["G"], "R_RES": img["R"], "Z_L": img["Zl"], "GELU_FRAC": img["gelu_frac"],
              "GELU_RANGE": img["gelu_range"], "MAX_NEW": MAX_NEW,
              "TOK_PAD": tok.id(PAD), "TOK_EOS": tok.id(EOS), "TOK_UNK": tok.id(UNK),
              "TOK_USER": tok.id(USER), "TOK_BOT": tok.id(BOT)}
    consts.update({"OFF_" + k: v for k, v in b.offsets.items()})
    write(out / "config.inc", ["; generated by python/export.py", *[f"{k} = {v}" for k, v in consts.items()]])
    write(out / "config.h", ["/* generated by python/export.py */", *[f"#define {k} {v}" for k, v in consts.items()]])

    # layer descriptors for src/model.s
    W = lambda name: f"_img + {b.offsets[name]}"   # noqa: E731
    lns = layernorms(cfg)
    lines = ["; generated by python/export.py: layer descriptors (src/model.s)", "layer_desc:"]
    for i in range(cfg.n_layer):
        words = []
        for n in MATRICES:
            k = n[-2:]
            words += [W(f"W_{i}_{k}"), W(f"K_{i}_{k}"), W(f"B_{i}_{k}")]
        for j in (2 * i, 2 * i + 1):
            words += [W(f"LNG_{j}"), str(int(img[lns[j] + "g.k"]) & 0xFFFF), W(f"LNB_{j}")]
        words += [W(f"{n}_{i}") for n in ("Kq", "Kk", "Kv", "Zs")]
        words += [f"kcache + {i * 32 * cfg.context}", f"vcache + {i * 32 * cfg.context}"]
        lines.append("        .word " + ", ".join(words))
    n = 2 * cfg.n_layer
    lines.append("final_desc:")
    lines.append("        .word " + ", ".join([W(f"LNG_{n}"), str(int(img["lnf.g.k"]) & 0xFFFF), W(f"LNB_{n}"),
                                             W("WTE"), W("K_WTE"), W("B_OUT"), W("WPE"), W("K_WPE")]))
    write(out / "desc.inc", lines)

    # tables of the primitives: ilog16 mantissa, 2^(e/16) for e = 0..15 and for e = 0..255
    p2 = t.pow2_16(np.arange(256))
    assert p2.max() < 65536
    write(out / "tables.inc", ["; generated by python/export.py: tables of the primitives (src/prim.s)",
                               "logm_t:", rows(img["t.logm"]), "powlo:", rows(img["t.pow"] & 255),
                               "powhi:", rows(img["t.pow"] >> 8), "p2lo:", rows(p2 & 255), "p2hi:", rows(p2 >> 8)])

    # tokenizer: the 6502 applies the merges in rank order, merge r makes token n_base + r
    n_base = len(tok) - len(tok.merges)
    assert all(tok.merged[tuple(mg)] == n_base + r for r, mg in enumerate(tok.merges))
    assert n_base + len(tok.merges) <= 256
    chartab = [tok.stoi.get(chr(c), tok.id(UNK)) for c in range(256)]
    words = [tok.decode([i]).encode("ascii") for i in range(len(tok))]   # specials: empty strings
    offs = np.cumsum([0] + [len(w) + 1 for w in words])[:-1]
    write(out / "tokenizer.inc", [
        "; generated by python/export.py: BPE tokenizer and vocabulary (src/tokenizer.s, src/chat.s)",
        f"TOK_BASE = {n_base}", f"N_MERGES = {len(tok.merges)}",
        "chartab:", rows(chartab),
        "merge_a:", rows([mg[0] for mg in tok.merges]),
        "merge_b:", rows([mg[1] for mg in tok.merges]),
        "voc_lo:", rows([f"voc_str+{o}" for o in offs], ".lobytes"),
        "voc_hi:", rows([f"voc_str+{o}" for o in offs], ".hibytes"),
        "voc_str:", rows([c for w in words for c in list(w) + [0]])])

    # sampler: weight of a token = samp[min(zmax - z, SAMP_L - 1)]
    sampler = intengine.IntSampler(TEMPERATURE, TOP_P)
    assert len(sampler.table) <= 256 and sampler.table.max() < 65536
    write(out / "sampler.inc", [
        "; generated by python/export.py: temperature table and top-p threshold (src/sampler.s)",
        f"SAMP_L = {len(sampler.table)}", f"TOP_P = {sampler.top_p}",
        "samp_lo:", rows(sampler.table & 255), "samp_hi:", rows(sampler.table >> 8)])
    print(f"{out}: model.bin {len(b.data)} bytes, {cfg.n_layer} layers, {len(tok)} tokens")


if __name__ == "__main__":
    main()

#-----------------------------------------------------------------------------
