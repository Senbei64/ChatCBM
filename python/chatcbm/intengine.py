#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""All-integer inference: no floats at run time, only lookups, shifts, sums and compares.

This is the executable specification of the 6502 engine (src/), which reproduces it bit
for bit. compile() works offline in float: it calibrates the scales, quantizes, builds the
tables and produces an "image" made of integers only. IntModel runs the image one token at
a time with a KV cache.

Data types:

    weights and embeddings  nf4 (4 bits, Gaussian levels) with one scale 2^(k/16) per neuron, k int8
    activations             log8: 1 sign bit + 7 exponent bits, values in [-1, 1] times the scale
    biases                  int16 with one shift per tensor
    LayerNorm               g in log8, b int16; mean, variance and 1/sqrt with integers only

Run-time representations:

    log8      sign + exponent e (0..126, 127 = zero) + scale exponent k:
              value = ±2^((k - e)/16). k is per vector (dynamic, from the maximum), per head
              (static, calibrated: q, k, v in the cache) or per neuron (nf4 weights -> log8)
    acc       integer with a unit of 2^(u/16) per element: u = k_x + k_w - 16F in linear layers
    residual  int16 fixed point with R fractional bits (R is calibrated)
    "log"     L = log16 of a value: L ≈ 16·log2|v|; for an integer it comes from the number of
              significant bits (leading zeros) and a 128-entry table on the mantissa

Primitives (all on integers):
    ilog16(v)    integer > 0 -> round(16·log2 v)    leading zeros + LOGM[128] table
    pow2_16(E)   2^(E/16) as an integer             POW[E & 15] shifted by (E >> 4) - 15
    PROD[e]      2^(F - e/16) for e = e_x + e_w     one 256-entry table per F

Multiplying two log8 values is adding their exponents: PROD turns the sum back into a linear
product, so a dot product is a sequence of table lookups and additions.

The softmax constant 16·log2(e)/√hs is folded into the wq weights (fractional part) and into
the static scales (integer part): the integer score Q·K comes out, after a shift, already in
"exponent /16" units. So exp() is the pow2_16 table and the attention weights are born in log8.
"""
import json
import math

import numpy as np

from . import model
from .config import Config

LIN = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "ffn.w1", "ffn.w2")
BIAS = {"attn.wq": "attn.bq", "attn.wk": "attn.bk", "attn.wv": "attn.bv",
        "attn.wo": "attn.bo", "ffn.w1": "ffn.b1", "ffn.w2": "ffn.b2"}
EMBEDDINGS = ("wte", "wpe")
NEG_INF = -(1 << 40)
ZERO = 127  # log8 exponent reserved for zero

# NF4 (QLoRA, Dettmers et al. 2023): the quantiles of a Gaussian, normalized to [-1, 1]
NF4 = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])


# ---------------------------------------------------------------------------
# Quantization (offline)
# ---------------------------------------------------------------------------
def log_exponent(mag):
    """|x| in [0, 1] -> log8 exponent e, with |x| ≈ 2^(-e/16); 0 -> 127."""
    with np.errstate(divide="ignore"):
        e = np.round(-16 * np.log2(mag))
    return np.clip(np.where(mag > 0, e, ZERO), 0, ZERO).astype(np.int64)


NF4_SIGN = NF4 < 0                       # nf4 -> log8: a 16-entry table
NF4_EXP = log_exponent(np.abs(NF4))


def quantize_nf4(w, axis):
    """-> (nf4 codes, k) with a scale 2^(k/16) per row or column, k rounded to nearest
    (the largest value may be clipped to 1, by 2.2% at most)."""
    m = np.abs(w).max(axis, keepdims=True).astype(np.float64)
    k = np.round(16 * np.log2(np.where(m == 0, 1, m)))
    codes = np.abs((w / 2.0 ** (k / 16))[..., None] - NF4).argmin(-1)
    return codes.astype(np.uint8), k.astype(np.int64)


def bias_to_int(b, unit, bits=16):
    """Bias -> (q, s) with b ≈ q · 2^-s · unit: `bits`-bit integers, s chosen per tensor."""
    r = b / unit
    m = np.abs(r).max()
    s = bits - 1 - int(np.ceil(np.log2(m))) if m > 0 else 0
    q = np.clip(np.round(r * 2.0 ** s), -2 ** (bits - 1), 2 ** (bits - 1) - 1).astype(np.int64)
    return q, s


def log8_param(v):
    """Static parameter in log8 with scale 2^(k/16): the largest value has exponent 0."""
    m = np.abs(v).max()
    k = int(np.round(16 * np.log2(m))) if m > 0 else 0
    return v < 0, log_exponent(np.minimum(np.abs(v) / 2.0 ** (k / 16), 1)), k


# ---------------------------------------------------------------------------
# Tables (built offline, then only read)
# ---------------------------------------------------------------------------
def make_tables(F, F_attn, gelu_frac, gelu_range):
    e = np.arange(256)
    gelu_u = np.arange(-gelu_range << gelu_frac, gelu_range << gelu_frac) / 2 ** gelu_frac
    gelu = 0.5 * gelu_u * (1 + np.tanh(math.sqrt(2 / math.pi) * (gelu_u + 0.044715 * gelu_u ** 3)))
    return {
        # one byte each; 0 from e = 127 on: zero (127) needs no test and products below
        # 2^(F - 127/16) fall to zero, exactly as in the 6502 kernel
        "prod": np.where(e < ZERO, np.minimum(np.round(2.0 ** (F - e / 16)), 255), 0).astype(np.int64),
        "prod_attn": np.where(e < ZERO, np.minimum(np.round(2.0 ** (F_attn - e / 16)), 255), 0).astype(np.int64),
        "pow": np.round(2.0 ** (15 + np.arange(16) / 16)).astype(np.int64),  # 16 entries, 16 bits
        "logm": np.round(16 * np.log2(np.arange(128, 256) / 128)).astype(np.int64),  # 128 entries
        "gelu": np.round(gelu * 2 ** gelu_frac).astype(np.int64),         # 2·range·2^gf entries
    }


# ---------------------------------------------------------------------------
# Integer primitives
# ---------------------------------------------------------------------------
def _check(*arrays):
    for a in arrays:
        assert np.asarray(a).dtype.kind in "iub", f"float at run time: {np.asarray(a).dtype}"


def shift(v, n):
    """v · 2^n with an arithmetic shift (to the right it rounds down)."""
    n = np.clip(n, -62, 40)
    return np.where(n >= 0, v << np.maximum(n, 0), v >> np.maximum(-n, 0))


def bitlen(v):
    """Number of significant bits (v >= 0): on the 6502, counting leading zeros."""
    n = np.zeros(np.shape(v), np.int64)
    for i in range(48):
        n += (v >> i) > 0
    return n


class Tables:
    def __init__(self, t):
        self.__dict__.update(t)

    def ilog16(self, v):
        """v > 0 -> ≈ 16·log2 v. Mantissa = the 8 most significant bits (128..255)."""
        n = bitlen(v) - 1
        mant = shift(v, 7 - n)
        return 16 * n + self.logm[mant - 128]

    def pow2_16(self, E):
        """2^(E/16) as an integer (rounded down)."""
        return shift(self.pow[E & 15], (E >> 4) - 15)

    def to_log(self, v, u):
        """integer v with unit 2^(u/16) -> (negative, L, zero) with |value| ≈ 2^(L/16)."""
        zero = v == 0
        return v < 0, self.ilog16(np.where(zero, 1, np.abs(v))) + u, zero

    def to_fixed(self, neg, L, zero, frac):
        """(negative, L, zero) -> integer with `frac` fractional bits."""
        m = np.where(zero, 0, self.pow2_16(L + 16 * frac))
        return np.where(neg, -m, m)


def enc_dynamic(neg, L, zero):
    """log8 with one scale per vector (last axis): k = maximum, the largest value has e = 0."""
    k = np.where(zero, NEG_INF, L).max(-1, keepdims=True)
    k = np.where(k == NEG_INF, 0, k)
    e = k - L
    return neg, np.where(zero | (e > 126), ZERO, e), k


def enc_static(neg, L, zero, K):
    """log8 with a fixed scale 2^(K/16): above the maximum it saturates to e = 0."""
    e = np.maximum(K - L, 0)
    return neg, np.where(zero | (e > 126), ZERO, e)


def sat16(v):
    return np.clip(v, -32768, 32767)


# ---------------------------------------------------------------------------
# Compilation (offline, in float): calibration, quantization, integer image
# ---------------------------------------------------------------------------
def _calibrate(p, cfg, ids):
    """Maxima of q, k, v per head and of the residual stream, over a batch of windows."""
    stats = {"resid": float(np.abs(p["wte"][ids] + p["wpe"][:ids.shape[1]]).max())}
    x = p["wte"][ids] + p["wpe"][:ids.shape[1]]
    for i in range(cfg.n_layer):
        b = f"h{i}."
        h, _ = model.layernorm_fwd(x, p[b + "ln1.g"], p[b + "ln1.b"])
        att, (_, q, k, v, _, _) = model.attention_fwd(h, p, b + "attn.", cfg.n_head)
        for n, t in (("q", q), ("k", k), ("v", v)):
            stats[f"{b}{n}"] = np.abs(t).max((0, 2, 3))
        x = x + att
        h, _ = model.layernorm_fwd(x, p[b + "ln2.g"], p[b + "ln2.b"])
        f, _ = model.ffn_fwd(h, p, b + "ffn.")
        x = x + f
        stats["resid"] = max(stats["resid"], float(np.abs(x).max()))
    return stats


def compile(p, cfg, calib_ids, F=8, F_attn=8, F_ln=9, G=10, gelu_frac=5, gelu_range=4,
            margin=1.0):
    """Float model -> integer image (dict of integer arrays)."""
    p = dict(p)
    hs = cfg.d_model // cfg.n_head
    # softmax: z = 16·log2(e)·q·k/√hs. 2^(c/16) = 16·log2(e)/√hs; the fractional part of c goes into wq
    c = 16 * math.log2(16 * math.log2(math.e) / math.sqrt(hs))
    c_int = math.floor(c)
    fold = 2.0 ** ((c - c_int) / 16)
    for i in range(cfg.n_layer):
        b = f"h{i}.attn."
        p[b + "wq"], p[b + "bq"] = p[b + "wq"] * fold, p[b + "bq"] * fold
    st = _calibrate(p, cfg, calib_ids)

    img = {"F": F, "F_attn": F_attn, "F_ln": F_ln, "G": G, "gelu_frac": gelu_frac,
           "gelu_range": gelu_range}
    img["R"] = 15 - math.ceil(math.log2(st["resid"] * margin))  # fractional bits of the residual

    def k_of(m):
        return np.ceil(16 * np.log2(m * margin)).astype(np.int64)

    for i in range(cfg.n_layer):
        b = f"h{i}."
        Kk, Kv = k_of(st[b + "k"]), k_of(st[b + "v"])
        Kq = k_of(st[b + "q"])
        Kq += (-(Kq + Kk + c_int)) % 16                    # Kq + Kk + c is a multiple of 16
        img[b + "Kq"], img[b + "Kk"], img[b + "Kv"] = Kq, Kk, Kv
        img[b + "Zs"] = (Kq + Kk + c_int) // 16 - F_attn   # z = S << Zs (or >> -Zs)
    # logits -> z = 16·log2(e)·logit (for the sampling softmax); Zl is rounded
    cl = 16 * math.log2(16 * math.log2(math.e))
    img["Zl"] = round(cl)
    img["Zl_err"] = cl - img["Zl"]  # only for float metrics, not used at run time

    for name in [k for k in p if k.endswith(LIN) or k in EMBEDDINGS]:
        # W (input, output): one scale per column; wte, wpe: one per row (token, position)
        codes, k = quantize_nf4(p[name], 1 if name in EMBEDDINGS else 0)
        img[name + ".q"], img[name + ".k"] = codes, k.reshape(-1).astype(np.int64)
        suffix = name.split(".", 1)[-1]
        bname = "b_out" if name == "wte" else name[:-len(suffix)] + BIAS[suffix] if suffix in BIAS else None
        if bname:
            q, s = bias_to_int(p[bname], 2.0 ** (img[name + ".k"] / 16))
            img[bname + ".i"], img[bname + ".s"] = q.reshape(-1), s
    for name in p:
        if name.endswith(".g"):
            sign, e, k = log8_param(p[name])
            img[name + ".neg"], img[name + ".e"], img[name + ".k"] = sign.astype(np.int64), e, k
            q, s = bias_to_int(p[name[:-1] + "b"], 2.0 ** (k / 16))
            img[name[:-1] + "b.i"], img[name[:-1] + "b.s"] = q, s
    img.update({"t." + k: v for k, v in make_tables(F, F_attn, gelu_frac, gelu_range).items()})
    img["config"] = cfg.to_dict()
    return img


def save(path, img):
    arrays = {k: np.asarray(v) for k, v in img.items() if k not in ("config", "Zl_err")}
    for k, v in arrays.items():
        assert v.dtype.kind in "iub", f"{k} is not an integer array"
    meta = {"config": img["config"], "Zl_err": img["Zl_err"]}
    np.savez(path, __intrun__=np.array(json.dumps(meta)), **arrays)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["__intrun__"]))
        img = {k: data[k] for k in data.files if k != "__intrun__"}
    img = {k: (int(v) if v.ndim == 0 else v.astype(np.int64)) for k, v in img.items()}
    img.update(meta)
    return img


def image_bytes(img):
    """Bytes needed on an 8-bit machine: nf4 at 4 bits, log8 codes and scales at 8,
    biases and integer tables at 16 (prod and logm at 8)."""
    out = {"nf4 weights": 0, "scales k": 0, "biases": 0, "LayerNorm": 0, "tables": 0, "constants": 0}
    for k, v in img.items():
        if k in ("config", "Zl_err"):
            continue
        n = np.size(v)
        if k.endswith(".q"):
            out["nf4 weights"] += n // 2
        elif k.startswith("t."):
            out["tables"] += n * (1 if k in ("t.prod", "t.prod_attn", "t.logm") else 2)
        elif ".ln" in k or k.startswith("lnf"):
            out["LayerNorm"] += n * (2 if k.endswith(".i") else 1)
        elif k.endswith((".i", ".s")):
            out["biases"] += n * (2 if k.endswith(".i") else 1)
        elif k.endswith(".k"):
            out["scales k"] += n
        else:
            out["constants"] += n
    return out


# ---------------------------------------------------------------------------
# Run time: integers only
# ---------------------------------------------------------------------------
class IntModel:
    def __init__(self, img):
        self.img = img
        self.cfg = Config(**img["config"])
        self.t = Tables({k[2:]: v for k, v in img.items() if k.startswith("t.")})
        self.F, self.FA, self.F_ln, self.G, self.R = (img[k] for k in ("F", "F_attn", "F_ln", "G", "R"))
        self.gf, self.gr = img["gelu_frac"], img["gelu_range"]
        # nf4 weights -> log8: a 16-entry table (here once, on the 6502 in the inner loop)
        self.W = {}
        for name in [k[:-2] for k in img if k.endswith(".q")]:
            codes = img[name + ".q"]
            if name == "wte":
                codes = codes.T  # as output weights: (C, V), one neuron per token
            self.W[name] = (NF4_SIGN[codes].astype(np.int64), NF4_EXP[codes], img[name + ".k"])
        self.stats = {"sat": 0, "sums": 0}

    # ---- blocks ---------------------------------------------------------------
    def linear(self, x, wname, bname):
        """x = (negative, e, k) log8 per vector. -> (acc int16, unit u per element)."""
        sx, ex, kx = x
        sw, ew, kw = self.W[wname]
        e = ex[:, :, None] + ew[None]
        prod = np.where((ex[:, :, None] == ZERO) | (ew[None] == ZERO), 0, self.t.prod[e])
        acc = np.where(sx[:, :, None] ^ sw[None], -prod, prod).sum(1)
        q, s = self.img[bname + ".i"], self.img[bname + ".s"]
        nq, Lq, zq = self.t.to_log(q, 0)
        b = np.where(zq, 0, self.t.pow2_16(Lq + 16 * (self.F - s) - kx))  # bias / input scale
        out = sat16(acc + np.where(nq, -b, b))
        self._count(acc + np.where(nq, -b, b), out)
        _check(out)
        return out, kx + kw - 16 * self.F

    def layernorm(self, x, pre):
        """x int (residual) -> (acc int16, unit)."""
        t, F = self.t, self.F_ln
        n = x.shape[-1]
        d = x - (x.sum(-1, keepdims=True) >> (n.bit_length() - 1))       # mean: sum >> 5
        nd, Ld, zd = t.to_log(d, 0)
        _, ed, _ = enc_dynamic(nd, Ld, zd)                                 # d / scale_d in log8
        sq = np.where(ed == ZERO, 0, t.pow2_16(16 * F - 2 * ed))            # square: e × 2
        var = np.maximum(sq.sum(-1, keepdims=True) >> (n.bit_length() - 1), 1)
        r = (16 * F - t.ilog16(var) + 1) >> 1                               # 1/sqrt: e / 2
        ng, eg, kg = self.img[pre + "g.neg"], self.img[pre + "g.e"], self.img[pre + "g.k"]
        y = t.pow2_16(16 * F - (ed + eg - r))
        y = np.where((ed == ZERO) | (eg == ZERO), 0, np.where(nd ^ ng.astype(bool), -y, y))
        out = sat16(y + shift(self.img[pre + "b.i"], F - self.img[pre + "b.s"]))
        _check(out)
        return out, np.full(out.shape, kg - 16 * F)

    def to_log8(self, acc_u):
        return enc_dynamic(*self.t.to_log(*acc_u))

    def to_resid(self, acc_u):
        return self.t.to_fixed(*self.t.to_log(*acc_u), self.R)

    def gelu(self, acc_u):
        u = self.t.to_fixed(*self.t.to_log(*acc_u), self.gf)
        top = self.gr << self.gf
        idx = np.clip(u + top, 0, 2 * top - 1)
        g = np.where(u >= top, u, np.where(u < -top, 0, self.t.gelu[idx]))
        return g, np.full(g.shape, -16 * self.gf)

    # ---- autoregressive step ----------------------------------------------------
    def reset(self, batch):
        cfg = self.cfg
        H, hs, T = cfg.n_head, cfg.d_model // cfg.n_head, cfg.context
        shape = (cfg.n_layer, batch, H, T, hs)
        self.kn, self.ke = np.zeros(shape, bool), np.full(shape, ZERO)
        self.vn, self.ve = np.zeros(shape, bool), np.full(shape, ZERO)
        self.pos = 0

    def step(self, tokens):
        """tokens (B,) at position self.pos -> z (B, V): logits · 16·log2(e), integers."""
        cfg, t, img = self.cfg, self.t, self.img
        B, H, C = len(tokens), cfg.n_head, cfg.d_model
        hs, pos = C // H, self.pos
        assert pos < cfg.context

        def emb(name, rows):
            codes = img[name + ".q"][rows]
            L = img[name + ".k"][rows][:, None] - NF4_EXP[codes]
            return t.to_fixed(NF4_SIGN[codes], L, NF4_EXP[codes] == ZERO, self.R)

        x = emb("wte", tokens) + emb("wpe", np.full(B, pos))
        for i in range(cfg.n_layer):
            b = f"h{i}."
            h = self.to_log8(self.layernorm(x, b + "ln1."))
            qkv = {}
            for n in "qkv":
                acc, u = self.linear(h, b + f"attn.w{n}", b + f"attn.b{n}")
                neg, L, zero = t.to_log(acc, u)
                K = img[b + "K" + n][:, None]
                qkv[n] = enc_static(neg.reshape(B, H, hs), L.reshape(B, H, hs), zero.reshape(B, H, hs), K)
            self.kn[i, :, :, pos], self.ke[i, :, :, pos] = qkv["k"]
            self.vn[i, :, :, pos], self.ve[i, :, :, pos] = qkv["v"]
            kn, ke = self.kn[i, :, :, :pos + 1], self.ke[i, :, :, :pos + 1]
            vn, ve = self.vn[i, :, :, :pos + 1], self.ve[i, :, :, :pos + 1]
            qn, qe = qkv["q"]
            # Q·K scores: (B, H, t)
            e = qe[:, :, None, :] + ke
            pr = np.where((qe[:, :, None, :] == ZERO) | (ke == ZERO), 0, t.prod_attn[e])
            S = np.where(qn[:, :, None, :] ^ kn, -pr, pr).sum(-1)
            z = shift(S, img[b + "Zs"][None, :, None])
            # softmax in the log domain: exp is a table, the weights a come out in log8
            ez = z.max(-1, keepdims=True) - z
            pw = np.where(ez > 16 * self.G, 0, t.pow2_16(16 * self.G - ez))
            ea = ez + t.ilog16(pw.sum(-1, keepdims=True)) - 16 * self.G
            ea = np.where(ea > 126, ZERO, ea)
            # A·V: (B, H, hs), a is always positive
            e = ea[..., None] + ve
            pr = np.where((ea[..., None] == ZERO) | (ve == ZERO), 0, t.prod_attn[e])
            y = np.where(vn, -pr, pr).sum(2).reshape(B, C)
            uy = np.repeat(img[b + "Kv"], hs)[None] - 16 * self.FA
            att = self.linear(self.to_log8((y, uy)), b + "attn.wo", b + "attn.bo")
            x = sat16(x + self.to_resid(att))
            h = self.to_log8(self.layernorm(x, b + "ln2."))
            g = self.gelu(self.linear(h, b + "ffn.w1", b + "ffn.b1"))
            f = self.linear(self.to_log8(g), b + "ffn.w2", b + "ffn.b2")
            x = sat16(x + self.to_resid(f))
            _check(x, z, ea, y)
        h = self.to_log8(self.layernorm(x, "lnf."))
        acc, u = self.linear(h, "wte", "b_out")
        z = t.to_fixed(*t.to_log(acc, u + img["Zl"]), 0)
        _check(z)
        self.pos += 1
        return z

    def _count(self, raw, out):
        self.stats["sat"] += int((raw != out).sum())
        self.stats["sums"] += out.size

    # ---- use --------------------------------------------------------------------
    def run(self, ids):
        """ids (B, T) -> z (B, T, V), one token at a time with the KV cache."""
        self.reset(ids.shape[0])
        return np.stack([self.step(ids[:, i]) for i in range(ids.shape[1])], 1)

    def logits(self, z):
        """For metrics only: integer z -> float logits."""
        return z * (math.log(2) / 16) * 2.0 ** (self.img["Zl_err"] / 16)

    def generate(self, ids, max_new_tokens=60, stop_ids=(), banned=(), sampler=None):
        ids = list(ids)
        T = self.cfg.context

        def window():  # positions are absolute: when the window slides, recompute it
            self.reset(1)
            for tok in ids[-T:]:
                z = self.step(np.array([tok]))
            return z

        z = window()
        for _ in range(max_new_tokens):
            z = z[0].copy()
            z[list(banned)] = NEG_INF
            nxt = int(np.argmax(z)) if sampler is None else sampler(z)
            if nxt in stop_ids:
                return
            ids.append(nxt)
            yield nxt
            z = self.step(np.array([nxt])) if len(ids) <= T else window()


class IntSampler:
    """Temperature + top-p with integers only: a 2^(-e/(16·T)) table, the top-p threshold in
    /256 and an xorshift32 random number with rejection (no multiplications at run time)."""

    def __init__(self, temperature=0.6, top_p=0.9, seed=1, G=14):
        t = np.round(2.0 ** (G - np.arange(4096) / (16 * temperature))).astype(np.int64)  # offline
        self.table = t[:int(np.argmax(t == 0)) + 1]  # up to the first zero
        self.top_p = round(top_p * 256)
        self.state = seed or 1

    def _rand(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x

    def __call__(self, z):
        e = np.minimum(z.max() - z, len(self.table) - 1)
        p = self.table[e]
        order = np.argsort(-p, kind="stable")
        cum = np.cumsum(p[order])
        thr = (int(cum[-1]) * self.top_p) >> 8   # constant · integer: shifts and sums on the 6502
        cut = int(np.searchsorted(cum, thr)) + 1
        total = int(cum[cut - 1])
        mask = (1 << total.bit_length()) - 1
        while True:                               # uniform draw in [0, total)
            r = self._rand() & mask
            if r < total:
                return int(order[np.searchsorted(cum[:cut], r, side="right")])

#-----------------------------------------------------------------------------
