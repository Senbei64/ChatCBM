#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Tiny decoder-only transformer, written from scratch in NumPy.

Both the forward pass (inference) and the backward pass (gradients for training).
Each block is pre-LayerNorm, as in GPT-2:

    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

Input:  token embedding + positional embedding (a lookup and a sum, no multiplications)
Output: final LayerNorm, then projection onto the vocabulary with the same matrix as the
        embeddings (weight tying) plus an output bias.
"""
import json
import math

import numpy as np

from .config import Config

GELU_K = math.sqrt(2.0 / math.pi)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
def init_params(cfg, seed=0):
    rng = np.random.default_rng(seed)
    C, F, V, T = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.context
    proj = 1.0 / math.sqrt(2 * cfg.n_layer)  # smaller projections into the residual stream

    def w(n_in, n_out, scale=1.0):
        return (rng.standard_normal((n_in, n_out)) * scale / math.sqrt(n_in)).astype(np.float32)

    def zeros(n):
        return np.zeros(n, np.float32)

    def ones(n):
        return np.ones(n, np.float32)

    p = {
        "wte": (rng.standard_normal((V, C)) * 0.1).astype(np.float32),
        "wpe": (rng.standard_normal((T, C)) * 0.1).astype(np.float32),
    }
    for i in range(cfg.n_layer):
        b = f"h{i}."
        p[b + "ln1.g"], p[b + "ln1.b"] = ones(C), zeros(C)
        for n in "qkv":
            p[b + f"attn.w{n}"], p[b + f"attn.b{n}"] = w(C, C), zeros(C)
        p[b + "attn.wo"], p[b + "attn.bo"] = w(C, C, proj), zeros(C)
        p[b + "ln2.g"], p[b + "ln2.b"] = ones(C), zeros(C)
        p[b + "ffn.w1"], p[b + "ffn.b1"] = w(C, F), zeros(F)
        p[b + "ffn.w2"], p[b + "ffn.b2"] = w(F, C, proj), zeros(C)
    p["lnf.g"], p["lnf.b"] = ones(C), zeros(C)
    p["b_out"] = zeros(V)
    return p


def count_params(p):
    return sum(v.size for v in p.values())


def save(path, p, cfg):
    np.savez(path, __config__=np.array(json.dumps(cfg.to_dict())), **p)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        cfg = Config(**json.loads(str(data["__config__"])))
        p = {k: data[k] for k in data.files if k != "__config__"}
    return p, cfg


# ---------------------------------------------------------------------------
# Layers: forward returns (output, cache), backward uses the cache
# ---------------------------------------------------------------------------
def _flat(x):
    return x.reshape(-1, x.shape[-1])


def layernorm_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    rstd = 1.0 / np.sqrt((xc * xc).mean(-1, keepdims=True) + eps)
    xhat = xc * rstd
    return xhat * g + b, (xhat, rstd, g)


def layernorm_bwd(dy, cache):
    xhat, rstd, g = cache
    dg = _flat(dy * xhat).sum(0)
    db = _flat(dy).sum(0)
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True)
                 - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def attention_fwd(x, p, pre, n_head):
    B, T, C = x.shape
    hs = C // n_head

    def heads(t):  # (B,T,C) -> (B,H,T,hs)
        return t.reshape(B, T, n_head, hs).transpose(0, 2, 1, 3)

    q = heads(x @ p[pre + "wq"] + p[pre + "bq"])
    k = heads(x @ p[pre + "wk"] + p[pre + "bk"])
    v = heads(x @ p[pre + "wv"] + p[pre + "bv"])

    scale = 1.0 / math.sqrt(hs)
    s = (q @ k.transpose(0, 1, 3, 2)) * scale
    causal = np.triu(np.ones((T, T), dtype=bool), 1)  # a token does not see the future
    s = np.where(causal, -1e9, s)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    a = e / e.sum(-1, keepdims=True)

    y = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
    out = y @ p[pre + "wo"] + p[pre + "bo"]
    return out, (x, q, k, v, a, y)


def attention_bwd(dout, cache, p, pre, n_head, g):
    x, q, k, v, a, y = cache
    B, T, C = x.shape
    hs = C // n_head

    g[pre + "wo"] = _flat(y).T @ _flat(dout)
    g[pre + "bo"] = _flat(dout).sum(0)
    dy = (dout @ p[pre + "wo"].T).reshape(B, T, n_head, hs).transpose(0, 2, 1, 3)

    da = dy @ v.transpose(0, 1, 3, 2)
    dv = a.transpose(0, 1, 3, 2) @ dy
    ds = a * (da - (da * a).sum(-1, keepdims=True)) * (1.0 / math.sqrt(hs))
    dq = ds @ k
    dk = ds.transpose(0, 1, 3, 2) @ q

    dx = np.zeros_like(x)
    for n, d in (("q", dq), ("k", dk), ("v", dv)):
        d = d.transpose(0, 2, 1, 3).reshape(B, T, C)
        g[pre + f"w{n}"] = _flat(x).T @ _flat(d)
        g[pre + f"b{n}"] = _flat(d).sum(0)
        dx += d @ p[pre + f"w{n}"].T
    return dx


def ffn_fwd(x, p, pre):
    u = x @ p[pre + "w1"] + p[pre + "b1"]
    t = np.tanh(GELU_K * (u + 0.044715 * u ** 3))
    h = 0.5 * u * (1 + t)  # GELU
    out = h @ p[pre + "w2"] + p[pre + "b2"]
    return out, (x, u, t, h)


def ffn_bwd(dout, cache, p, pre, g):
    x, u, t, h = cache
    g[pre + "w2"] = _flat(h).T @ _flat(dout)
    g[pre + "b2"] = _flat(dout).sum(0)
    dh = dout @ p[pre + "w2"].T
    du = dh * (0.5 * (1 + t) + 0.5 * u * (1 - t * t) * GELU_K * (1 + 3 * 0.044715 * u * u))
    g[pre + "w1"] = _flat(x).T @ _flat(du)
    g[pre + "b1"] = _flat(du).sum(0)
    return du @ p[pre + "w1"].T


# ---------------------------------------------------------------------------
# Whole model
# ---------------------------------------------------------------------------
def forward(p, cfg, idx):
    """idx: integers (B, T) with T <= context. Returns logits (B, T, V) and the cache."""
    B, T = idx.shape
    assert T <= cfg.context, f"sequence of {T} tokens exceeds the context of {cfg.context}"
    x = p["wte"][idx] + p["wpe"][:T]
    caches = []
    for i in range(cfg.n_layer):
        b = f"h{i}."
        h, c_ln1 = layernorm_fwd(x, p[b + "ln1.g"], p[b + "ln1.b"])
        att, c_att = attention_fwd(h, p, b + "attn.", cfg.n_head)
        x = x + att
        h, c_ln2 = layernorm_fwd(x, p[b + "ln2.g"], p[b + "ln2.b"])
        f, c_ffn = ffn_fwd(h, p, b + "ffn.")
        x = x + f
        caches.append((c_ln1, c_att, c_ln2, c_ffn))
    hf, c_lnf = layernorm_fwd(x, p["lnf.g"], p["lnf.b"])
    logits = hf @ p["wte"].T + p["b_out"]
    return logits, (idx, caches, hf, c_lnf)


def backward(p, cfg, cache, dlogits):
    idx, caches, hf, c_lnf = cache
    T = idx.shape[1]
    g = {k: np.zeros_like(v) for k, v in p.items()}

    g["b_out"] = _flat(dlogits).sum(0)
    g["wte"] += _flat(dlogits).T @ _flat(hf)
    dx, g["lnf.g"], g["lnf.b"] = layernorm_bwd(dlogits @ p["wte"], c_lnf)

    for i in reversed(range(cfg.n_layer)):
        b = f"h{i}."
        c_ln1, c_att, c_ln2, c_ffn = caches[i]
        dh = ffn_bwd(dx, c_ffn, p, b + "ffn.", g)
        d, g[b + "ln2.g"], g[b + "ln2.b"] = layernorm_bwd(dh, c_ln2)
        dx = dx + d
        dh = attention_bwd(dx, c_att, p, b + "attn.", cfg.n_head, g)
        d, g[b + "ln1.g"], g[b + "ln1.b"] = layernorm_bwd(dh, c_ln1)
        dx = dx + d

    np.add.at(g["wte"], idx, dx)
    g["wpe"][:T] += dx.sum(0)
    return g


def loss_and_grads(p, cfg, idx, targets):
    """Next-token cross-entropy. Targets < 0 are ignored."""
    logits, cache = forward(p, cfg, idx)
    lg = _flat(logits)
    tg = targets.reshape(-1)
    valid = tg >= 0
    n = max(int(valid.sum()), 1)
    tsafe = np.where(valid, tg, 0)
    rows = np.arange(len(tg))

    lg = lg - lg.max(1, keepdims=True)
    e = np.exp(lg)
    sum_e = e.sum(1)
    logp = lg[rows, tsafe] - np.log(sum_e)
    loss = -float((logp * valid).sum()) / n

    dlogits = e / sum_e[:, None]
    dlogits[rows, tsafe] -= 1
    dlogits *= (valid[:, None] / n).astype(dlogits.dtype)
    grads = backward(p, cfg, cache, dlogits.reshape(logits.shape))
    return loss, grads

#-----------------------------------------------------------------------------
