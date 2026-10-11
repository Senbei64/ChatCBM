# The integer engine

The 6502 code (`src/`) reproduces bit for bit `python/chatcbm/intengine.py`, an inference
engine that uses integers only: lookups, shifts, additions and compares. The Python file is
the executable specification; `python/compile.py` turns the trained float model into its
integer image.

## Numbers

| data | format |
|---|---|
| weights and embeddings | **nf4**: 4 bits on 16 Gaussian levels, one scale 2^(k/16) per neuron, k int8 |
| activations | **log8**: a sign bit and a 7-bit exponent e, \|v\| = 2^((k - e)/16), e = 127 for zero |
| product weight × input | sum of the exponents, then `PROD[e] = 2^(F - e/16)`, a 256-byte table |
| accumulators | int16, with a unit 2^(u/16) per element |
| biases | int16 with one shift per tensor, stored as logarithms for the 6502 |
| residual stream | int16 fixed point, R fractional bits (calibrated) |
| q, k, v and the KV cache | log8 with a static scale per head (calibrated) |

Two primitives turn between the domains: `ilog16(v)` ≈ 16·log2 v (leading zeros and a
128-entry table on the mantissa) and `pow2_16(E)` = 2^(E/16) (a 16-entry table and a shift).

## The matrix kernel

`src/matmul.s` multiplies 32 log8 inputs by a 32 × 32 block of packed nf4 weights, about 30
cycles per product:

- **Table pointer moved by the input exponent** (from the author's 2018 notes, "exp with
  bias"). For a whole input row the pointer is `tp = PROD + e_x`, so product and conversion
  are a single `ADC/SBC (tp),Y` with Y = weight exponent.
- **Free signs**: the weight sign is bit 7 of its log8 byte, and `LDY` sets the N flag.
  The input sign is decided once per row, with two copies of the loop.
- **No tests for zero**: zero is exponent 127 and `PROD[e >= 127] = 0`. A zero input skips its
  row.
- **Packed nf4**: two weights per byte. Even rows go through a 256-entry table on the low
  nibble, odd rows through one on the high nibble; each returns the log8 byte of the weight.
  This costs 6 cycles per product and halves the memory of the weights.

## One step

`src/model.s` runs one token through the model:

1. The embeddings of token and position are added to the residual stream.
2. Each layer runs: LayerNorm, log8, q k v, k and v into the KV cache, 4 attention heads,
   log8, wo, residual, LayerNorm, log8, w1, GELU, log8, w2, residual.
3. Then the final LayerNorm, log8, and the 256 logits with the embedding matrix.

Some of the steps need extra tricks:

- **Softmax in the log domain** (`src/attention.s`). The constant 16·log2(e)/√8 is folded into
  wq and into the static scales, so the shifted Q·K score is already an exponent: exp is the
  `pow2_16` table, and the attention weights come out as log8 exponents for the A·V products.
- **LayerNorm** (`src/layernorm.s`). The mean is a sum and a shift. Squaring doubles a log8
  exponent and 1/sqrt halves it.
- **Biases** (`src/linear.s`). They are prepared offline as logarithms, so adding one costs a
  `pow2_16` lookup and a 16-bit addition with saturation.
- **Sampler** (`src/sampler.s`). Temperature uses a 2^(-e/(16·T)) table and top-p uses no
  sorting: tokens with the same weight form contiguous groups of exponents. Random numbers come
  from xorshift32.

Prompt tokens except the last run a reduced step: only k and v of the last layer, no logits.

| step (4 layers, sim65) | cycles |
|---|---|
| full step, positions 0-9 | 2.23 M |
| full step, positions 10-19 | 2.33 M |
| reduced prompt step | 1.5 M |

A full step is about 33,000 products, so the kernel takes about 1 M cycles; the rest goes to
biases, LayerNorm, attention and logits.

## Modules

| file | content |
|---|---|
| `prim.s` | ilog16, pow2_16, fixed-point helpers |
| `buffers.s` | activation buffers shared by the modules |
| `matmul.s` | the matrix kernel |
| `linear.s` | bias and saturation of a linear layer |
| `log8.s` | conversions to log8: dynamic scale (layer inputs), static scale (KV cache) |
| `layernorm.s`, `gelu.s`, `residual.s`, `logits.s` | the element-wise operations |
| `attention.s` | one attention head |
| `model.s` | the step, the KV cache, the model image |
| `tokenizer.s` | BPE tokenizer of the question |
| `sampler.s` | argmax, temperature and top-p |
| `chat.s` | one chat turn |
| `c64/main.s`, `sim65/main.c` | the machine-dependent main programs |

`python/export.py` writes the model image and the generated includes (`build/gen/`): constants,
layer descriptors, tables, tokenizer and sampler data.
