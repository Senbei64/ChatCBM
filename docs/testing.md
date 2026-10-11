# Tests

Every layer of the project is checked against the one below it:

1. **Float model.** `python/tests` checks the hand-written backward pass against finite
   differences, the tokenizer, and that the integer engine stays close to the float model
   while using integers only.
2. **6502 modules.** There is one unit test per module, a C program in `tests/unit/` built for
   the `sim6502` target and run in sim65. `python/vectors.py` writes its inputs and the outputs
   computed by the Python integer engine, and the 6502 code must match them exactly. Inputs are
   real intermediate values of the model (weights, LayerNorm outputs, the q, k, v of a real
   prompt) plus random and edge cases: zeros, saturation, the full 64-token context.
3. **Whole program on sim65.** `make system` runs `build/sim65/chatcbm.prg` on a list of
   questions and compares its answers with the Python engine.
4. **C64 in VICE.** `make e2e` does the same with the real C64 program in the emulator.

| unit test | what it checks |
|---|---|
| `prim` | ilog16 and pow2_16 over their whole range |
| `matmul` | 32 × 32 blocks of real weights, real and random inputs |
| `log8` | dynamic and static log8 encoding, with a stride as in the KV cache |
| `linear` | biases and saturation, 32-output layers and a block of logits |
| `layernorm` | residuals of the model and random ones, a constant vector |
| `gelu`, `residual`, `logits` | the element-wise operations, inside and outside their ranges |
| `attention` | heads of a real prompt, random caches up to 64 positions, every shift |
| `model` | logits after a prompt with full and reduced steps, a dozen greedy tokens |
| `tokenizer` | upper case, tabs, newlines, underscores, digits, apostrophes, too long questions |
| `sampler` | argmax and consecutive samples on real and synthetic logits |

The residual test found a branch that never saturated negative contributions above 65535.
The trained model never reaches it, but now the 6502 code matches the Python engine there too.

## End-to-end protocol

`tests/e2e.py` writes the questions to `build/e2e/<target>/questions` and the expected answers
to `expected`. For every question these are the greedy answer and the answer sampled with
seed 2026. The program writes its own to `answers`:

- **sim65.** `sim65 chatcbm.prg questions answers`.
- **VICE.** Drive 8 is the host directory (`+drive8truedrive -busdevice8 -fs8 <dir>`, as in
  [VIC20UnitTests](https://github.com/Senbei64/VIC20UnitTests)). The test build
  `chatcbm-test.prg` opens `QUESTIONS` and `ANSWERS` with the kernal file routines, shows the
  sampled answers on screen and ends with a JAM instruction, so with `-jamaction 5` VICE quits.
  `--screenshot` saves the final screen.

The questions do not go through `-keybuf`. VICE writes the keyboard buffer and its counter at
$C6 while the program is computing, when that zero page belongs to the engine. A real C64 does
not read the keyboard at all while interrupts are off.

## CI

GitHub Actions runs `make` on Ubuntu with cc65 and NumPy from the distribution: Python tests,
unit tests, the sim65 system test and the C64 build. The VICE test runs only locally, because
Ubuntu's VICE package ships without the Commodore ROMs.
