# The model and its training

## The world

`python/corpus.py` generates the training data (`data/corpus.txt`): one chat turn per line,
`<user> question<bot> answer<eos>`, about 10,000 lines.

- **16 animals**, each with color, home, food, sound, action, size and legs.
- **Facts** in several wordings, yes/no questions on color and food, size comparisons between
  two animals, and lists: who lives where, who has which color or how many legs.
- **Yes/no balance.** The questions answered "yes" are repeated as often as the "no" ones,
  otherwise the model learns to always answer "no".
- **Small talk**: greetings, who are you, thanks.
- **Stories** about one animal or about two: 5 meetings × 4 endings for every pair. They are
  compact, so that a whole turn fits the 64-token context (62 tokens at most).

`corpus.facts()` lists the 1,191 questions with a single correct answer, by category; the
stories are correct only if they are one of the possible ones for the request.

## Pipeline

```
make corpus     python/corpus.py     data/corpus.txt
make train      python/train.py      models/tokenizer.json, models/model.npz   (float32)
make qat        python/qat.py        models/model_qat.npz                      (nf4-aware)
make compile    python/compile.py    models/model_int.npz                      (integer image)
make evaluate   python/evaluate.py   float32 vs integer engine
```

`make model` runs the first four (about 10 minutes on a CPU). The geometry is a parameter of
`train.py` (`--n-layer`, `--context`, `--d-model`, ...); `qat.py` and `compile.py` read it from
the checkpoint, and `python/stats.py` counts the parameters and products of a geometry.

- **The transformer.** `python/chatcbm/model.py` has a hand-written forward and backward pass:
  pre-LayerNorm, GELU and weight tying. Training uses Adam with a cosine learning rate, 10,000
  steps of 32 windows.
- **The tokenizer.** BPE with merges that never cross word boundaries, 256 tokens.
- **QAT.** The forward pass uses the weights rounded to nf4, and the gradient updates the
  float weights (straight-through estimator).

## Results

| | float32 | integer engine (= C64) |
|---|---|---|
| validation loss | 0.358 | 0.368 |
| facts | 256/256 | 244/256 |
| size | 32/32 | 32/32 |
| legs | 32/32 | 27/32 |
| yes/no | 368/368 | 361/368 |
| comparisons | 480/480 | 472/480 |
| lists | 22/23 | 17/23 |
| **all exact answers** | **1190/1191** | **1153/1191** |
| stories about one animal | 48/48 | 47/48 |
| stories about two animals | 48/48 | 48/48 |

Choices behind the geometry:

- **4 layers instead of 3.** With 3 layers the integer engine gets 1,142 exact answers and
  92/96 stories, against 1,153 and 95. On an earlier corpus, before the yes/no balance, 3
  layers got only 9/48 stories about two animals right.
- **Context 64 instead of 32.** A whole story fits the window, and the KV cache never has to
  slide.
- **Wider is not better.** In an earlier study on a simpler corpus, a wider FFN or embedding
  brought no gain.
- **Where the integer engine loses.** The gap between float and integer engine comes from the
  integer arithmetic (log8 activations, rounding), not from the 4-bit weights: longer QAT
  does not reduce it.
