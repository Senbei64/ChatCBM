#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------

# main make targets
.PHONY: all corpus train qat compile model evaluate pytest clean

# keep intermediate files
.SECONDARY:

# build on Windows and Linux: file operations go through Python, so the same
# commands work with cmd, sh and bash
ifeq ($(OS),Windows_NT)
  PYTHON = python
else
  PYTHON = python3
endif
MKDIR = $(PYTHON) -c "import os, sys; [os.makedirs(d, exist_ok=True) for d in sys.argv[1:]]"
RMDIR = $(PYTHON) -c "import shutil, sys; [shutil.rmtree(d, True) for d in sys.argv[1:]]"
TOUCH = $(PYTHON) -c "import sys; open(sys.argv[1], 'w').close()"

# - BUILD --------------------------------------------------------------------
all: pytest

# - TESTS --------------------------------------------------------------------
# Python: hand-written backward pass, tokenizer, integer engine
pytest:
	$(PYTHON) -m unittest discover -s python/tests -t python

# - MODEL --------------------------------------------------------------------
# training data: a small controlled world of animals (deterministic)
corpus:
	$(PYTHON) python/corpus.py

# float32 model and tokenizer (about 5 minutes on a CPU)
train: corpus
	$(PYTHON) python/train.py

# quantization-aware training with nf4 weights (about 3 minutes)
qat:
	$(PYTHON) python/qat.py

# all-integer image for the 6502 engine
compile:
	$(PYTHON) python/compile.py

# the whole pipeline, from the corpus to models/model_int.npz
model: train
	$(MAKE) qat
	$(MAKE) compile

# float32 and integer engine on every fact and on sampled stories (a few minutes)
evaluate:
	$(PYTHON) python/evaluate.py

# - CLEAN --------------------------------------------------------------------
clean:
	$(RMDIR) build

#-----------------------------------------------------------------------------
