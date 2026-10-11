#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------

# main make targets
.PHONY: all corpus train qat compile model evaluate chat test pytest unit clean

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

# build directories
GEN = build/gen
OBJ = build/obj
TEST = build/test

# - BUILD --------------------------------------------------------------------
all: test

# files generated from the model, the tokenizer and the Python engine
$(GEN)/stamp: models/model_int.npz models/tokenizer.json python/export.py python/vectors.py               python/chatcbm/intengine.py python/chatcbm/tokenizer.py
	$(PYTHON) python/export.py --out $(GEN)
	$(PYTHON) python/vectors.py --out $(GEN)
	$(TOUCH) $@

# 6502 modules: one object per source file
$(OBJ)/%.o: src/%.s $(GEN)/stamp
	$(MKDIR) $(OBJ)
	ca65 -I $(GEN) --bin-include-dir $(GEN) -o $@ $<

# helpers of the unit tests written in assembler
$(OBJ)/%.o: tests/unit/%.s
	$(MKDIR) $(OBJ)
	ca65 -o $@ $<

# - TESTS --------------------------------------------------------------------
test: pytest unit

# Python: hand-written backward pass, tokenizer, integer engine
pytest:
	$(PYTHON) -m unittest discover -s python/tests -t python

# 6502 unit tests on sim65: tests/unit/test_<name>.c against the vectors of the
# Python engine, linked with the modules listed in UNIT_<name>
UNIT_TESTS = prim matmul log8 linear
UNIT_prim = prim
UNIT_matmul = matmul buffers aligned
UNIT_log8 = log8 prim buffers
UNIT_linear = linear matmul prim buffers

unit: $(UNIT_TESTS:%=$(TEST)/%.ok)

$(OBJ)/test_%.o: tests/unit/test_%.c tests/unit/check.h $(GEN)/stamp
	$(MKDIR) $(OBJ)
	cl65 -t sim6502 -c -O -I $(GEN) -o $@ $<

.SECONDEXPANSION:
$(TEST)/%.prg: $(OBJ)/test_%.o $$(addprefix $(OBJ)/,$$(addsuffix .o,$$(UNIT_$$*))) tests/sim65.cfg
	$(MKDIR) $(TEST)
	cl65 -t sim6502 -C tests/sim65.cfg -o $@ $(filter %.o,$^)

$(TEST)/%.ok: $(TEST)/%.prg
	sim65 $<
	$(TOUCH) $@

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

# chat with the integer engine on the PC
chat:
	$(PYTHON) python/chat.py

# - CLEAN --------------------------------------------------------------------
clean:
	$(RMDIR) build

#-----------------------------------------------------------------------------
