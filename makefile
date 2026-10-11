#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------

# main make targets
.PHONY: all prg corpus train qat compile model evaluate chat test pytest unit system e2e clean

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
all: prg test

# the C64 program and its test build (questions from a file, for tests/e2e.py)
prg: build/c64/chatcbm.prg build/c64/chatcbm-test.prg

# files generated from the model, the tokenizer and the Python engine
$(GEN)/stamp: models/model_int.npz models/tokenizer.json python/export.py python/vectors.py               python/chatcbm/intengine.py python/chatcbm/tokenizer.py
	$(PYTHON) python/export.py --out $(GEN)
	$(PYTHON) python/vectors.py --out $(GEN)
	$(TOUCH) $@

# 6502 modules: one object per source file
$(OBJ)/%.o: src/%.s $(GEN)/stamp
	$(MKDIR) $(OBJ)
	ca65 -I $(GEN) --bin-include-dir $(GEN) -o $@ $<

# the engine: every module of src/ except the machine-dependent main programs
ENGINE = chat tokenizer model sampler matmul linear log8 layernorm gelu residual logits          attention prim buffers
ENGINE_OBJ = $(ENGINE:%=$(OBJ)/%.o)

# sim65: chat on stdin/stdout, or the file test of tests/e2e.py
build/sim65/chatcbm.prg: $(OBJ)/sim65_main.o $(ENGINE_OBJ) src/sim65/sim65.cfg
	$(MKDIR) build/sim65
	cl65 -t sim6502 -C src/sim65/sim65.cfg -o $@ $(filter %.o,$^)

$(OBJ)/sim65_main.o: src/sim65/main.c
	$(MKDIR) $(OBJ)
	cl65 -t sim6502 -c -O -o $@ $<

# C64: the main program comes first, its zero page must stay below the kernal's
build/c64/chatcbm.prg: $(OBJ)/c64_main.o $(ENGINE_OBJ) src/c64/c64.cfg
	$(MKDIR) build/c64
	ld65 -C src/c64/c64.cfg -m build/c64/chatcbm.map -o $@ $(filter %.o,$^)

build/c64/chatcbm-test.prg: $(OBJ)/c64_test.o $(ENGINE_OBJ) src/c64/c64.cfg
	$(MKDIR) build/c64
	ld65 -C src/c64/c64.cfg -o $@ $(filter %.o,$^)

$(OBJ)/c64_main.o: src/c64/main.s
	$(MKDIR) $(OBJ)
	ca65 -o $@ $<

$(OBJ)/c64_test.o: src/c64/main.s
	$(MKDIR) $(OBJ)
	ca65 -D C64_TEST=1 -o $@ $<

# helpers of the unit tests written in assembler
$(OBJ)/%.o: tests/unit/%.s
	$(MKDIR) $(OBJ)
	ca65 -o $@ $<

# - TESTS --------------------------------------------------------------------
test: pytest unit system

# Python: hand-written backward pass, tokenizer, integer engine
pytest:
	$(PYTHON) -m unittest discover -s python/tests -t python

# end to end on sim65: the whole program answers as the Python engine
system: build/sim65/chatcbm.prg
	$(PYTHON) tests/e2e.py sim65

# end to end on the C64, in VICE (x64sc in PATH): not in the CI, Ubuntu's VICE has no ROMs
e2e: build/c64/chatcbm-test.prg
	$(PYTHON) tests/e2e.py c64

# 6502 unit tests on sim65: tests/unit/test_<name>.c against the vectors of the
# Python engine, linked with the modules listed in UNIT_<name>
UNIT_TESTS = prim matmul log8 linear layernorm gelu residual logits attention model tokenizer sampler
UNIT_prim = prim
UNIT_matmul = matmul buffers aligned
UNIT_log8 = log8 prim buffers
UNIT_linear = linear matmul prim buffers
UNIT_layernorm = layernorm prim buffers
UNIT_gelu = gelu prim buffers
UNIT_residual = residual prim buffers
UNIT_logits = logits prim buffers
UNIT_attention = attention prim
UNIT_model = model sampler matmul linear log8 layernorm gelu residual logits attention prim buffers
UNIT_tokenizer = tokenizer
UNIT_sampler = sampler buffers

unit: $(UNIT_TESTS:%=$(TEST)/%.ok)

$(OBJ)/test_%.o: tests/unit/test_%.c tests/unit/check.h $(GEN)/stamp
	$(MKDIR) $(OBJ)
	cl65 -t sim6502 -c -O -I $(GEN) -o $@ $<

.SECONDEXPANSION:
$(TEST)/%.prg: $(OBJ)/test_%.o $$(addprefix $(OBJ)/,$$(addsuffix .o,$$(UNIT_$$*))) src/sim65/sim65.cfg
	$(MKDIR) $(TEST)
	cl65 -t sim6502 -C src/sim65/sim65.cfg -o $@ $(filter %.o,$^)

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
