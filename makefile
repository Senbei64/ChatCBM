#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------

# main make targets
.PHONY: all clean

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
all:

# - CLEAN --------------------------------------------------------------------
clean:
	$(RMDIR) build

#-----------------------------------------------------------------------------
