#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Model hyperparameters."""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    vocab_size: int = 256  # BPE tokens (words or word pieces + punctuation + specials): one byte each
    context: int = 64      # maximum context window, in tokens
    d_model: int = 32      # embedding size
    n_layer: int = 4       # transformer blocks
    n_head: int = 4        # attention heads (32 / 4 = 8 dimensions per head)
    d_ff: int = 32         # FFN inner size, no 4x expansion

    def to_dict(self):
        return asdict(self)

#-----------------------------------------------------------------------------
