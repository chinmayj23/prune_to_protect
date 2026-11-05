"""Training engine + high-level API.

The engine (``engine.py``) dispatches method (standard / infobatch / wlib / ...) and model
(r20/r18/r50/...) off the ExptConfig.
``train_full`` is the high-level entry for full-dataset training + test accuracy.
"""

from __future__ import annotations

from .engine import run_single_training, set_seed
from .api import train_full

__all__ = ["train_full", "run_single_training", "set_seed"]
