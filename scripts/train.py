#!/usr/bin/env python
"""Thin wrapper so `python scripts/train.py ...` works without installing the package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prunepriv.cli.train import main  # noqa: E402

if __name__ == "__main__":
    main()
