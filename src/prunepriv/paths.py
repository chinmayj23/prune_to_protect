"""Single source of truth for filesystem locations.

Every path in the repo is derived from here. Override any location with an environment variable;
otherwise everything lives under the repo root, so a fresh clone works with no configuration.

Env vars:
    PRUNEPRIV_DATA     datasets + fast-loader blobs + betons   (default: <repo>/data)
    PRUNEPRIV_RESULTS  training checkpoints, scores, figures   (default: <repo>/results)
    PRUNEPRIV_BANKS    RMIA/LiRA shadow/target model banks     (default: <repo>/banks)
"""

from __future__ import annotations

import os
from pathlib import Path

# <repo>/src/prunepriv/paths.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(env_var: str, default_subdir: str) -> Path:
    raw = os.environ.get(env_var)
    return Path(raw).expanduser().resolve() if raw else (REPO_ROOT / default_subdir)


DATA_DIR = _resolve("PRUNEPRIV_DATA", "data")
RESULTS_DIR = _resolve("PRUNEPRIV_RESULTS", "results")
BANKS_DIR = _resolve("PRUNEPRIV_BANKS", "banks")


def data_dir(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def results_dir(*parts: str) -> Path:
    return RESULTS_DIR.joinpath(*parts)


def banks_dir(*parts: str) -> Path:
    return BANKS_DIR.joinpath(*parts)


def ensure(path: Path) -> Path:
    """Create ``path`` (as a directory) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "REPO_ROOT", "DATA_DIR", "RESULTS_DIR", "BANKS_DIR",
    "data_dir", "results_dir", "banks_dir", "ensure",
]
