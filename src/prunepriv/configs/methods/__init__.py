"""Method registry.

A method selects training behaviour (standard / infobatch / wlib=ib_prw / the paper's ablations) as
a config over the base ExptConfig. The single source of truth is ``build.py``:
``build_method_config(spec, model, method, data_root, **overrides)`` returns the configured
ExptConfig for a method chosen by string.

To add a method, extend ``VERIFIED_METHODS`` and the dispatch in ``build.py``.
"""

from __future__ import annotations

from .build import (
    build_method_config,
    canonical_method,
    VERIFIED_METHODS,
    TRAIN_DEFAULTS,
)


def available_methods() -> list[str]:
    return sorted(VERIFIED_METHODS)


__all__ = [
    "build_method_config", "canonical_method", "available_methods",
    "VERIFIED_METHODS", "TRAIN_DEFAULTS",
]
