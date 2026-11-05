"""Memorization scores.

``compute_loo_scores`` is the subset leave-one-out estimator (mean-correct-in minus mean-correct-out).
"""

from __future__ import annotations

from .loo import compute_loo_scores

__all__ = ["compute_loo_scores"]
