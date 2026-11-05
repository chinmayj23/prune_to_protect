"""Membership-inference attacks.

``attacks/rmia/`` holds the attack (main.py + signals.py + plot.py). ``runner.py`` orchestrates the
end-to-end pipeline (train banks -> export logits -> attack).
"""

from __future__ import annotations

from .runner import run_attack_pipeline

__all__ = ["run_attack_pipeline"]
