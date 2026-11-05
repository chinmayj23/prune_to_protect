#!/usr/bin/env python
"""Train the CIFAR-10 memorization bank and score it (paper Appendix A.2).

Runs the engine's ``run_loo``: 400 ResNet-20 models per method, each on an independent random 70%
subset, then ``compute_loo_scores`` at each tracked epoch. The subsets are shared across methods and
stored in ``PRUNEPRIV_DATA``. Output under ``results/studies/cifar10_loo/<method>/``:

    run_<i>/epoch_<e>.pt
    memorization_scores/loo_scores_epoch_<e>.pt

Split the work with ``--start-run``/``--end-run`` and ``--no-score``, then score once with
``--score-only``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prunepriv import paths
from prunepriv.configs import build_config, canonical_method
from prunepriv.memorization import compute_loo_scores
from prunepriv.training.engine import make_subset_path, run_loo

DATASET = "cifar10"
MODEL = "resnet20"
TRACKED_EPOCHS = [10, 18, 25, 50, 57, 73, 75, 87, 88, 99]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", nargs="+", default=["standard", "infobatch", "wlib"])
    ap.add_argument("--num-runs", type=int, default=400)
    ap.add_argument("--start-run", type=int, default=0)
    ap.add_argument("--end-run", type=int, default=None, help="default: --num-runs")
    ap.add_argument("--no-score", action="store_true", help="train only")
    ap.add_argument("--score-only", action="store_true", help="skip training")
    args = ap.parse_args()

    for name in args.methods:
        method = canonical_method(name)
        method_dir = paths.ensure(paths.results_dir("studies", "cifar10_loo", method))

        cfg = build_config(DATASET, MODEL, method)
        cfg.is_loo = True
        cfg.num_subsets = args.num_runs
        cfg.save_outputs = True
        cfg.save_epoch_checkpoints = True
        cfg.sample_at_epoch = list(TRACKED_EPOCHS)
        cfg.loo_output_path = str(method_dir)

        if not args.score_only:
            run_loo(cfg, start_run=args.start_run, end_run=args.end_run)
        if not args.no_score:
            compute_loo_scores(method_dir, num_runs=args.num_runs, epochs=TRACKED_EPOCHS,
                               subset_file=make_subset_path(cfg), dataset=DATASET, model=MODEL,
                               out_dir=method_dir / "memorization_scores")


if __name__ == "__main__":
    main()
