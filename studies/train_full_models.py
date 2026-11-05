#!/usr/bin/env python
"""Train the full-data models that the group analyses and figures read.

Runs the engine's ``run_standard_or_ib`` for each method. Output under
``results/full_models/<dataset>_<model>/<method>/``:

    seed_avg.pt                                accuracy curves and pruning rates over seeds
    seed_<s>/last_epoch.pt                     final model
    seed_<s>/step_<N>.pt                       models at matched optimizer steps (--steps)
    seed_<s>/prune_rate.npy                    per-sample cumulative pruning rate (WLIB only)
    seed_<s>/cifar_evolution_log_<method>.pt   per-sample loss and entropy (--evolution-seed only)

    python studies/train_full_models.py --dataset cifar10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prunepriv import paths
from prunepriv.configs import build_config, canonical_method, get_dataset_spec
from prunepriv.training.engine import run_standard_or_ib

STEPS = [1000, 2000, 3000, 4000, 5000, 6300]
TRACKED_EPOCHS = [10, 18, 25, 50, 57, 73, 75, 87, 88, 99]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--model", default=None, help="default: the dataset's default model")
    ap.add_argument("--methods", nargs="+", default=["standard", "infobatch", "wlib"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42, 1234, 3407])
    ap.add_argument("--evolution-seed", type=int, default=3407,
                    help="seed that records the per-sample evolution log")
    ap.add_argument("--steps", nargs="*", type=int, default=STEPS,
                    help="optimizer steps at which to save a checkpoint")
    args = ap.parse_args()

    model = args.model or get_dataset_spec(args.dataset).default_model
    root = paths.ensure(paths.results_dir("full_models", f"{args.dataset}_{model}"))

    for name in args.methods:
        method = canonical_method(name)
        cfg = build_config(args.dataset, model, method)
        cfg.seed_list = list(args.seeds)
        cfg.experiment_name = method
        cfg.non_loo_root = str(root)
        cfg.save_outputs = True
        cfg.save_last_checkpoint = True
        cfg.save_at_steps = list(args.steps)
        cfg.sample_at_epoch = list(TRACKED_EPOCHS)
        cfg.log_sample_evolution = True
        cfg.evolution_seed = args.evolution_seed
        cfg.log_evolution_activations = False
        cfg.log_evolution_grad_norms = False
        run_standard_or_ib(cfg)


if __name__ == "__main__":
    main()
