#!/usr/bin/env python
"""Full-dataset accuracy sweep: the paper's methods and ablations, 5 seeds, CIFAR-10 + CIFAR-100.

Resumable: skips a method if its JSON already exists. Writes one JSON per (dataset, model, method)
under results/studies/accuracy/.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
from prunepriv.training import train_full  # noqa: E402
from prunepriv import paths  # noqa: E402

SEEDS = [0, 1, 42, 1234, 3407]
OUT = paths.results_dir("studies", "accuracy")
OUT.mkdir(parents=True, exist_ok=True)

JOBS = []
for m in ["standard", "infobatch", "ib_prw", "wlib_loss", "wlib_shuffle", "wlib_hardconst",
          "ib_noanneal", "ib_earlystop_cooled"]:
    JOBS.append(("cifar10", "resnet20", m, m, {}))
for m in ["standard", "infobatch", "ib_prw"]:
    JOBS.append(("cifar100", "wrn28_2", m, m, {}))


def log(msg):
    print(msg, flush=True)


def main():
    for dataset, model, method, tag, extra in JOBS:
        out = OUT / f"{dataset}_{model}_{tag}.json"
        if out.exists():
            log(f"[skip] {out.name} already exists")
            continue
        runs = []
        for seed in SEEDS:
            log(f"\n===== {dataset}/{model}/{tag} seed {seed} =====")
            try:
                r = train_full(dataset, model=model, method=method, seed=seed, **extra)
            except Exception as e:  # keep going; record the failure
                log(f"[ERROR] {tag} seed {seed}: {e}")
                traceback.print_exc()
                r = {"seed": seed, "final_test_accuracy": float("nan"), "error": str(e)}
            runs.append(r)
            out.write_text(json.dumps({"dataset": dataset, "model": model, "method": method,
                                       "tag": tag, "seeds": SEEDS, "per_seed": runs}, indent=2))
        finals = np.array([x.get("final_test_accuracy", float("nan")) for x in runs], float)
        valid = finals[~np.isnan(finals)]
        summary = {
            "dataset": dataset, "model": model, "method": method, "tag": tag, "seeds": SEEDS,
            "per_seed": runs,
            "mean": float(valid.mean()) if valid.size else float("nan"),
            "std": float(valid.std(ddof=1)) if valid.size > 1 else 0.0,
            "n_valid": int(valid.size),
        }
        out.write_text(json.dumps(summary, indent=2))
        log(f"[done] {tag}: {summary['mean']:.2f} +/- {summary['std']:.2f} (n={valid.size})")
    log("\nACCURACY SWEEP COMPLETE")


if __name__ == "__main__":
    main()
