"""CLI: full-dataset training + test accuracy.

    prunepriv-train --dataset cifar10 --model resnet20 --method wlib --seeds 3407
    python scripts/train.py --dataset cifar10 --method standard --seeds 0 1 42 1234 3407

Method/model/dataset are arguments — no source edits. Results are written as JSON under
PRUNEPRIV_RESULTS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..configs import available_datasets, available_methods, get_dataset_spec
from .. import paths
from ..training import train_full


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-dataset training + test accuracy.")
    p.add_argument("--dataset", required=True, choices=available_datasets())
    p.add_argument("--model", default=None, help="default: the dataset's default model")
    p.add_argument("--method", default="standard",
                   help=f"one of: {', '.join(available_methods())} (aliases: wlib=ib_prw, ib=infobatch)")
    p.add_argument("--seeds", nargs="+", type=int, default=[3407])
    p.add_argument("--epochs", type=int, default=None, help="override num_epochs (default: config)")
    p.add_argument("--no-compile", action="store_true", help="disable torch.compile (faster startup)")
    p.add_argument("--out", default=None, help="results JSON path (default: PRUNEPRIV_RESULTS/train/<tag>.json)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    spec = get_dataset_spec(args.dataset)
    model = args.model or spec.default_model

    runs = []
    for seed in args.seeds:
        print(f"\n==== {args.dataset}/{model}/{args.method} seed {seed} ====")
        runs.append(train_full(
            args.dataset, model=model, method=args.method, seed=seed,
            epochs=args.epochs, use_compile=(False if args.no_compile else None),
        ))

    finals = np.array([r["final_test_accuracy"] for r in runs], dtype=np.float64)
    summary = {
        "dataset": args.dataset, "model": model, "method": args.method,
        "seeds": args.seeds, "num_seeds": len(runs),
        "final_test_accuracy_mean": float(finals.mean()),
        "final_test_accuracy_std": float(finals.std(ddof=1)) if finals.size > 1 else 0.0,
        "per_seed": runs,
    }

    tag = f"{args.dataset}_{model}_{args.method}"
    out_path = Path(args.out) if args.out else (paths.ensure(paths.results_dir("train")) / f"{tag}.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[{tag}] final test acc over {len(runs)} seed(s): "
          f"{summary['final_test_accuracy_mean']:.2f}% +/- {summary['final_test_accuracy_std']:.2f}%")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
