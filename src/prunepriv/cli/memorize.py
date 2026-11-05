"""CLI: subset-LOO memorization scores.

    python scripts/memorize.py --dataset cifar10 --model resnet20 \
        --runs-dir <LOO_experiment_dir> --num-runs 400 --subset-file <subsets.pt> \
        --epochs 99 --out <out_dir>

Each LOO run dir must contain run_<i>/epoch_<e>.pt. Non-LOO memorization = multi-seed full-data
training: use `scripts/train.py --seeds ...`.
"""

from __future__ import annotations

import argparse

from ..configs import available_datasets
from ..memorization import compute_loo_scores


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Subset-LOO memorization scores.")
    p.add_argument("--dataset", required=True, choices=available_datasets())
    p.add_argument("--model", default=None)
    p.add_argument("--runs-dir", required=True, help="dir containing run_<i>/epoch_<e>.pt")
    p.add_argument("--num-runs", type=int, required=True)
    p.add_argument("--subset-file", required=True, help="subset-membership cache (.pt)")
    p.add_argument("--epochs", nargs="+", type=int, required=True)
    p.add_argument("--out", default=None, help="output dir for loo_scores_epoch_<e>.pt")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    res = compute_loo_scores(
        args.runs_dir, num_runs=args.num_runs, epochs=args.epochs,
        subset_file=args.subset_file, dataset=args.dataset, model=args.model, out_dir=args.out,
    )
    for e, r in res.items():
        m = r["memorization"]
        print(f"epoch {e}: n={len(m)} mean={m.mean():.4f} std={m.std():.4f} "
              f"memorized(>0.1)={int((m > 0.1).sum())}")
    if args.out:
        print(f"Wrote loo_scores_epoch_*.pt to {args.out}")


if __name__ == "__main__":
    main()
