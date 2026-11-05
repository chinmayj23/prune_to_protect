#!/usr/bin/env python
"""Stage experiment outputs into the layout the figure code reads.

Run after ``train_full_models.py``, ``cifar10_loo.py`` and the defense-unaware ``scripts/attack.py``
runs. Writes to the analysis root (``PRUNEPRIV_ANALYSIS_ROOT``, default ``results/analysis``):

    cifar10/subsets_cifar_400.pt
    cifar10/<method dir>/seed_avg.pt
    cifar10/<method dir>/cifar_evolution_log_<method>.pt
    cifar10/<method dir>/memorization_scores/loo_scores_epoch_<e>.pt
    cifar10/<method dir>/derived/{rmia_overall_summary,rmia_per_target_metrics,
                                  per_sample_privacy_profile}_<method>.*
    cifar100_wrn28_2/<method dir>/seed_avg.pt

where <method dir> is ``standard``, ``IB`` or ``IB-WL``. A missing input is reported and skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from prunepriv import paths

METHOD_DIRS = {"standard": "standard", "infobatch": "IB", "ib_prw": "IB-WL"}
PROFILE_FPRS = ((0.01, "1pct"), (0.001, "0p1pct"), (0.0001, "0p01pct"))
TPR_FIELDS = ((0.01, "1pct"), (0.001, "0p1pct"), (0.0001, "0p01pct"), (0.00001, "0p001pct"), (0.0, "0pct"))


def copy(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[missing] {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_rmia_artifacts(method: str, report_dir: Path, keep: np.ndarray, labels: np.ndarray,
                         out: Path) -> None:
    """Per-target metrics, their summary, and the per-sample member-side privacy profile."""
    out.mkdir(parents=True, exist_ok=True)
    num_targets, num_samples = keep.shape
    member_counts = keep.sum(axis=0)
    score_sum = np.zeros(num_samples)
    score_sq_sum = np.zeros(num_samples)
    hits = {label: np.zeros(num_samples, dtype=np.int32) for _, label in PROFILE_FPRS}
    rows = []

    for i in range(num_targets):
        preds = np.load(report_dir / f"attack_predictions_{i}.npz", allow_pickle=True)
        stats = np.load(report_dir / f"attack_stats_{i}.npz", allow_pickle=True)
        membership = keep[i]
        if not np.array_equal(preds["answers"].astype(bool), membership):
            raise RuntimeError(f"Membership mismatch for target {i}: sample order is inconsistent.")

        scores = -preds["scores"].astype(np.float64)  # higher = more member-like
        score_sum[membership] += scores[membership]
        score_sq_sum[membership] += scores[membership] ** 2
        thresholds = stats["thresholds_dict"].item()
        for fpr, label in PROFILE_FPRS:
            hits[label][membership] += (scores >= thresholds[fpr])[membership]

        tpr = stats["tpr_dict"].item()
        row = {"target_model_index": i, "auc": float(stats["all_aucs"]),
               "accuracy": float(stats["all_accs"])}
        row.update({f"tpr_at_{label}_fpr": float(tpr[fpr]) for fpr, label in TPR_FIELDS})
        rows.append(row)

    np.savez(out / f"rmia_per_target_metrics_{method}.npz",
             **{key: np.array([r[key] for r in rows]) for key in rows[0]})

    summary = {"target_training_method": method, "num_target_models": num_targets}
    for key in rows[0]:
        if key != "target_model_index":
            values = np.array([r[key] for r in rows], dtype=np.float64)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std(ddof=1)) if num_targets > 1 else 0.0
    (out / f"rmia_overall_summary_{method}.json").write_text(json.dumps(summary, indent=2),
                                                             encoding="utf-8")

    mean = score_sum / member_counts
    std = np.sqrt(np.maximum(score_sq_sum / member_counts - mean ** 2, 0.0))
    profile = {
        "sample_index": np.arange(num_samples, dtype=np.int32),
        "class_label": labels.astype(np.int32),
        "member_target_count": member_counts.astype(np.int32),
        "mean_member_score": mean.astype(np.float32),
        "std_member_score": std.astype(np.float32),
    }
    for _, label in PROFILE_FPRS:
        profile[f"hit_rate_at_{label}_fpr"] = (hits[label] / member_counts).astype(np.float32)
    np.savez(out / f"per_sample_privacy_profile_{method}.npz", **profile)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evolution-seed", type=int, default=3407)
    ap.add_argument("--num-shadows", type=int, default=64)
    ap.add_argument("--num-targets", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None, help="default: PRUNEPRIV_ANALYSIS_ROOT")
    args = ap.parse_args()

    root = args.out or Path(os.environ.get("PRUNEPRIV_ANALYSIS_ROOT", paths.results_dir("analysis")))
    full_c10 = paths.results_dir("full_models", "cifar10_resnet20")
    full_c100 = paths.results_dir("full_models", "cifar100_wrn28_2")
    loo = paths.results_dir("studies", "cifar10_loo")
    data = torch.load(paths.data_dir("cifar10", "cifar10_train.pt"), map_location="cpu",
                      weights_only=False)
    labels = np.asarray(data["labels"]).ravel()

    copy(paths.data_dir("subsets_cifar_400.pt"), root / "cifar10" / "subsets_cifar_400.pt")
    for method, method_dir in METHOD_DIRS.items():
        dst = root / "cifar10" / method_dir
        copy(full_c10 / method / "seed_avg.pt", dst / "seed_avg.pt")
        evolution = f"cifar_evolution_log_{'ib' if method == 'infobatch' else method}.pt"
        copy(full_c10 / method / f"seed_{args.evolution_seed}" / evolution, dst / evolution)
        for score in sorted((loo / method / "memorization_scores").glob("loo_scores_epoch_*.pt")):
            copy(score, dst / "memorization_scores" / score.name)

        tag = f"cifar10_resnet20_{method}_vs_standard_rmia_online_unaware"
        report = paths.results_dir("attacks", tag, "report_rmia_online")
        keep_file = paths.banks_dir(f"cifar10_resnet20_{method}_targets_{args.num_targets}",
                                    "target_keep_matrix.npy")
        if report.exists() and keep_file.exists():
            write_rmia_artifacts(method, report, np.load(keep_file).astype(bool), labels,
                                 dst / "derived")
        else:
            print(f"[missing] {report} or {keep_file}")

        copy(full_c100 / method / "seed_avg.pt", root / "cifar100_wrn28_2" / method_dir / "seed_avg.pt")

    print(f"Staged analysis inputs under {root}")


if __name__ == "__main__":
    main()
