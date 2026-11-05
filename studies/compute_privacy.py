#!/usr/bin/env python
"""RMIA/LiRA privacy table from the per-target predictions written by scripts/attack.py.

Computes AUC + TPR@{1%, 0.1%, 0.01%, 0%} per target, then mean ± std over the target bank, for every
(dataset, method, threat, attack) result dir. Writes results/studies/privacy/privacy_metrics.json.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from prunepriv import paths

RESULT_ROOT = paths.results_dir("attacks")
MODEL = {"cifar10": "resnet20", "cifar100": "wrn28_2"}
OUT = paths.ensure(paths.results_dir("studies", "privacy"))
FPRS = [0.01, 0.001, 0.0001, 0.0]


def parse(name, dataset):
    """Parse <dataset>_<model>_<target>_vs_<shadow>_<attack>_<online|offline>_<aware|unaware>."""
    pre = f"{dataset}_{MODEL[dataset]}_"
    match = re.match(r"^(.+)_vs_(.+)_(rmia|lira)_(online|offline)_(aware|unaware)$", name)
    if not name.startswith(pre) or match is None:
        return None
    method, shadow, attack, mode, threat = match.groups()
    return dict(method=method[len(pre):], shadow=shadow, attack=attack, threat=threat,
                online=mode == "online")


def load_per_target(report_dir):
    preds = []
    for f in sorted(glob.glob(os.path.join(report_dir, "attack_predictions_*.npz"))):
        z = np.load(f)
        preds.append((z["scores"].ravel(), z["answers"].ravel()))
    return preds


def _one(scores, answers):
    y = answers.astype(int)
    s = -scores  # main.py: lower score = member -> higher -s = more member
    auc = float(roc_auc_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)
    tprs = []
    for f in FPRS:
        mask = fpr <= f
        tprs.append(float(tpr[mask].max()) if mask.any() else 0.0)
    return auc, tprs


def metrics(preds):
    """Paper aggregation: per-target metrics, then mean +/- sample std over the 20-target bank."""
    aucs, tpr_mat = [], []
    for s, a in preds:
        auc, tprs = _one(s, a)
        aucs.append(auc)
        tpr_mat.append(tprs)
    aucs = np.array(aucs)
    tpr_mat = np.array(tpr_mat)  # (n_targets, n_fprs)
    n = len(aucs)
    ddof = 1 if n > 1 else 0
    mean = {"auc": float(aucs.mean())}
    std = {"auc": float(aucs.std(ddof=ddof))}
    for i, f in enumerate(FPRS):
        mean[f] = float(tpr_mat[:, i].mean())
        std[f] = float(tpr_mat[:, i].std(ddof=ddof))
    return mean, std, n


def main():
    rows = []
    for dataset in MODEL:
        for d in sorted(RESULT_ROOT.glob(f"{dataset}_{MODEL[dataset]}_*")):
            if not d.is_dir():
                continue
            meta = parse(d.name, dataset)
            if meta is None:
                continue
            reports = [p for p in d.glob("report_*") if p.is_dir()]
            if not reports:
                continue
            preds = load_per_target(str(reports[0]))
            if not preds:
                continue
            mean, std, ntar = metrics(preds)
            thr = f"{meta['threat']}{'' if meta['online'] else '/offline'}"
            rows.append({
                "dataset": dataset, "model": MODEL[dataset], **meta, "n_targets": ntar,
                "dir": d.name,
                "auc": mean["auc"], "auc_std": std["auc"],
                "tpr@1%": mean[0.01], "tpr@1%_std": std[0.01],
                "tpr@0.1%": mean[0.001], "tpr@0.1%_std": std[0.001],
                "tpr@0.01%": mean[0.0001], "tpr@0.01%_std": std[0.0001],
                "tpr@0%": mean[0.0], "tpr@0%_std": std[0.0],
            })
            print(f"{dataset:8s} {meta['method']:22s} {meta['attack']:4s} {thr:14s} "
                  f"AUC={mean['auc']:.4f}±{std['auc']:.3f} "
                  f"TPR@1%={100*mean[0.01]:.2f} @0.1%={100*mean[0.001]:.2f} "
                  f"@0.01%={100*mean[0.0001]:.2f} @0%={100*mean[0.0]:.2f} (n={ntar})", flush=True)
    (OUT / "privacy_metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {OUT/'privacy_metrics.json'} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
