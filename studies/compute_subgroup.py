#!/usr/bin/env python
"""Per-class and hard-group (rho-tier) RMIA on CIFAR-10 (per-class AUC of Table 6).

CPU-only. Reads the per-target predictions from scripts/attack.py, the class labels, and the WLIB
cumulative pruning rates from studies/train_full_models.py. Metrics are computed per target and
averaged over the target bank. Writes results/studies/subgroup/subgroup_privacy.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from prunepriv import paths

RES = paths.results_dir("attacks")
FULLMODELS = paths.results_dir("full_models", "cifar10_resnet20")
LABELS_PT = paths.data_dir("cifar10", "cifar10_train.pt")
OUT = paths.ensure(paths.results_dir("studies", "subgroup"))
FPRS = [0.01, 0.001, 0.0001, 0.0]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# (label, result dir)
CONFIGS = {
    "standard/unaware": "cifar10_resnet20_standard_vs_standard_rmia_online_unaware",
    "infobatch/unaware": "cifar10_resnet20_infobatch_vs_standard_rmia_online_unaware",
    "infobatch/aware": "cifar10_resnet20_infobatch_vs_infobatch_rmia_online_aware",
    "wlib/unaware": "cifar10_resnet20_ib_prw_vs_standard_rmia_online_unaware",
    "wlib/aware": "cifar10_resnet20_ib_prw_vs_ib_prw_rmia_online_aware",
}


def load_labels():
    d = torch.load(LABELS_PT, map_location="cpu", weights_only=False)
    y = d["labels"] if isinstance(d, dict) else d
    return np.asarray(y).ravel()


def load_rho_tiers():
    files = sorted(glob.glob(str(FULLMODELS / "ib_prw" / "seed_*" / "prune_rate.npy")))
    rhos = [np.load(f).ravel() for f in files]
    rho = np.mean(np.stack(rhos), axis=0)
    edges = np.linspace(rho.min(), rho.max(), 4)   # three equal-width bins, as in the paper
    tier = np.full(len(rho), 1, np.int8)           # medium
    tier[rho <= edges[1]] = 0                      # hard (lowest rho = down-weighted)
    tier[rho > edges[2]] = 2                       # easy
    return tier


def per_target(report_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(report_dir, "attack_predictions_*.npz"))):
        z = np.load(f)
        out.append((z["scores"].ravel(), z["answers"].ravel()))
    return out


def auc_of(scores, answers):
    y = answers.astype(int)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, -scores)


def tprs_of(scores, answers):
    y = answers.astype(int)
    if y.min() == y.max():
        return [np.nan] * len(FPRS)
    fpr, tpr, _ = roc_curve(y, -scores)
    return [float(tpr[fpr <= f].max()) if (fpr <= f).any() else 0.0 for f in FPRS]


def main():
    labels = load_labels()
    tiers = load_rho_tiers()
    N = len(labels)
    result = {}
    for name, dirname in CONFIGS.items():
        d = RES / dirname
        reports = [p for p in d.glob("report_*") if p.is_dir()] if d.exists() else []
        if not reports:
            print(f"[skip] {name}: no predictions")
            continue
        preds = per_target(str(reports[0]))
        # per-class AUC: per target, per class -> average over targets
        cls_auc = np.full((len(preds), 10), np.nan)
        for ti, (s, a) in enumerate(preds):
            for c in range(10):
                m = labels == c
                cls_auc[ti, c] = auc_of(s[m], a[m])
        per_class = np.nanmean(cls_auc, axis=0)
        # hard-group tiers: per target, per tier -> average
        tier_auc = np.full((len(preds), 3), np.nan)
        tier_tpr = np.full((len(preds), 3, len(FPRS)), np.nan)
        for ti, (s, a) in enumerate(preds):
            for t in range(3):
                m = tiers == t
                tier_auc[ti, t] = auc_of(s[m], a[m])
                tier_tpr[ti, t] = tprs_of(s[m], a[m])
        result[name] = {
            "per_class_auc": {CLASSES[c]: round(float(per_class[c]), 4) for c in range(10)},
            "worst_class": CLASSES[int(np.nanargmax(per_class))],
            "worst_class_auc": round(float(np.nanmax(per_class)), 4),
        }
        # readable tier table
        tnames = ["hard", "medium", "easy"]
        result[name]["tiers"] = {
            tnames[t]: {
                "auc": round(float(np.nanmean(tier_auc[:, t])), 4),
                "tpr@1%": round(float(np.nanmean(tier_tpr[:, t, 0])), 4),
                "tpr@0.1%": round(float(np.nanmean(tier_tpr[:, t, 1])), 4),
                "tpr@0.01%": round(float(np.nanmean(tier_tpr[:, t, 2])), 4),
            } for t in range(3)
        }
        print(f"{name}: worst-class {result[name]['worst_class']} "
              f"AUC={result[name]['worst_class_auc']:.4f} | "
              f"hard-tier AUC={result[name]['tiers']['hard']['auc']:.4f} "
              f"TPR@1%={result[name]['tiers']['hard']['tpr@1%']:.4f}", flush=True)
    (OUT / "subgroup_privacy.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT/'subgroup_privacy.json'}")


if __name__ == "__main__":
    main()
