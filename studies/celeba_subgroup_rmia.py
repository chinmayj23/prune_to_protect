#!/usr/bin/env python
"""Per-GROUP RMIA privacy for CelebA (defense-unaware) — AUC + TPR@{1%,0.01%,0%}, logged to JSON.

Joins each target model's per-sample RMIA (scores, answers=membership) with the CelebA group labels
(Blond_Hair x Male), reporting unadjusted (overall) + adjusted (train-weighted) + per-group + the
blonde-male worst-group, averaged over the target bank. Writes subgroup_rmia_celeba.json for tables.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from prunepriv import paths

GROUP = ["non-blonde/female", "non-blonde/male", "blonde/female", "blonde/male"]
WORST = 3                                   # blonde/male = the rare worst-group / minority
W = np.array([71629, 66874, 22880, 1387], float); W /= W.sum()   # train proportions (adjusted weights)
FPRS = [0.01, 0.0001, 0.0]                  # TPR@1%, TPR@0.01%, TPR@0%
METHODS = ["standard", "infobatch", "ib_prw"]


def per_target(report_dir):
    out = []
    for f in sorted(glob.glob(str(Path(report_dir) / "attack_predictions_*.npz"))):
        z = np.load(f)
        out.append((z["scores"].ravel(), z["answers"].ravel()))
    return out


def auc_of(s, a):
    y = a.astype(int)
    return np.nan if y.min() == y.max() else roc_auc_score(y, -s)


def tprs_of(s, a):
    y = a.astype(int)
    if y.min() == y.max():
        return [np.nan] * len(FPRS)
    fpr, tpr, _ = roc_curve(y, -s)
    return [float(tpr[fpr <= f].max()) if (fpr <= f).any() else 0.0 for f in FPRS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet10t")
    ap.add_argument("--temperature", default="", help="tag suffix if a non-default temp report dir was used")
    args = ap.parse_args()
    groups = torch.load(paths.data_dir("celeba", "celeba_train.pt"),
                        map_location="cpu", weights_only=False)["groups"].numpy()
    result = {}
    for method in METHODS:
        tag = f"celeba_{args.model}_{method}_vs_standard_rmia_online_unaware"
        d = paths.results_dir("attacks", tag)
        reports = [p for p in d.glob("report_*") if p.is_dir()] if d.exists() else []
        preds = per_target(reports[0]) if reports else per_target(d)
        if not preds:
            print(f"[skip] {method}: no attack_predictions under {d}"); continue
        ov_auc = [auc_of(s, a) for s, a in preds]
        ov_tpr = np.array([tprs_of(s, a) for s, a in preds])
        g_auc = np.full((len(preds), 4), np.nan)
        g_tpr = np.full((len(preds), 4, len(FPRS)), np.nan)
        for ti, (s, a) in enumerate(preds):
            for g in range(4):
                m = groups == g
                g_auc[ti, g] = auc_of(s[m], a[m]); g_tpr[ti, g] = tprs_of(s[m], a[m])
        per_g = np.nanmean(g_auc, 0); per_g_tpr = np.nanmean(g_tpr, 0)

        def col(overall, groups_arr, worst_val):
            return {"unadjusted": round(float(overall), 4),
                    "adjusted": round(float((groups_arr * W).sum()), 4),
                    **{GROUP[g]: round(float(groups_arr[g]), 4) for g in range(4)},
                    "blonde_male": round(float(worst_val), 4)}

        result[method] = {
            "AUC": col(np.nanmean(ov_auc), per_g, per_g[WORST]),
            "TPR@1%": col(np.nanmean(ov_tpr[:, 0]), per_g_tpr[:, 0], per_g_tpr[WORST, 0]),
            "TPR@0.01%": col(np.nanmean(ov_tpr[:, 1]), per_g_tpr[:, 1], per_g_tpr[WORST, 1]),
            "TPR@0%": col(np.nanmean(ov_tpr[:, 2]), per_g_tpr[:, 2], per_g_tpr[WORST, 2]),
        }
        print(f"[{method}] AUC {result[method]['AUC']['unadjusted']:.4f} | "
              f"TPR@1% {result[method]['TPR@1%']['unadjusted']:.4f} | blonde-male AUC {per_g[WORST]:.4f}",
              flush=True)

    out = paths.ensure(paths.results_dir("studies", "celeba_subgroup")) / "subgroup_rmia_celeba.json"
    out.write_text(json.dumps(result, indent=2))
    cols = ["unadjusted", "adjusted"] + GROUP + ["blonde_male"]
    for metric in ["AUC", "TPR@1%", "TPR@0.01%", "TPR@0%"]:
        print(f"\n=== CelebA RMIA {metric} (unaware; higher = more leakage) ===")
        print(f"{'method':10s} " + " ".join(f"{c[:14]:>15s}" for c in cols))
        for m in METHODS:
            if m in result:
                print(f"{m:10s} " + " ".join(f"{result[m][metric][c]:15.4f}" for c in cols))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
