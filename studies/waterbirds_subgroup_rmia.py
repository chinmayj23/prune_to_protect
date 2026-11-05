#!/usr/bin/env python
"""Per-GROUP RMIA privacy for Waterbirds (defense-unaware) — the fairness-of-privacy readout.

Post-processing only: joins each target model's per-sample RMIA (scores, answers=membership) with the
Waterbirds group labels, and reports overall + per-group AUC and TPR@{1%,0.1%} averaged over the 8
target models.

Reads the per-target predictions written by scripts/attack.py (report_*/attack_predictions_*.npz).
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

GROUPS = ["landbird/land", "landbird/water", "waterbird/land", "waterbird/water"]
MINORITY = {1, 2}                              # landbird/water, waterbird/land
FPRS = [0.01, 0.0001, 0.0]                     # TPR@1%, TPR@0.01%, TPR@0%
METHODS = ["standard", "infobatch", "ib_prw"]  # all vs standard shadows (defense-unaware)


def per_target(report_dir):
    out = []
    for f in sorted(glob.glob(str(Path(report_dir) / "attack_predictions_*.npz"))):
        z = np.load(f)
        out.append((z["scores"].ravel(), z["answers"].ravel()))
    return out


def auc_of(s, a):
    y = a.astype(int)
    return np.nan if y.min() == y.max() else roc_auc_score(y, -s)  # lower score = member


def tprs_of(s, a):
    y = a.astype(int)
    if y.min() == y.max():
        return [np.nan] * len(FPRS)
    fpr, tpr, _ = roc_curve(y, -s)
    return [float(tpr[fpr <= f].max()) if (fpr <= f).any() else 0.0 for f in FPRS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18_224")
    ap.add_argument("--pretrained", action="store_true", help="analyze the _pt (pretrained) banks")
    args = ap.parse_args()
    rec = "_pt" if args.pretrained else ""
    groups = torch.load(paths.data_dir("waterbirds", "waterbirds_train.pt"),
                        map_location="cpu", weights_only=False)["groups"].numpy()
    result = {}
    for method in METHODS:
        tag = f"waterbirds_{args.model}{rec}_{method}_vs_standard_rmia_online_unaware"
        d = paths.results_dir("attacks", tag)
        reports = [p for p in d.glob("report_*") if p.is_dir()] if d.exists() else []
        preds = per_target(reports[0]) if reports else per_target(d)
        if not preds:
            print(f"[skip] {method}: no attack_predictions under {d}")
            continue
        # overall + per-group AUC/TPR: per target, then average over targets
        ov_auc = [auc_of(s, a) for s, a in preds]
        ov_tpr = np.array([tprs_of(s, a) for s, a in preds])                # [T, F]
        g_auc = np.full((len(preds), 4), np.nan)
        g_tpr = np.full((len(preds), 4, len(FPRS)), np.nan)
        for ti, (s, a) in enumerate(preds):
            for g in range(4):
                m = groups == g
                g_auc[ti, g] = auc_of(s[m], a[m])
                g_tpr[ti, g] = tprs_of(s[m], a[m])
        per_g = np.nanmean(g_auc, axis=0)
        per_g_tpr = np.nanmean(g_tpr, axis=0)                               # [4, F]
        mino = list(MINORITY)

        def col(auc_overall, auc_groups, auc_min):
            return {"overall": round(float(auc_overall), 4),
                    **{GROUPS[g]: round(float(auc_groups[g]), 4) for g in range(4)},
                    "minority": round(float(auc_min), 4)}

        result[method] = {
            "AUC": col(np.nanmean(ov_auc), per_g, np.nanmean(per_g[mino])),
            "TPR@1%": col(np.nanmean(ov_tpr[:, 0]), per_g_tpr[:, 0], np.nanmean(per_g_tpr[mino, 0])),
            "TPR@0.01%": col(np.nanmean(ov_tpr[:, 1]), per_g_tpr[:, 1], np.nanmean(per_g_tpr[mino, 1])),
            "TPR@0%": col(np.nanmean(ov_tpr[:, 2]), per_g_tpr[:, 2], np.nanmean(per_g_tpr[mino, 2])),
        }
        print(f"[{method}] AUC {result[method]['AUC']['overall']:.4f} | "
              f"TPR@1% {result[method]['TPR@1%']['overall']:.4f} | "
              f"TPR@0.01% {result[method]['TPR@0.01%']['overall']:.4f} | "
              f"TPR@0% {result[method]['TPR@0%']['overall']:.4f}", flush=True)

    out = paths.ensure(paths.results_dir("studies", "waterbirds_subgroup")) / f"subgroup_rmia{rec}.json"
    out.write_text(json.dumps(result, indent=2))
    cols = ["overall"] + GROUPS + ["minority"]
    for metric in ["AUC", "TPR@1%", "TPR@0.01%", "TPR@0%"]:
        print(f"\n=== Per-group RMIA {metric} ({'pretrained' if rec else 'scratch'}, unaware; "
              f"higher = more leakage) ===")
        print(f"{'method':10s} " + " ".join(f"{c[:12]:>13s}" for c in cols))
        for m in METHODS:
            if m not in result:
                continue
            print(f"{m:10s} " + " ".join(f"{result[m][metric][c]:13.4f}" for c in cols))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
