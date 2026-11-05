#!/usr/bin/env python
"""InfoMax RMIA: train infomax shadow+target banks (coreset ∩ 50% pool), then attack.

- aware:   infomax targets vs infomax shadows   (via run_attack_pipeline)
- unaware: infomax targets vs the standard shadow bank from scripts/attack.py
Per-target mean±std over the 20-target bank. RMIA, online.
Writes results/studies/infomax/infomax_rmia_metrics.json.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from prunepriv.attacks.runner import run_attack_pipeline, write_attack_config  # noqa: E402
from prunepriv import paths  # noqa: E402

ATTACK_DIR = paths.REPO_ROOT / "src" / "prunepriv" / "attacks" / "rmia"
STD_SHADOWS = paths.banks_dir("cifar10_resnet20_standard_shadows_64")
OUT = paths.ensure(paths.results_dir("studies", "infomax"))
FPRS = [0.01, 0.001, 0.0001, 0.0]


def metrics(report_dir):
    aucs, tpr = [], []
    for f in sorted(glob.glob(os.path.join(report_dir, "attack_predictions_*.npz"))):
        z = np.load(f)
        y = z["answers"].ravel().astype(int)
        s = -z["scores"].ravel()
        if y.min() == y.max():
            continue
        aucs.append(roc_auc_score(y, s))
        fpr, t, _ = roc_curve(y, s)
        tpr.append([float(t[fpr <= fp].max()) if (fpr <= fp).any() else 0.0 for fp in FPRS])
    aucs, tpr = np.array(aucs), np.array(tpr)
    return {"auc": float(aucs.mean()), "auc_std": float(aucs.std(ddof=1)),
            "tpr@1%": float(tpr[:, 0].mean()), "tpr@0.1%": float(tpr[:, 1].mean()),
            "tpr@0.01%": float(tpr[:, 2].mean()), "tpr@0%": float(tpr[:, 3].mean()),
            "n_targets": len(aucs)}


def run_unaware(target_dir):
    results_dir = paths.ensure(paths.results_dir("attacks", "cifar10_resnet20_infomax_vs_standard_rmia_online_unaware"))
    report = "report_rmia_online_std_shadows"
    (results_dir / report).mkdir(parents=True, exist_ok=True)
    cfg_path = results_dir / "attack_config.yaml"
    write_attack_config(cfg_path, results_dir, report, Path(target_dir), STD_SHADOWS,
                        50000, 64, "rmia", True)
    env = dict(os.environ, MPLBACKEND="Agg")
    with (results_dir / "attack.log").open("w") as log:
        subprocess.run([sys.executable, "main.py", f"--cf={cfg_path}"], cwd=str(ATTACK_DIR),
                       stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    return str(results_dir / report)


def main():
    print("=== training infomax banks + AWARE attack (this is the long part) ===", flush=True)
    aware = run_attack_pipeline(dataset="cifar10", model="resnet20", target_method="infomax",
                                shadow_method="infomax", attack="rmia", online=True,
                                num_shadows=64, num_targets=20)
    print("=== UNAWARE attack (infomax targets vs existing standard shadows) ===", flush=True)
    unaware_report = run_unaware(aware["target_dir"])

    out = {"aware": metrics(aware["report_dir"]), "unaware": metrics(unaware_report)}
    (OUT / "infomax_rmia_metrics.json").write_text(json.dumps(out, indent=2))
    for k, m in out.items():
        print(f"infomax {k}: AUC={m['auc']:.4f}±{m['auc_std']:.3f} TPR@1%={100*m['tpr@1%']:.2f} "
              f"@0.01%={100*m['tpr@0.01%']:.2f} @0%={100*m['tpr@0%']:.2f} (n={m['n_targets']})", flush=True)
    print(f"Wrote {OUT/'infomax_rmia_metrics.json'}")


if __name__ == "__main__":
    main()
