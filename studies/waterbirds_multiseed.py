#!/usr/bin/env python
"""Multi-seed Waterbirds accuracy comparison (standard / infobatch / ib_prw), FINAL-epoch.

Worst-group accuracy on Waterbirds is high-variance (the waterbird/land minority has 56 training
examples), so this reports mean +/- std per group and worst-group across seeds. All methods use
TRAIN_DEFAULTS (SGD lr0.1, wd5e-4, triangular schedule, RandomCrop+flip, 100 epochs).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from prunepriv.training import train_full
from prunepriv import paths
from waterbirds_eval import per_group, GROUP

MODEL = "resnet10t"
METHODS = ["standard", "infobatch", "ib_prw"]
SEEDS = [0, 1, 7, 1234, 3407]
EPOCHS = 100


def main():
    root = paths.ensure(paths.results_dir("studies", "waterbirds", "multiseed"))
    results = {m: {"worst": [], "raw": [], "groups": [[] for _ in range(4)]} for m in METHODS}
    for m in METHODS:
        for s in SEEDS:
            out = paths.ensure(root / f"{MODEL}_{m}_s{s}")
            train_full("waterbirds", model=MODEL, method=m, seed=s, epochs=EPOCHS,
                       output_dir=out, save_last_checkpoint=True)
            accs, raw = per_group(out / "last_epoch.pt", MODEL, split="test")
            gv = list(accs.values())
            results[m]["worst"].append(min(gv)); results[m]["raw"].append(raw)
            for i in range(4):
                results[m]["groups"][i].append(gv[i])
            print(f"[{m} s{s}] raw {raw:.2f}  worst {min(gv):.2f}  "
                  f"[{' '.join(f'{v:.1f}' for v in gv)}]", flush=True)

    def ms(a):
        return f"{np.mean(a):.2f} ± {np.std(a):.2f}"

    print("\n================ Waterbirds resnet10t FROM SCRATCH | final-epoch, "
          f"{len(SEEDS)} seeds ================")
    hdr = f"{'method':10s} {'raw':>14s} {'worst-grp':>14s} | " + " ".join(f"{g[:11]:>13s}" for g in GROUP)
    print(hdr)
    for m in METHODS:
        row = f"{m:10s} {ms(results[m]['raw']):>14s} {ms(results[m]['worst']):>14s} | " + \
              " ".join(f"{ms(results[m]['groups'][i]):>13s}" for i in range(4))
        print(row)
    with open(root / "multiseed_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {root/'multiseed_summary.json'}")


if __name__ == "__main__":
    main()
