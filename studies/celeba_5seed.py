#!/usr/bin/env python
"""CelebA 5-seed accuracy for standard / infobatch / ib_prw (resnet10t, 50ep). Reports mean +/- std of
raw / adjusted / worst-group + per-group. Adjusted = per-group reweighted by train proportions."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src")); sys.path.insert(0, str(_REPO / "studies"))
import numpy as np
from prunepriv.training import train_full
from prunepriv import paths
from celeba_eval import per_group, GROUP

W = np.array([71629, 66874, 22880, 1387], float); W /= W.sum()   # train group proportions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet10t")
    ap.add_argument("--methods", nargs="+", default=["standard", "infobatch", "ib_prw"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 7, 1234, 3407])
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    root = paths.ensure(paths.results_dir("studies", "celeba", "fiveseed"))
    res = {}
    for m in args.methods:
        worst, raw, adj, groups = [], [], [], [[] for _ in range(4)]
        for s in args.seeds:
            out = paths.ensure(root / f"{args.model}_{m}_s{s}")
            train_full("celeba", model=args.model, method=m, seed=s, epochs=args.epochs,
                       output_dir=out, save_last_checkpoint=True)
            accs, r = per_group(out / "last_epoch.pt", args.model, split="test")
            gv = np.array(list(accs.values()))
            worst.append(gv.min()); raw.append(r); adj.append(float((gv / 100 * W).sum() * 100))
            for i in range(4):
                groups[i].append(gv[i])
            print(f"[{m} s{s}] raw {r:.2f} adj {adj[-1]:.2f} worst {gv.min():.2f} "
                  f"[{' '.join(f'{v:.1f}' for v in gv)}]", flush=True)
        res[m] = dict(worst=worst, raw=raw, adj=adj, groups=groups)

    def ms(a):
        return f"{np.mean(a):.2f} ± {np.std(a):.2f}"

    print(f"\n===== CelebA {args.model} FROM SCRATCH | {args.epochs}ep | {len(args.seeds)} seeds =====")
    print(f"{'method':10s} {'worst':>13s} {'raw':>13s} {'adjusted':>13s} | "
          + " ".join(f"{g[:13]:>15s}" for g in GROUP))
    for m in args.methods:
        d = res[m]
        print(f"{m:10s} {ms(d['worst']):>13s} {ms(d['raw']):>13s} {ms(d['adj']):>13s} | "
              + " ".join(f"{ms(d['groups'][i]):>15s}" for i in range(4)))

    # LOG everything to JSON so tables can be generated later
    summary = {"model": args.model, "epochs": args.epochs, "seeds": args.seeds, "groups": GROUP,
               "per_method": {}}
    for m in args.methods:
        d = res[m]
        summary["per_method"][m] = {
            "raw": {"mean": float(np.mean(d["raw"])), "std": float(np.std(d["raw"])), "per_seed": d["raw"]},
            "adjusted": {"mean": float(np.mean(d["adj"])), "std": float(np.std(d["adj"])), "per_seed": d["adj"]},
            "worst": {"mean": float(np.mean(d["worst"])), "std": float(np.std(d["worst"])), "per_seed": d["worst"]},
            "per_group": {GROUP[i]: {"mean": float(np.mean(d["groups"][i])),
                                     "std": float(np.std(d["groups"][i])), "per_seed": d["groups"][i]}
                          for i in range(4)},
        }
    jp = root / "celeba_accuracy_5seed.json"
    jp.write_text(json.dumps(summary, indent=2))
    print(f"\nLOGGED -> {jp}")


if __name__ == "__main__":
    main()
