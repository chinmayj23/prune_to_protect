#!/usr/bin/env python
"""Train a Waterbirds model and report overall + per-GROUP test accuracy (fairness / worst-group)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.amp import autocast  # noqa: E402

from prunepriv.configs import build_model  # noqa: E402
from prunepriv.training import train_full  # noqa: E402
from prunepriv import paths  # noqa: E402

GROUP = ["landbird/land", "landbird/water", "waterbird/land", "waterbird/water"]
device = torch.device("cuda")


def load_sd(p):
    sd = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    return sd


_BLOBS = {}


def _load_split(split):
    if split not in _BLOBS:
        d = torch.load(paths.data_dir("waterbirds", f"waterbirds_{split}.pt"),
                       map_location="cpu", weights_only=False)
        _BLOBS[split] = (d["images"].to(device), d["labels"].to(device), d["groups"].numpy())
    return _BLOBS[split]


@torch.no_grad()
def per_group(ckpt, model_name, split="test"):
    imgs, labels, groups = _load_split(split)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    net = build_model(model_name, num_classes=2).to(device)
    net.load_state_dict(load_sd(ckpt))
    net.eval()
    correct = np.zeros(len(imgs), bool)
    for i in range(0, len(imgs), 256):
        x = imgs[i:i + 256].float().div(255)
        x = x[:, :, 16:240, 16:240]                 # center crop 224 from 256
        x = (x - mean) / std
        with autocast("cuda", dtype=torch.bfloat16):
            pred = net(x.to(memory_format=torch.channels_last)).float().argmax(1)
        correct[i:i + 256] = (pred == labels[i:i + 256]).cpu().numpy()
    accs = {GROUP[g]: 100 * correct[groups == g].mean() for g in range(4)}
    return accs, 100 * correct.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet10t")
    ap.add_argument("--method", default="standard")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--pretrained", action="store_true", help="load ImageNet weights (fine-tune)")
    ap.add_argument("--lr", type=float, default=None, help="override learning rate")
    ap.add_argument("--wd", type=float, default=None, help="override weight decay")
    ap.add_argument("--tag", default="", help="extra tag suffix for the output dir")
    ap.add_argument("--track", type=int, default=0,
                    help="if >0, save a checkpoint every N epochs and report best-worst-group epoch")
    ap.add_argument("--prw-mode", default=None,
                    help="ib_prw weighting mode: pruning_rate|pruning_rate_inv|loss_final|hard_tercile_const|shuffle")
    ap.add_argument("--prw-eps", type=float, default=None, help="ib_prw weight floor (0.05=aggressive .. 1.0=uniform)")
    args = ap.parse_args()
    tag = f"{args.model}_{args.method}{'_pt' if args.pretrained else ''}_s{args.seed}{args.tag}"
    out = paths.ensure(paths.results_dir("studies", "waterbirds") / tag)
    over = dict(pretrained=args.pretrained)
    if args.lr is not None:
        over["learning_rate"] = args.lr
    if args.wd is not None:
        over["weight_decay"] = args.wd
    if args.prw_mode is not None:
        over["prw_weight_mode"] = args.prw_mode
    if args.prw_eps is not None:
        over["prw_epsilon"] = args.prw_eps
    track_epochs = []
    if args.track > 0:
        track_epochs = list(range(0, args.epochs, args.track))
        over["save_epoch_checkpoints"] = True
        over["sample_at_epoch"] = track_epochs
    r = train_full("waterbirds", model=args.model, method=args.method, seed=args.seed,
                   epochs=args.epochs, output_dir=out, save_last_checkpoint=True, **over)
    accs, overall = per_group(out / "last_epoch.pt", args.model)
    worst = min(accs.values())
    print(f"\n=== Waterbirds {args.model}/{args.method} ({args.epochs}ep){args.tag} ===")
    print(f"[final epoch] raw avg {overall:.2f}%  worst-group {worst:.2f}%")
    for g, a in accs.items():
        print(f"  {g:18s}: {a:.2f}%")

    if track_epochs:
        # Literature protocol: SELECT the epoch by best worst-group on VAL, REPORT that epoch's TEST.
        print("\n[model selection on VAL worst-group -> report TEST]")
        cands = list(track_epochs) + [args.epochs - 1]  # include final epoch
        best = (-1, -1.0, None, -1.0)  # epoch, val-worst, test_accs, test_raw
        for e in cands:
            ck = out / (f"epoch_{e}.pt" if e in track_epochs else "last_epoch.pt")
            if not ck.exists():
                continue
            vg, _ = per_group(ck, args.model, split="val")
            vw = min(vg.values())
            tg, traw = per_group(ck, args.model, split="test")
            print(f"  ep {e:3d}: VAL-worst {vw:5.2f}%  ->  TEST worst {min(tg.values()):5.2f}% "
                  f"raw {traw:5.2f}%  [{'  '.join(f'{v:.0f}' for v in tg.values())}]")
            if vw > best[1]:
                best = (e, vw, tg, traw)
        be, bvw, tga, traw = best
        print(f"\n*** MODEL-SELECTED (best VAL worst-group {bvw:.2f}% @ epoch {be}) -> TEST: "
              f"worst-group {min(tga.values()):.2f}%  raw {traw:.2f}% ***")
        for g, a in tga.items():
            print(f"  {g:18s}: {a:.2f}%")


if __name__ == "__main__":
    main()
