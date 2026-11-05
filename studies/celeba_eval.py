#!/usr/bin/env python
"""Train a CelebA model (Blond_Hair) and report overall + per-GROUP test accuracy (worst = blonde male)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
import numpy as np, torch
from torch.amp import autocast
from prunepriv.configs import build_model
from prunepriv.training import train_full
from prunepriv import paths

GROUP = ["non-blonde/female", "non-blonde/male", "blonde/female", "blonde/male"]  # minority = blonde/male (3)
device = torch.device("cuda")
STORE, CROP = 128, 112
_BLOBS = {}


def _split(split):
    if split not in _BLOBS:
        d = torch.load(paths.data_dir("celeba", f"celeba_{split}.pt"), map_location="cpu", weights_only=False)
        _BLOBS[split] = (d["images"], d["labels"], d["groups"].numpy())
    return _BLOBS[split]


def load_sd(p):
    sd = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    return sd


@torch.no_grad()
def per_group(ckpt, model_name, split="test"):
    imgs, labels, groups = _split(split)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    net = build_model(model_name, num_classes=2).to(device); net.load_state_dict(load_sd(ckpt)); net.eval()
    o = (STORE - CROP) // 2
    correct = np.zeros(len(imgs), bool)
    for i in range(0, len(imgs), 512):
        x = imgs[i:i+512].to(device).float().div(255)[:, :, o:o+CROP, o:o+CROP]
        x = (x - mean) / std
        with autocast("cuda", dtype=torch.bfloat16):
            pred = net(x.to(memory_format=torch.channels_last)).float().argmax(1)
        correct[i:i+512] = (pred == labels[i:i+512].to(device)).cpu().numpy()
    accs = {GROUP[g]: 100 * correct[groups == g].mean() for g in range(4)}
    return accs, 100 * correct.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18_224")
    ap.add_argument("--method", default="standard")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    out = paths.ensure(paths.results_dir("studies", "celeba") / f"{args.model}_{args.method}_s{args.seed}")
    r = train_full("celeba", model=args.model, method=args.method, seed=args.seed,
                   epochs=args.epochs, output_dir=out, save_last_checkpoint=True)
    accs, overall = per_group(out / "last_epoch.pt", args.model, split="test")
    worst = min(accs.values())
    # adjusted avg = per-group reweighted by train proportions
    W = np.array([71629, 66874, 22880, 1387], float); W /= W.sum()
    adj = float((np.array(list(accs.values())) / 100 * W).sum() * 100)
    print(f"\n=== CelebA {args.model}/{args.method} ({args.epochs}ep) ===")
    print(f"raw {overall:.2f}%  adjusted {adj:.2f}%  worst-group {worst:.2f}%  (model-report {r['final_test_accuracy']:.2f}%)")
    for g, a in accs.items():
        print(f"  {g:18s}: {a:.2f}%")


if __name__ == "__main__":
    main()
