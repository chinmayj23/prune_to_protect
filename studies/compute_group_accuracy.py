#!/usr/bin/env python
"""Per-class test accuracy, hard-group train accuracy, and per-step easy/medium/hard group accuracy
(CIFAR-10 / ResNet-20; Tables 6 and 9). Reads the seed-3407 checkpoints written by
studies/train_full_models.py. Light GPU inference.

Groups: TRAIN tiers = three equal-width bins of the cumulative pruning rate rho (hard = lowest rho), as
in the paper. WLIB's pruning phase is InfoBatch's, so its rho is used. TEST tiers =
per-sample test-loss terciles under the fully-trained standard model (method-independent).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from prunepriv import paths
from prunepriv.configs import build_model
from prunepriv.training.engine import CifarLoader

FULL = paths.results_dir("full_models", "cifar10_resnet20")
OUT = paths.ensure(paths.results_dir("studies", "perstep"))
DATA = str(paths.data_dir("cifar10"))
SEED = 3407
METHODS = {"standard": "standard", "infobatch": "infobatch", "ib_prw": "wlib"}
STEPS = [1000, 2000, 3000, 4000, 5000, 6300]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
device = torch.device("cuda")


def load_sd(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    return sd


def predict(model, imgs, labels, normalize, bs=1024):
    corr = torch.empty(len(imgs), dtype=torch.bool, device=device)
    loss = torch.empty(len(imgs), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = normalize(imgs[i:i + bs]).to(memory_format=torch.channels_last)
            with autocast("cuda", dtype=torch.bfloat16):
                out = model(x).float()
            corr[i:i + bs] = out.argmax(1) == labels[i:i + bs]
            loss[i:i + bs] = F.cross_entropy(out, labels[i:i + bs], reduction="none")
    return corr.cpu().numpy(), loss.cpu().numpy()


def rho_tiers(rho):
    edges = np.linspace(rho.min(), rho.max(), 4)
    t = np.full(len(rho), 1, np.int8)   # medium
    t[rho <= edges[1]] = 0              # hard = lowest rho
    t[rho > edges[2]] = 2               # easy
    return t


def loss_terciles(x):
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    t = np.full(len(x), 1, np.int8)   # medium
    t[x >= q2] = 0                    # hard = highest test loss
    t[x <= q1] = 2                    # easy
    return t


def main():
    tr = CifarLoader(path=DATA, dataset_name="cifar10", train=True, batch_size=512, shuffle=False, device=device)
    te = CifarLoader(path=DATA, dataset_name="cifar10", train=False, batch_size=512, shuffle=False, device=device)
    tr_imgs, tr_lab = tr.images, tr.labels.to(device)
    te_imgs, te_lab = te.images, te.labels.to(device)
    model = build_model("resnet20", num_classes=10).to(device)

    # rho tiers (train); test-loss tiers (from fully-trained standard)
    rhos = [np.load(f).ravel() for f in sorted(glob.glob(str(FULL / "ib_prw" / "seed_*" / "prune_rate.npy")))]
    rho = np.mean(np.stack(rhos), axis=0)
    train_tier = rho_tiers(rho)
    model.load_state_dict(load_sd(FULL / "standard" / f"seed_{SEED}" / "last_epoch.pt"))
    _, ref_test_loss = predict(model, te_imgs, te_lab, te.normalize)
    test_tier = loss_terciles(ref_test_loss)
    tnames = ["hard", "medium", "easy"]

    result = {"per_class_test_acc": {}, "hard_group_train_acc": {}, "per_step": {}}
    tr_lab_np = tr_lab.cpu().numpy()

    for mkey, disp in METHODS.items():
        ck = FULL / mkey / f"seed_{SEED}" / "last_epoch.pt"
        model.load_state_dict(load_sd(ck))
        c_te, _ = predict(model, te_imgs, te_lab, te.normalize)
        c_tr, _ = predict(model, tr_imgs, tr_lab, tr.normalize)
        te_lab_np = te_lab.cpu().numpy()
        result["per_class_test_acc"][disp] = {CLASSES[c]: round(100 * float(c_te[te_lab_np == c].mean()), 2) for c in range(10)}
        result["hard_group_train_acc"][disp] = {tnames[t]: round(100 * float(c_tr[train_tier == t].mean()), 2) for t in range(3)}
        print(f"[last] {disp}: worst-class {min(result['per_class_test_acc'][disp].values()):.1f}% "
              f"hard-train {result['hard_group_train_acc'][disp]['hard']:.1f}%", flush=True)

    # per-step easy/mid/hard train+test accuracy
    for mkey, disp in METHODS.items():
        result["per_step"][disp] = {}
        for s in STEPS:
            ck = FULL / mkey / f"seed_{SEED}" / f"step_{s}.pt"
            if not ck.exists():
                continue
            model.load_state_dict(load_sd(ck))
            c_te, _ = predict(model, te_imgs, te_lab, te.normalize)
            c_tr, _ = predict(model, tr_imgs, tr_lab, tr.normalize)
            result["per_step"][disp][s] = {
                "train": {tnames[t]: round(100 * float(c_tr[train_tier == t].mean()), 1) for t in range(3)},
                "test": {tnames[t]: round(100 * float(c_te[test_tier == t].mean()), 1) for t in range(3)},
            }
        hard_tr = {s: result["per_step"][disp][s]["train"]["hard"] for s in result["per_step"][disp]}
        print(f"[step] {disp} hard-train by step: {hard_tr}", flush=True)

    (OUT / "group_accuracy.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT/'group_accuracy.json'}")


if __name__ == "__main__":
    main()
