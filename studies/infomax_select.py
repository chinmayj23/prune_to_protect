#!/usr/bin/env python
"""InfoMax coreset selection (Tan et al., ICLR 2025, Alg. 1, arXiv:2506.01701).

InfoMax is a Discrete Quadratic Program:
    max_X  Σ X_z I(z) − α Σ K_{z,s} X_z X_s   s.t.  Σ X_z = p,  X∈{0,1}^N
solved by the continuous-relaxation fixed point (Eq. 3):
    X^{t+1} = Softmax(p·I − 2pα·K·X^t),  X^0 = 1/N,  T=20,  α=0.3,  K = top-k(=5) feature similarity.
Then take the top-p samples by X^T.

  * importance I = EL2N (||softmax(logits) − onehot||_2) from a full-data ResNet-20, normalized to a
    distribution (I ← I/ΣI).
  * K = inner-product feature similarity F·F^T, top-5 per row, row-stochastic.
  * These normalizations keep the softmax argument O(1). The InfoMax paper leaves the scaling
    unspecified, and Eq. 3 with raw I/K and p≈32k saturates the softmax to one-hot.

The budget is set to InfoBatch's retained fraction (0.645). Writes
results/studies/infomax/coreset_index_cifar10.npy.
"""
from __future__ import annotations

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

OUT = paths.ensure(paths.results_dir("studies", "infomax"))
CKPT = paths.results_dir("full_models", "cifar10_resnet20", "standard", "seed_3407", "last_epoch.pt")
DATA = str(paths.data_dir("cifar10"))
RATIO, ALPHA, T, K = 0.645, 0.3, 20, 5
device = torch.device("cuda")


def load_sd(p):
    sd = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    return sd


def main():
    train = CifarLoader(path=DATA, dataset_name="cifar10", train=True, batch_size=1000,
                        shuffle=False, device=device, aug={})
    imgs, labels = train.images, train.labels.to(device)
    N = len(imgs)
    model = build_model("resnet20", num_classes=10).to(device)
    model.load_state_dict(load_sd(CKPT))
    model.eval()

    # importance I = EL2N ; features = penultimate (fc input)
    feats = torch.empty(N, 64, device=device)
    el2n = torch.empty(N, device=device)
    cap = {}
    h = model.fc.register_forward_pre_hook(lambda m, inp: cap.__setitem__("f", inp[0].detach()))
    with torch.no_grad():
        for i in range(0, N, 1000):
            x = train.normalize(imgs[i:i + 1000]).to(memory_format=torch.channels_last)
            with autocast("cuda", dtype=torch.bfloat16):
                logit = model(x).float()
            feats[i:i + 1000] = cap["f"].float()
            p_ = F.softmax(logit, 1)
            oh = F.one_hot(labels[i:i + 1000], 10).float()
            el2n[i:i + 1000] = (p_ - oh).norm(dim=1)
    h.remove()

    # K = top-5 neighbours by inner-product similarity F·F^T, row-stochastic for a stable softmax.
    fu = feats
    nbr_idx = torch.empty(N, K, dtype=torch.long, device=device)
    nbr_w = torch.empty(N, K, device=device)
    for i in range(0, N, 2000):
        sim = fu[i:i + 2000] @ fu.T                     # (chunk, N)
        sim.scatter_(1, torch.arange(i, min(i + 2000, N), device=device)[:, None], -1e9)  # drop self
        w, idx = sim.topk(K, dim=1)
        w = w.clamp(min=0.0)
        w = w / w.sum(1, keepdim=True).clamp(min=1e-12)  # row-stochastic
        nbr_idx[i:i + 2000] = idx
        nbr_w[i:i + 2000] = w

    def Kmatvec(X):  # (K @ X)_z = Σ_j w[z,j] X[nbr[z,j]]
        return (nbr_w * X[nbr_idx]).sum(1)

    I = el2n / el2n.sum()                                # importance as a distribution
    p = int(round(RATIO * N))
    X = torch.full((N,), 1.0 / N, device=device)         # X^0 uniform
    for _ in range(T):
        X = torch.softmax(p * I - 2 * p * ALPHA * Kmatvec(X), dim=0)
    coreset = torch.topk(X, p).indices.sort().values.cpu().numpy()

    np.save(OUT / "coreset_index_cifar10.npy", coreset)
    # class balance sanity
    lab = labels.cpu().numpy()[coreset]
    counts = np.bincount(lab, minlength=10)
    print(f"InfoMax coreset: {len(coreset)}/{N} = {len(coreset)/N:.4f}")
    print(f"per-class counts: {counts.tolist()} (mean {counts.mean():.0f})")
    print(f"EL2N of selected vs all: sel {el2n.cpu().numpy()[coreset].mean():.3f} vs all {el2n.mean():.3f}")
    print(f"Wrote {OUT/'coreset_index_cifar10.npy'}")


if __name__ == "__main__":
    main()
