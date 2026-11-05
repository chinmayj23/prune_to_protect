#!/usr/bin/env python
"""Download Waterbirds (HF grodino/waterbirds) and preprocess to GPU-loadable uint8 blobs (256px).

Stores label (landbird/waterbird) AND group metadata for the fairness analysis:
  group = 2*label + place  (0 landbird/land, 1 landbird/water, 2 waterbird/land, 3 waterbird/water;
  minority = groups 1 & 2, the group-conflicting cases). place = land/water background (spurious).
Train pool = official train (4795); test = official test (5794).
Writes data/waterbirds/waterbirds_{train,test}.pt =
  {'images': uint8[N,3,256,256], 'labels': long[N], 'groups': long[N], 'place': long[N]}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from datasets import load_dataset

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
from prunepriv import paths  # noqa: E402

OUT = paths.ensure(paths.data_dir("waterbirds"))


def to_blob(split):
    n = len(split)
    imgs = torch.empty(n, 3, 256, 256, dtype=torch.uint8)
    labels = torch.empty(n, dtype=torch.long)
    place = torch.empty(n, dtype=torch.long)
    for i in range(n):
        ex = split[i]
        img = ex["image"].convert("RGB")
        img = TF.center_crop(TF.resize(img, 256), 256)
        imgs[i] = TF.pil_to_tensor(img)
        labels[i] = int(ex["label"])
        place[i] = int(ex["place"])
    groups = 2 * labels + place
    return {"images": imgs, "labels": labels, "groups": groups, "place": place}


def main():
    ds = load_dataset("grodino/waterbirds")
    print(f"splits: { {k: len(v) for k, v in ds.items()} }", flush=True)
    tr = to_blob(ds["train"])
    te = to_blob(ds["test"])
    torch.save(tr, OUT / "waterbirds_train.pt")
    torch.save(te, OUT / "waterbirds_test.pt")
    for name, b in [("train", tr), ("test", te)]:
        g = torch.bincount(b["groups"], minlength=4).tolist()
        print(f"{name}: {b['images'].shape} labels {b['labels'].bincount().tolist()} "
              f"groups[LL,LW,WL,WW]={g}", flush=True)
    print(f"Wrote {OUT}/waterbirds_train.pt + waterbirds_test.pt")


if __name__ == "__main__":
    main()
