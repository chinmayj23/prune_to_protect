#!/usr/bin/env python
"""Build CelebA blobs (Blond_Hair target, Male spurious) from the HF mirror tpremoli/CelebA-attrs.

Preprocess: CenterCrop(178) on the aligned 178x218 face, Resize(128) -> uint8 [N,3,128,128]. Stores
label (Blond_Hair 0/1), male (0/1), and group = 2*label + male (0 nb/f, 1 nb/m, 2 b/f, 3 b/m;
minority = group 3, blonde males). Writes <DATA>/celeba/celeba_{train,val,test}.pt.
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

OUT = paths.ensure(paths.data_dir("celeba"))
STORE = 128


def to_blob(split):
    n = len(split)
    imgs = torch.empty(n, 3, STORE, STORE, dtype=torch.uint8)
    labels = torch.empty(n, dtype=torch.long)
    male = torch.empty(n, dtype=torch.long)
    for i in range(n):
        ex = split[i]
        img = ex["image"].convert("RGB")
        img = TF.resize(TF.center_crop(img, 178), STORE)
        imgs[i] = TF.pil_to_tensor(img)
        labels[i] = 1 if int(ex["Blond_Hair"]) == 1 else 0   # +1/-1 -> 1/0
        male[i] = 1 if int(ex["Male"]) == 1 else 0
        if (i + 1) % 20000 == 0:
            print(f"    {i+1}/{n}", flush=True)
    groups = 2 * labels + male
    return {"images": imgs, "labels": labels, "groups": groups, "place": male}


def main():
    ds = load_dataset("tpremoli/CelebA-attrs")
    print(f"splits: { {k: len(v) for k, v in ds.items()} }", flush=True)
    # HF: train / validation / test  ->  celeba_{train,val,test}
    for hf_split, name in [("train", "train"), ("validation", "val"), ("test", "test")]:
        print(f"building {name} from HF '{hf_split}' ...", flush=True)
        b = to_blob(ds[hf_split])
        torch.save(b, OUT / f"celeba_{name}.pt")
        g = torch.bincount(b["groups"], minlength=4).tolist()
        print(f"  {name}: {b['images'].shape} groups[nb/f, nb/m, b/f, b/m]={g}  "
              f"blonde={int((b['labels']==1).sum())}", flush=True)
    print(f"Wrote {OUT}/celeba_{{train,val,test}}.pt")


if __name__ == "__main__":
    main()
