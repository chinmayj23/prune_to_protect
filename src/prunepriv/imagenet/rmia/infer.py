"""
DDP inference over the train beton to produce per-augmentation RMIA signals.

For every model checkpoint, we write signal_train.npy of shape
[NUM_TRAIN, 18], float32, aligned with the beton `index` field. Entry
[n, k] is the softmax probability of the ground-truth class of sample n
under augmentation query k (T=1).

The 18 queries follow the RMIA paper (arXiv:2312.03262v3, App. B.5):
the Cartesian product of
    reflect ∈ {False, True}                (2)
    dx     ∈ {0, shift, 2*shift}           (3)
    dy     ∈ {0, shift, 2*shift}           (3)
applied after reflect-padding the 224x224 center crop by `shift` pixels
on each side, giving a spatial offset of {-shift, 0, +shift} around the
centered crop. We use shift=7 — 3.1% of 224, matching CIFAR's relative
perturbation magnitude (1 pixel / 32). Single-pixel shifts on 224x224
produce near-zero diversity and would collapse the ensemble.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m prunepriv.imagenet.rmia.infer \
        --config-file src/prunepriv/imagenet/configs/resnet50_standard.yaml \
        --ckpt <path_to_model.pt> \
        --out <path_to_signal_train.npy>
"""
from __future__ import annotations

import argparse
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from fastargs import get_current_config
from torch.amp import autocast
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness  # registers fastargs sections  # noqa: E402
from utils.models import build_model  # noqa: E402
from utils.dataset import build_test_loader  # noqa: E402


NUM_TRAIN = harness.NUM_TRAIN_SAMPLES
NB_AUGMENTATIONS = 18
SHIFT = 7  # pixels; 3.1% of 224, mirrors CIFAR's 1-pixel relative scale


def get_query_specs() -> list[tuple[bool, int, int]]:
    """18 specs: reflect × 3 dx offsets × 3 dy offsets. Offsets in pixels."""
    offsets = (0, SHIFT, 2 * SHIFT)
    specs = []
    for reflect in (False, True):
        for dx in offsets:
            for dy in offsets:
                specs.append((reflect, dx, dy))
    assert len(specs) == NB_AUGMENTATIONS
    return specs


def apply_query_spec(padded: torch.Tensor, padded_flipped: torch.Tensor,
                     spec: tuple[bool, int, int], h: int, w: int) -> torch.Tensor:
    """Crop the appropriate padded tensor at (dx, dy).

    Same order as the CIFAR attack: flip → pad(reflect) → crop. We
    pre-compute both pads once per batch (`padded`, `padded_flipped`) and
    pick based on the spec to avoid re-padding 18 times.
    """
    reflect, dx, dy = spec
    src = padded_flipped if reflect else padded
    return src[:, :, dx:dx + h, dy:dy + w]


def setup_distributed(rank, world_size, address, port):
    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def worker(rank: int, world_size: int, ckpt_path: str, out_path: str):
    # torch.multiprocessing.spawn re-execs Python, so the child's sys.argv is
    # the original command line — which still contains --ckpt and --out.
    # Strip them before fastargs parses, since fastargs' argparse wrapper may
    # reject unknowns.
    filtered = [sys.argv[0]]
    skip_next = False
    for i, tok in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if tok in ("--ckpt", "--out"):
            skip_next = True
            continue
        filtered.append(tok)
    sys.argv = filtered

    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")

    setup_distributed(rank, world_size,
                      config["dist_params.address"], config["dist_params.port"])
    device = torch.device(f"cuda:{rank}")

    model = build_model(
        model_name=config["model_params.model_name"],
        num_classes=config["dataset.num_classes"],
        pretrained=False,
    ).to(device)
    state = torch.load(ckpt_path, map_location=f"cuda:{rank}")
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    use_amp = config["experiment_params.training_precision"] in {"bfloat16", "float16"}
    precision = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        config["experiment_params.training_precision"], torch.float32)

    # Deterministic center-crop loader; augmentations are generated on-GPU
    # from the resulting normalized tensor via reflect-pad + crop + flip.
    loader = build_test_loader(
        data_root=config["dataset.data_root"],
        beton_name=config["dataset.train_beton"],
        batch_size=config["dataset.batch_size"],
        num_workers=config["dataset.num_workers"],
        device=device,
        seed=config["experiment_params.seed"],
        distributed=True,
    )

    specs = get_query_specs()
    local_signal = torch.full((NUM_TRAIN, NB_AUGMENTATIONS), float("nan"),
                              dtype=torch.float32, device=device)

    pbar = tqdm(
        loader,
        desc=f"infer[r{rank}]",
        disable=rank != 0,
        dynamic_ncols=True,
        leave=False,
    )
    with torch.no_grad():
        for images, labels, indices in pbar:
            images = images.contiguous()
            h, w = images.shape[-2], images.shape[-1]
            padded = F.pad(images, (SHIFT, SHIFT, SHIFT, SHIFT), mode="reflect")
            padded_flipped = F.pad(images.flip(-1), (SHIFT, SHIFT, SHIFT, SHIFT), mode="reflect")
            lbl = labels.long().view(-1, 1)
            idx = indices.long()
            for k, spec in enumerate(specs):
                aug = apply_query_spec(padded, padded_flipped, spec, h, w)
                with autocast(device_type="cuda", dtype=precision, enabled=use_amp):
                    logits = model(aug).float()
                probs = torch.softmax(logits, dim=1)
                true_probs = probs.gather(1, lbl).squeeze(1).float()
                local_signal[idx, k] = true_probs

    # Merge across ranks: each rank filled a disjoint subset; per-index max
    # over ranks collapses NaN→-inf placeholders. With drop_last=False and
    # a beton covering [0, NUM_TRAIN), every row receives exactly one rank
    # write per aug.
    merged = local_signal.clone()
    merged[torch.isnan(merged)] = float("-inf")
    dist.all_reduce(merged, op=dist.ReduceOp.MAX)

    if rank == 0:
        missing = int((merged == float("-inf")).sum().item())
        if missing > 0:
            print(f"[warn] {missing} (sample, aug) cells had no assigned rank output", file=sys.stderr)
            merged[merged == float("-inf")] = float("nan")
        arr = merged.detach().cpu().numpy().astype(np.float32)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.save(out_path, arr)
        print(f"[infer] wrote {out_path}  shape={arr.shape}  nan={int(np.isnan(arr).sum())}")

    dist.destroy_process_group()


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ckpt", required=True)
    pre.add_argument("--out", required=True)
    known, remaining = pre.parse_known_args()

    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    sys.argv = [sys.argv[0]] + remaining
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")

    world_size = torch.cuda.device_count()
    expected = config["dist_params.gpus_per_run"]
    if world_size != expected:
        raise RuntimeError(f"Expected {expected} visible GPUs, found {world_size}.")

    mp.spawn(worker, args=(world_size, known.ckpt, known.out), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
