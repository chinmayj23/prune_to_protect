"""
Generate shadow/target membership masks for RMIA.

Each sample is present in exactly half of the shadow models and exactly half of
the target models. Shadows and targets are generated independently; the same
sample can be member in any combination of (shadow_j, target_k) pairs.

Output layout (under --out-dir):
    shadows/keep.npy                # bool [num_shadows, num_train]
    shadows/run_{i:03d}/keep.npy    # bool [num_train] — per-run include mask
    shadows/run_{i:03d}/indices.pt  # LongTensor of included sample ids (sorted)
    targets/keep.npy                # bool [num_targets, num_train]
    targets/run_{i:03d}/keep.npy
    targets/run_{i:03d}/indices.pt
    meta.json

The "standard" target bank re-uses `shadows/`: standard targets use the same
runs as standard shadows. For infobatch and ib_prw_norm, a fresh
targets bank is generated under --out-dir/targets.

When --num-shadows == --num-targets and --same-seed is set, shadow and target
banks will be *different* random partitions (different sub-seeds) — we never
want identical membership for targets and shadows of the same method.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def build_exact_half_masks(num_samples: int, num_runs: int, seed: int) -> np.ndarray:
    """
    Return a bool matrix [num_runs, num_samples] where each column has exactly
    num_runs // 2 True entries. For num_runs=4 each sample appears in 2 runs.

    Per-sample membership is independently permuted so runs are uncorrelated
    across samples (but every sample hits the 50/50 balance exactly).
    """
    if num_runs < 2 or num_runs % 2 != 0:
        raise ValueError(f"num_runs must be even and >=2, got {num_runs}")

    half = num_runs // 2
    rng = np.random.default_rng(seed)

    keep = np.zeros((num_runs, num_samples), dtype=bool)
    base = np.concatenate([np.ones(half, dtype=bool), np.zeros(num_runs - half, dtype=bool)])
    # Per-sample independent permutation.
    for s in range(num_samples):
        perm = rng.permutation(num_runs)
        keep[:, s] = base[perm]

    # Invariants.
    assert keep.sum(axis=0).min() == half and keep.sum(axis=0).max() == half, \
        "Per-sample half-membership invariant broken"
    # Each run should be ~50% of the dataset (within rng noise).
    return keep


def save_bank(out_dir: Path, keep: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "keep.npy", keep)
    num_runs = keep.shape[0]
    for i in range(num_runs):
        run_dir = out_dir / f"run_{i:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        mask = keep[i]
        np.save(run_dir / "keep.npy", mask)
        indices = torch.from_numpy(np.flatnonzero(mask)).long()
        torch.save(indices, run_dir / "indices.pt")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-train", type=int, default=1281167)
    p.add_argument("--num-shadows", type=int, default=4)
    p.add_argument("--num-targets", type=int, default=4)
    p.add_argument("--seed-shadows", type=int, default=2025)
    p.add_argument("--seed-targets", type=int, default=4051)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating shadow bank: {args.num_shadows} runs × {args.num_train} samples")
    shadow_keep = build_exact_half_masks(args.num_train, args.num_shadows, args.seed_shadows)
    save_bank(args.out_dir / "shadows", shadow_keep)

    print(f"Generating target bank: {args.num_targets} runs × {args.num_train} samples")
    target_keep = build_exact_half_masks(args.num_train, args.num_targets, args.seed_targets)
    save_bank(args.out_dir / "targets", target_keep)

    meta = {
        "num_train": int(args.num_train),
        "num_shadows": int(args.num_shadows),
        "num_targets": int(args.num_targets),
        "seed_shadows": int(args.seed_shadows),
        "seed_targets": int(args.seed_targets),
        "per_sample_shadow_members": int(args.num_shadows // 2),
        "per_sample_target_members": int(args.num_targets // 2),
        "shadow_run_sizes": shadow_keep.sum(axis=1).tolist(),
        "target_run_sizes": target_keep.sum(axis=1).tolist(),
    }
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved membership to {args.out_dir}")
    print(f"  Shadow per-run sizes: {meta['shadow_run_sizes']}")
    print(f"  Target per-run sizes: {meta['target_run_sizes']}")


if __name__ == "__main__":
    main()
