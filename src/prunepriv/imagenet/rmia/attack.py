"""
Offline RMIA attack with offline_a=1.0 and 18-augmentation queries.

Per-sample signals are loaded as [N, K] (K=18) from `signal_train.npy`.
For each augmentation k we compute a per-target detection score using the
same searchsorted trick as before, then average detection scores across
augmentations. This matches the RMIA paper's multi-query formulation
(arXiv:2312.03262v3, App. B.5) except that we keep the inner count as a
continuous fraction rather than thresholding per-(x,z) and then doing a
discrete majority vote — the continuous version dominates at low FPR
(never coarser) and, crucially, sidesteps the 1.28M × 640k × K outer
product that hard majority voting would require. Mathematically, this
corresponds to averaging the indicator 1[score_k(x,z) > gamma] across
both k and z instead of first thresholding the per-k mean.

Per target model t, augmentation k:
    score_k_t(x, z) = prob_ratio_x_k(x) * (1 / prob_ratio_z_k(z))
    prob_ratio_x_k(x) = target_signal_k_t[x] / mean_out_x_k[x]
    prob_ratio_z_k(z) = target_signal_k_t[z] / mean_ref_z_k[z]
    detection_k_t[x]  = mean_{z in non_members(t)} 1[score_k_t(x,z) > gamma]
    detection_t[x]    = mean_k detection_k_t[x]

Inputs
------
shadow_signals : [S, N, K] float32  -- shadow softmax-of-true-label per aug
shadow_keeps   : [S, N]    bool     -- membership mask per shadow
target_signals : [T, N, K] float32  -- target softmax-of-true-label per aug
target_keeps   : [T, N]    bool     -- membership mask per target

Outputs
-------
summary.json with per-target AUC / TPR@{1,0.1,0.01,0.001,0}%FPR and means.
target_<name>.npz with per-sample score, label, roc, tprs, auc, acc.
avg_detection_member.npy / any_member_mask.npy for downstream analysis.

Usage:
    python -m prunepriv.imagenet.rmia.attack \
        --shadow-dir <base>/<arch>_standard_shadow_bank \
        --target-dir <base>/<arch>_<method>_target_bank \
        --membership-dir <membership_dir> \
        --out-dir <out> \
        --gamma 2.0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve


NUM_TRAIN = 1281167
NB_AUGMENTATIONS = 18


def _list_runs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")])


def _load_signal(run_dir: Path) -> np.ndarray:
    path = run_dir / "signal_train.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing signal file: {path}. Run prunepriv.imagenet.rmia.infer first.")
    sig = np.load(path).astype(np.float32)
    if sig.shape != (NUM_TRAIN, NB_AUGMENTATIONS):
        raise ValueError(
            f"Bad signal shape {sig.shape} in {path}; expected {(NUM_TRAIN, NB_AUGMENTATIONS)}. "
            f"Re-run prunepriv.imagenet.rmia.infer to regenerate."
        )
    if np.isnan(sig).any():
        raise ValueError(f"NaNs in signal {path}: {int(np.isnan(sig).sum())}")
    return sig


def _load_keep(run_dir: Path, membership_root: Path, role: str) -> np.ndarray:
    keep_path = membership_root / role / run_dir.name / "keep.npy"
    if not keep_path.exists():
        raise FileNotFoundError(f"Missing keep mask: {keep_path}")
    keep = np.load(keep_path).astype(bool)
    if keep.shape != (NUM_TRAIN,):
        raise ValueError(f"Bad keep shape {keep.shape} in {keep_path}")
    return keep


TARGET_FPRS = (0.01, 0.001, 0.0001, 1e-5, 0.0)


def _tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray,
                target_fprs=TARGET_FPRS) -> tuple[dict, dict]:
    """Return (tpr_at_fpr, threshold_at_fpr) dicts keyed by target FPR."""
    tpr_out, thr_out = {}, {}
    for f in target_fprs:
        below = np.where(fpr <= f)[0]
        if below.size:
            i = below[-1]
            tpr_out[f] = float(tpr[i])
            thr_out[f] = float(thresholds[i])
        else:
            tpr_out[f] = 0.0
            thr_out[f] = float("inf")
    return tpr_out, thr_out


def _compute_means(shadow_signals: torch.Tensor, shadow_keeps: torch.Tensor):
    """Return (mean_out_x [N, K], mean_ref_z [N, K]).

    mean_out_x uses only shadows where sample is a non-member (offline a=1.0).
    mean_ref_z uses all shadows (for the z denominator).
    """
    eps = 1e-8
    # shadow_keeps: [S, N]; expand to [S, N, 1] for broadcasting over K.
    out_mask = (~shadow_keeps).float().unsqueeze(-1)  # [S, N, 1]
    out_counts = out_mask.sum(dim=0).clamp(min=1.0)  # [N, 1]
    mean_out_x = (shadow_signals * out_mask).sum(dim=0) / out_counts  # [N, K]
    mean_ref_z = shadow_signals.mean(dim=0)  # [N, K]
    return mean_out_x.clamp(min=eps), mean_ref_z.clamp(min=eps)


def _detection_per_target(
    target_signal: torch.Tensor,    # [N, K]
    target_keep: torch.Tensor,      # [N] bool
    mean_out_x: torch.Tensor,       # [N, K]
    mean_ref_z: torch.Tensor,       # [N, K]
    gamma: float,
) -> torch.Tensor:
    """Mean over augmentations of per-aug detection fractions. Returns [N]."""
    eps = 1e-8
    K = target_signal.shape[1]
    pop_mask = ~target_keep  # [N]
    N = target_signal.shape[0]

    # Accumulate detection over augs in float64 for numerical stability.
    accum = torch.zeros(N, dtype=torch.float64, device=target_signal.device)

    for k in range(K):
        t_sig = target_signal[:, k]                        # [N]
        pr_x = t_sig / mean_out_x[:, k]                    # [N]
        inv_pr_z = mean_ref_z[:, k] / t_sig.clamp(min=eps) # [N]
        pop_inv_pr_z = inv_pr_z[pop_mask]                  # [P]
        pop_sorted, _ = torch.sort(pop_inv_pr_z)
        P = pop_sorted.numel()
        thresh = gamma / pr_x.clamp(min=eps)               # [N]
        idx = torch.searchsorted(pop_sorted, thresh, right=True)  # [N]
        det_k = (P - idx).double() / max(P, 1)
        accum += det_k

    return (accum / K).float()  # [N]


def run_attack(
    shadow_dir: Path,
    target_dir: Path,
    membership_dir: Path,
    out_dir: Path,
    gamma: float,
    device: str = "cuda",
) -> dict:
    shadow_runs = _list_runs(shadow_dir)
    target_runs = _list_runs(target_dir)
    if not shadow_runs or not target_runs:
        raise RuntimeError(f"No run_* dirs under {shadow_dir} or {target_dir}")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    shadow_signals = torch.from_numpy(
        np.stack([_load_signal(r) for r in shadow_runs])
    ).to(dev)  # [S, N, K]
    shadow_keeps = torch.from_numpy(
        np.stack([_load_keep(r, membership_dir, "shadows") for r in shadow_runs])
    ).to(dev)  # [S, N]
    target_signals = torch.from_numpy(
        np.stack([_load_signal(r) for r in target_runs])
    ).to(dev)  # [T, N, K]
    target_keeps = torch.from_numpy(
        np.stack([_load_keep(r, membership_dir, "targets") for r in target_runs])
    ).to(dev)  # [T, N]

    print(f"[attack] shadows={tuple(shadow_signals.shape)} "
          f"targets={tuple(target_signals.shape)} gamma={gamma}")

    mean_out_x, mean_ref_z = _compute_means(shadow_signals, shadow_keeps)  # [N, K]

    per_target_aucs, per_target_tprs, per_target_accs = [], [], []
    sum_det_member = torch.zeros(NUM_TRAIN, device=dev, dtype=torch.float64)
    count_member = torch.zeros(NUM_TRAIN, device=dev, dtype=torch.float64)
    sum_det_nonmember = torch.zeros(NUM_TRAIN, device=dev, dtype=torch.float64)
    count_nonmember = torch.zeros(NUM_TRAIN, device=dev, dtype=torch.float64)

    # For pooled ROC: concatenate (score, label) across all targets.
    pooled_scores: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []

    out_dir.mkdir(parents=True, exist_ok=True)

    for t, run in enumerate(target_runs):
        det = _detection_per_target(
            target_signals[t], target_keeps[t], mean_out_x, mean_ref_z, gamma,
        )
        score = det.detach().cpu().numpy()
        labels = target_keeps[t].detach().cpu().numpy().astype(bool)

        fpr, tpr, thr = roc_curve(labels, score)
        roc_auc = float(auc(fpr, tpr))
        acc = float(np.max(1 - (fpr + (1 - tpr)) / 2))
        tprs, thr_at_fpr = _tpr_at_fpr(fpr, tpr, thr)
        per_target_aucs.append(roc_auc)
        per_target_accs.append(acc)
        per_target_tprs.append(tprs)

        # Per-target inferred masks at each FPR operating point.
        # inferred_at_f[n] = True iff this run would flag sample n as a member
        # at the threshold achieving FPR <= f on this target's ROC.
        inferred_masks = {f: (score >= thr_at_fpr[f]) for f in TARGET_FPRS}

        member_mask = target_keeps[t].double()
        nonmember_mask = 1.0 - member_mask
        sum_det_member += det.double() * member_mask
        count_member += member_mask
        sum_det_nonmember += det.double() * nonmember_mask
        count_nonmember += nonmember_mask

        pooled_scores.append(score)
        pooled_labels.append(labels)

        save_payload = dict(
            score=score, labels=labels,
            fpr=fpr, tpr=tpr, thresholds=thr,
            auc=roc_auc, acc=acc,
            tprs=np.array(list(tprs.values())),
            tpr_fpr_keys=np.array(list(tprs.keys()), dtype=np.float64),
            thr_at_fpr=np.array(list(thr_at_fpr.values())),
        )
        for f, mask in inferred_masks.items():
            key = f"inferred_at_fpr_{f:.0e}" if f > 0 else "inferred_at_fpr_0"
            save_payload[key] = mask
        np.savez(out_dir / f"target_{run.name}.npz", **save_payload)
        print(f"[attack] {run.name}  AUC={roc_auc:.4f}  acc={acc:.4f}  "
              f"TPR@1%FPR={tprs[0.01]:.4f}  TPR@0.1%FPR={tprs[0.001]:.4f}")

    avg_det_member = (sum_det_member / count_member.clamp(min=1.0)).cpu().numpy().astype(np.float32)
    any_member = (count_member > 0).cpu().numpy()
    avg_det_nonmember = (sum_det_nonmember / count_nonmember.clamp(min=1.0)).cpu().numpy().astype(np.float32)
    any_nonmember = (count_nonmember > 0).cpu().numpy()
    np.save(out_dir / "avg_detection_member.npy", avg_det_member)
    np.save(out_dir / "any_member_mask.npy", any_member)
    np.save(out_dir / "avg_detection_nonmember.npy", avg_det_nonmember)
    np.save(out_dir / "any_nonmember_mask.npy", any_nonmember)

    # Pooled ROC: concatenate all (target, sample) pairs. This is the
    # paper-standard RMIA curve and differs from the mean-of-per-target-AUC.
    pool_score = np.concatenate(pooled_scores)
    pool_label = np.concatenate(pooled_labels)
    p_fpr, p_tpr, p_thr = roc_curve(pool_label, pool_score)
    pool_auc = float(auc(p_fpr, p_tpr))
    pool_acc = float(np.max(1 - (p_fpr + (1 - p_tpr)) / 2))
    pool_tprs, pool_thr_at_fpr = _tpr_at_fpr(p_fpr, p_tpr, p_thr)
    np.savez(out_dir / "pooled_roc.npz",
             fpr=p_fpr, tpr=p_tpr, thresholds=p_thr,
             auc=pool_auc, acc=pool_acc,
             tprs=np.array(list(pool_tprs.values())),
             tpr_fpr_keys=np.array(list(pool_tprs.keys()), dtype=np.float64),
             thr_at_fpr=np.array(list(pool_thr_at_fpr.values())))

    summary = {
        "gamma": gamma,
        "num_shadows": shadow_signals.shape[0],
        "num_targets": target_signals.shape[0],
        "num_augmentations": int(shadow_signals.shape[2]),
        "mean_auc": float(np.mean(per_target_aucs)),
        "mean_acc": float(np.mean(per_target_accs)),
        "mean_tpr_at_fpr": {str(k): float(np.mean([d[k] for d in per_target_tprs]))
                            for k in per_target_tprs[0]},
        "per_target_auc": per_target_aucs,
        "pooled_auc": pool_auc,
        "pooled_acc": pool_acc,
        "pooled_tpr_at_fpr": {str(k): float(v) for k, v in pool_tprs.items()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[attack] mean AUC={summary['mean_auc']:.4f}  "
          f"mean TPR@1%FPR={summary['mean_tpr_at_fpr']['0.01']:.4f}  "
          f"mean TPR@0.1%FPR={summary['mean_tpr_at_fpr']['0.001']:.4f}")
    print(f"[attack] pooled AUC={pool_auc:.4f}  "
          f"pooled TPR@1%FPR={pool_tprs[0.01]:.4f}  "
          f"pooled TPR@0.1%FPR={pool_tprs[0.001]:.4f}")
    print(f"[attack] wrote: {out_dir}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shadow-dir", required=True)
    p.add_argument("--target-dir", required=True)
    p.add_argument("--membership-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run_attack(
        shadow_dir=Path(args.shadow_dir),
        target_dir=Path(args.target_dir),
        membership_dir=Path(args.membership_dir),
        out_dir=Path(args.out_dir),
        gamma=args.gamma,
        device=args.device,
    )


if __name__ == "__main__":
    main()
