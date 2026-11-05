#!/usr/bin/env python
"""LOO (subset) memorization for Waterbirds, tracked across epochs.

Trains standard / infobatch / ib_prw banks on balanced 50% subsets and computes, for each tracked
epoch e (default every 20: epochs 19,39,59,79,99, plus 87 and 88):

    memorization_e(x) = mean_correct_e(models WITH x) - mean_correct_e(models WITHOUT x)

Each model is trained, its per-sample correctness at the tracked epochs is cached
(`run_<i>/correctness.npy`, ~24 KB), and its checkpoints are deleted, so peak disk is about one
model's checkpoints. A run whose `correctness.npy` exists is skipped, so interrupted jobs resume.

Output: `waterbirds_loo_scores.pt` = {method: {memorization [n_epochs, 4795], epochs, labels, groups}}.

    python studies/waterbirds_loo.py --num-runs 200 --epochs 100 --score-every 20
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "studies"))

import numpy as np
import torch

from prunepriv import paths
from prunepriv.configs import build_config, get_dataset_spec, build_model
from prunepriv.training.engine import run_single_training, set_seed
from prunepriv.attacks.runner import generate_keep_matrix
from prunepriv.data.image_loader import ImageLoader

DATASET = "waterbirds"
BASE_SEED = 42
KEEP = 0.5  # each sample IN exactly num_runs/2 models -> every sample has IN and OUT models


def _strip(sd):
    while any(k.startswith(("_orig_mod.", "_module.")) for k in sd):
        sd = {k.split(".", 1)[1] if k.startswith(("_orig_mod.", "_module.")) else k: v
              for k, v in sd.items()}
    return sd


@torch.no_grad()
def _correctness(net, loader, device):
    net.eval()
    out = []
    for batch in loader.iterate_wo_aug(None):          # eval preprocessing, sample order
        xb, yb = batch[0], batch[1]
        out.append((net(xb.to(device)).argmax(1) == yb.to(device)).cpu())
    return torch.cat(out)                              # bool [n_train], in sample order


def _eval_tracked(run_dir, tracked, model_token, num_classes, loader, device):
    net = build_model(model_token, num_classes=num_classes).to(device)
    n = len(loader.images)
    res = np.zeros((len(tracked), n), dtype=bool)
    for i, e in enumerate(tracked):
        ck = run_dir / f"epoch_{e}.pt"
        if ck.exists():
            net.load_state_dict(_strip(torch.load(ck, map_location=device)))
            res[i] = _correctness(net, loader, device).numpy()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18_224")
    ap.add_argument("--methods", nargs="+", default=["standard", "infobatch", "ib_prw"])
    ap.add_argument("--num-runs", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--score-every", type=int, default=20,
                    help="track memorization every N epochs (plus the final epoch)")
    ap.add_argument("--extra-epochs", nargs="+", type=int, default=[87, 88],
                    help="always track these epochs too (default 87,88 = WLIB annealing switch for "
                         "delta=0.875, epochs=100: pruning ends at 87, WLIB weighting starts at 88)")
    ap.add_argument("--keep-ckpts", action="store_true", help="keep checkpoints (default: delete after scoring)")
    ap.add_argument("--out", default=None, help="output dir (default <repo>/results/studies/waterbirds_loo)")
    args = ap.parse_args()
    tracked = sorted((set(range(args.score_every - 1, args.epochs, args.score_every))
                      | {args.epochs - 1} | {e for e in args.extra_epochs if 0 <= e < args.epochs}))

    spec = get_dataset_spec(DATASET)
    n = spec.train_size
    out = Path(args.out) if args.out else paths.results_dir("studies", "waterbirds_loo")
    out.mkdir(parents=True, exist_ok=True)

    # auto-prepare data (download + blob) if missing -> single command from a fresh clone
    if not paths.data_dir(DATASET, f"{DATASET}_train.pt").exists():
        print("waterbirds blobs missing -> downloading + preparing ...", flush=True)
        import prepare_waterbirds
        prepare_waterbirds.main()

    # shared balanced subsets (same across methods so memorization is comparable)
    keep = generate_keep_matrix(args.num_runs, n, KEEP, BASE_SEED)      # [num_runs, n] bool = IN mask
    torch.save({i: torch.from_numpy(np.flatnonzero(keep[i])) for i in range(args.num_runs)},
               out / "subsets.pt")
    print(f"subsets: {args.num_runs} models, each sample IN {int(keep[:, 0].sum())}; "
          f"tracked epochs {tracked}", flush=True)

    cfg0 = build_config(DATASET, args.model, "standard")
    device = cfg0.device
    model_token = cfg0.model_name
    eval_loader = ImageLoader(path=cfg0.data_root_dir, dataset_name=DATASET, train=True,
                              batch_size=512, shuffle=False, mean=spec.mean, std=spec.std,
                              crop_size=int(spec.input_size), device=device)
    groups = torch.load(paths.data_dir(DATASET, f"{DATASET}_train.pt"),
                        map_location="cpu", weights_only=False)["groups"]

    m_in = keep.astype(np.float64)                     # [R, n]
    m_out = (~keep).astype(np.float64)
    all_scores = {}
    for method in args.methods:
        runs_dir = out / method / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        corr = np.zeros((args.num_runs, len(tracked), n), dtype=bool)
        for run_id in range(args.num_runs):
            run_dir = runs_dir / f"run_{run_id}"
            cache = run_dir / "correctness.npy"
            if cache.exists():
                corr[run_id] = np.load(cache)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = build_config(DATASET, args.model, method)
            cfg.num_epochs = int(args.epochs)
            cfg.save_epoch_checkpoints = True
            cfg.sample_at_epoch = list(tracked)        # save run_dir/epoch_<e>.pt at tracked epochs
            cfg.save_last_checkpoint = False
            subset = np.flatnonzero(keep[run_id])
            run_seed = BASE_SEED + run_id
            set_seed(run_seed)
            subset_dev = torch.as_tensor(subset, dtype=torch.long, device=device)
            run_single_training(cfg, seed=run_seed, subset_indices=subset_dev,
                                unit_name=f"loo_{method}_run{run_id}", output_dir=run_dir)
            c = _eval_tracked(run_dir, tracked, model_token, spec.num_classes, eval_loader, device)
            np.save(cache, c)
            corr[run_id] = c
            if not args.keep_ckpts:                     # storage-lean: bits cached, drop checkpoints
                for e in tracked:
                    (run_dir / f"epoch_{e}.pt").unlink(missing_ok=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{method}] run {run_id+1}/{args.num_runs} (subset {subset.size})", flush=True)

        # memorization per tracked epoch
        mem = np.zeros((len(tracked), n), dtype=np.float64)
        for ei in range(len(tracked)):
            c = corr[:, ei, :].astype(np.float64)       # [R, n]
            avg_in = (c * m_in).sum(0) / m_in.sum(0).clip(min=1e-20)
            avg_out = (c * m_out).sum(0) / m_out.sum(0).clip(min=1e-20)
            mem[ei] = avg_in - avg_out
        all_scores[method] = {"memorization": torch.from_numpy(mem), "epochs": list(tracked),
                                "labels": eval_loader.labels.cpu(), "groups": groups}
        torch.save(all_scores[method], out / method / f"loo_scores_{method}.pt")
        print(f"[{method}] final-epoch mean mem {mem[-1].mean():.4f}  "
              f"memorized(>0.1) {int((mem[-1] > 0.1).sum())}/{n}", flush=True)

    final = out / "waterbirds_loo_scores.pt"
    torch.save(all_scores, final)
    print(f"\nFinal scores -> {final}")
    print(f"per method: memorization[{len(tracked)} epochs x {n}], epochs={tracked}, labels, "
          f"groups (0..3 = LL/LW/WL/WW)")


if __name__ == "__main__":
    main()
