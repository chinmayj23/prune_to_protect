"""CLI: end-to-end RMIA/LiRA membership-inference attack.

    prunepriv-attack --dataset cifar10 --model resnet20 --target-method wlib \
        --attack rmia --online --defense-aware
    python scripts/attack.py --dataset cifar10 --target-method infobatch --attack lira --offline

Trains shadow + target banks (or reuses existing ones), exports logits, runs the attack, and reports
pooled AUC. Banks live under PRUNEPRIV_BANKS, results under PRUNEPRIV_RESULTS.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from ..configs import available_datasets, available_methods
from ..attacks.runner import run_attack_pipeline


def pooled_auc(report_dir: str):
    from sklearn.metrics import roc_auc_score
    scores, ans = [], []
    for f in glob.glob(os.path.join(report_dir, "attack_predictions_*.npz")):
        z = np.load(f)
        scores.append(z["scores"].ravel())
        ans.append(z["answers"].ravel())
    if not scores:
        return None
    s = np.concatenate(scores)
    a = np.concatenate(ans)
    return float(roc_auc_score(a.astype(int), -s))  # lower score = member -> use -score


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end RMIA/LiRA attack.")
    p.add_argument("--dataset", required=True, choices=available_datasets())
    p.add_argument("--model", default=None)
    p.add_argument("--target-method", "--method", dest="target_method", default="standard",
                   help=f"target training method (one of: {', '.join(available_methods())})")
    p.add_argument("--shadow-method", default="standard", help="shadow training method")
    p.add_argument("--defense-aware", action="store_true",
                   help="use method-matched shadows (sets shadow-method = target-method)")
    p.add_argument("--attack", choices=["rmia", "lira"], default="rmia")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--online", dest="online", action="store_true", default=True)
    grp.add_argument("--offline", dest="online", action="store_false")
    p.add_argument("--num-shadows", type=int, default=64)
    p.add_argument("--num-targets", type=int, default=1)
    p.add_argument("--epochs", type=int, default=None, help="override training epochs (banks)")
    p.add_argument("--pretrained", action="store_true",
                   help="train bank models from ImageNet-pretrained weights (fine-tune); banks/reports "
                        "get a _pt suffix so they never collide with the scratch banks")
    p.add_argument("--lr", type=float, default=None, help="override bank learning rate (e.g. 1e-3 for PT)")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="RMIA signal temperature; scales with logit magnitude (2.0 for scratch, "
                        "~0.25 for pretrained models whose logits are smaller)")
    p.add_argument("--population-cap", type=int, default=-1,
                   help="cap the OUT reference population to this many samples (RMIA Monte-Carlo); "
                        "bounds memory on large datasets like CelebA. -1 = off (default)")
    p.add_argument("--no-compile", action="store_true", help="disable torch.compile (faster for small banks)")
    p.add_argument("--force", action="store_true", help="retrain even if bank checkpoints exist")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    shadow_method = args.target_method if args.defense_aware else args.shadow_method
    res = run_attack_pipeline(
        dataset=args.dataset, model=args.model, target_method=args.target_method,
        shadow_method=shadow_method, attack=args.attack, online=args.online,
        num_shadows=args.num_shadows, num_targets=args.num_targets, epochs=args.epochs,
        force=args.force, use_compile=(False if args.no_compile else None),
        pretrained=args.pretrained, lr=args.lr, temperature=args.temperature,
        population_cap=args.population_cap,
    )
    auc = pooled_auc(res["report_dir"])
    print(f"\n=== {res['tag']} ===")
    print(f"defense_aware={res['defense_aware']} attack={res['attack']} "
          f"threat={'online' if res['online'] else 'offline'}")
    print(f"pooled AUC = {auc:.4f}" if auc is not None else "pooled AUC = (no predictions found)")
    print(f"results: {res['results_dir']}")


if __name__ == "__main__":
    main()
