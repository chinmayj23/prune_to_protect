"""High-level training entry points.

``train_full`` trains one model on the full training set and returns test accuracy, the utility
number reported in the paper (full-data models, distinct from the 50%-subset RMIA target models).
"""

from __future__ import annotations

import torch

from ..configs import build_config, get_dataset_spec
from .engine import run_single_training, set_seed


def train_full(dataset: str, model: str | None = None, method: str = "standard", seed: int = 3407,
               epochs: int | None = None, use_compile: bool | None = None,
               output_dir=None, **overrides) -> dict:
    """Train one full-dataset model and report final/best test accuracy.

    Args mirror the CLI: dataset/model/method select the config; epochs/use_compile override common
    knobs; extra kwargs pass through to the ExptConfig.
    """
    spec = get_dataset_spec(dataset)
    ov = dict(overrides)
    if epochs is not None:
        ov["num_epochs"] = int(epochs)
    if use_compile is not None:
        ov["use_compile"] = bool(use_compile)

    cfg = build_config(dataset, model, method, **ov)
    cfg.evaluate_every_epoch = True

    from ..configs.methods.build import canonical_method
    if canonical_method(method) == "infomax":
        import numpy as np
        coreset = np.load(cfg.infomax_coreset_path)  # global train indices selected by InfoMax
        subset_indices = torch.as_tensor(coreset, dtype=torch.long, device=cfg.device)
    else:
        subset_indices = torch.arange(spec.train_size, dtype=torch.long, device=cfg.device)
    set_seed(seed)
    result = run_single_training(
        cfg, seed=seed, subset_indices=subset_indices,
        unit_name=f"full_{method}_seed{seed}", output_dir=output_dir,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accs = [a for a in result["test_accuracy"] if a == a]  # drop NaNs
    return {
        "dataset": dataset,
        "model": cfg.model_name,
        "method": method,
        "seed": seed,
        "epochs_run": len(result["test_accuracy"]),
        "final_test_accuracy": float(accs[-1]) if accs else float("nan"),
        "best_test_accuracy": float(max(accs)) if accs else float("nan"),
    }
