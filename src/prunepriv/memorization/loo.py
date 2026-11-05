"""Subset-LOO memorization scores.

The estimator is:

    memorization(x) = mean_correct(models trained WITH x) - mean_correct(models trained WITHOUT x)

over a bank of ``num_runs`` models each trained on a random subset, using the subset-membership mask.

Reuses the engine's ``CifarLoader`` and the model factory, so scoring matches how the checkpoints were
trained. Pass ``model_factory`` for non-standard architectures.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from ..configs import build_config, get_dataset_spec, build_model
from ..training.engine import CifarLoader, set_seed


def _load_state_dict_safely(model, checkpoint_path, device):
    sd = torch.load(checkpoint_path, map_location=device)

    def strip(k):
        changed = True
        while changed:
            changed = False
            for pre in ("_module.", "_orig_mod."):
                if k.startswith(pre):
                    k = k[len(pre):]
                    changed = True
        return k

    if any(k.startswith(("_orig_mod.", "_module.")) for k in sd):
        sd = {strip(k): v for k, v in sd.items()}
    model.load_state_dict(sd)


@torch.no_grad()
def _all_correctness(model, loader, device, use_aug: bool = True) -> torch.Tensor:
    model.eval()
    out = []
    # image_pt loaders augment in .iterate(); for memorization scoring we want DETERMINISTIC eval
    # preprocessing (centre-crop, no flip) so correctness is reproducible -> iterate_wo_aug.
    stream = loader.iterate(None) if use_aug else loader.iterate_wo_aug(None)
    for batch in stream:
        inputs, targets = batch[0], batch[1]
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model(inputs).argmax(1)
        out.append(preds == targets)
    return torch.cat(out)


def compute_loo_scores(runs_dir, num_runs: int, epochs, subset_file, dataset: str,
                       model: str | None = None, model_factory=None, data_root: str | None = None,
                       batch_size: int = 512, out_dir=None, base_seed: int = 42) -> dict:
    """Compute LOO memorization for each epoch. Returns {epoch: {memorization, indices, labels}}."""
    runs_dir = Path(runs_dir)
    spec = get_dataset_spec(dataset)
    cfg = build_config(dataset, model, "standard")  # data_root + model token
    device = cfg.device
    data_root = data_root or cfg.data_root_dir

    set_seed(base_seed)
    image_pt = spec.loader_backend == "image_pt"
    if image_pt:
        from ..data.image_loader import ImageLoader
        loader = ImageLoader(path=data_root, dataset_name=spec.name, train=True,
                             batch_size=batch_size, shuffle=False, mean=spec.mean, std=spec.std,
                             crop_size=int(spec.input_size), device=device)
    else:
        loader = CifarLoader(path=data_root, dataset_name=spec.name, train=True,
                             batch_size=batch_size, shuffle=False, noise_type="clean", device=device)
    n_train = len(loader.images)

    subsets = torch.load(subset_file)
    subset_iter = list(subsets.values()) if isinstance(subsets, dict) else list(subsets)
    mask = torch.zeros(num_runs, n_train, dtype=torch.bool, device=device)
    for i, idx in enumerate(subset_iter[:num_runs]):
        mask[i, idx] = True

    net = model_factory() if model_factory is not None else build_model(cfg.model_name, spec.num_classes)
    net.to(device)

    results = {}
    for e in epochs:
        correctness = torch.zeros(num_runs, n_train, dtype=torch.bool, device=device)
        for run_id in range(num_runs):
            ckpt = runs_dir / f"run_{run_id}" / f"epoch_{e}.pt"
            if not ckpt.exists():
                continue  # missing run -> counted as all-incorrect
            _load_state_dict_safely(net, ckpt, device)
            correctness[run_id] = _all_correctness(net, loader, device, use_aug=not image_pt)

        m_f = mask.to(torch.float64)
        inv_f = (~mask).to(torch.float64)
        c_f = correctness.to(torch.float64)

        def masked_avg(x, mk):
            num = (x * mk).sum(0, dtype=torch.float64)
            den = mk.sum(0, dtype=torch.float64).clamp(min=1e-20)
            return num / den

        mem = (masked_avg(c_f, m_f) - masked_avg(c_f, inv_f)).cpu()
        indices = torch.arange(n_train)
        labels = loader.labels.cpu()
        results[e] = {"memorization": mem, "indices": indices, "labels": labels}
        if out_dir is not None:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(results[e], out / f"loo_scores_epoch_{e}.pt")
    return results
