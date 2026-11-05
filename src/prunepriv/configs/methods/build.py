"""Build a fully-configured ExptConfig for (dataset, model, method).

The method is selected by string, the dataset facts come from the ``DatasetSpec``, and the per-dataset
training defaults come from ``TRAIN_DEFAULTS``. New datasets add a ``TRAIN_DEFAULTS`` entry.
"""

from __future__ import annotations

import os

import torch

from ... import paths
from ..base import (
    Cifar10StandardConfig,
    Cifar10IBConfig,
    Cifar10IBPRWConfig,
)
from ..datasets import DatasetSpec

# Per-dataset training defaults (paper Table 3 for CIFAR-10 and CIFAR-100).
TRAIN_DEFAULTS: dict[str, dict] = {
    "cifar10": dict(
        batch_size=512, learning_rate=0.2, weight_decay=5e-4, momentum=0.9, num_epochs=100,
        infobatch_ratio=0.5, infobatch_delta=0.875,
        train_aug={"flip": True, "translate": 2}, altflip=False, use_compile=True, base_seed=42,
    ),
    # CIFAR-100 / WRN-28-2.
    "cifar100": dict(
        batch_size=128, learning_rate=0.1, weight_decay=3e-4, momentum=0.9, num_epochs=100,
        infobatch_ratio=0.5, infobatch_delta=0.875,
        train_aug={"flip": True, "translate": 2}, altflip=False, use_compile=True, base_seed=42,
    ),
    # Waterbirds / light ResNet (CNN) from scratch. SGD (CNNs train fine from scratch). image_pt loader.
    "waterbirds": dict(
        batch_size=128, learning_rate=0.1, weight_decay=5e-4, momentum=0.9, num_epochs=100,
        infobatch_ratio=0.5, infobatch_delta=0.875,
        train_aug={"crop": True, "flip": True}, altflip=False, use_compile=False, base_seed=42,
        optimizer="sgd", pretrained=False,
    ),
    # CelebA / light ResNet from scratch (162k imgs -> fewer epochs; CPU-resident blob). Blond_Hair
    # target, Male spurious. SGD; RandomCrop(112 from 128)+flip. image_pt loader.
    "celeba": dict(
        batch_size=256, learning_rate=0.1, weight_decay=5e-4, momentum=0.9, num_epochs=50,
        infobatch_ratio=0.5, infobatch_delta=0.875,
        train_aug={"crop": True, "flip": True}, altflip=False, use_compile=True, base_seed=42,
        optimizer="sgd", pretrained=False,
    ),
}

# user-facing model name -> engine model_name token (engine dispatches on r18/r20/r50/wrn28_2/...)
_MODEL_TOKEN = {
    "resnet20": "r20", "r20": "r20",
    "resnet18": "r18", "r18": "r18",
    "resnet50": "r50", "r50": "r50",
    "wrn28_2": "wrn28_2",
    "deit_small": "deit_small", "deit_tiny": "deit_tiny",
    "resnet10t": "resnet10t", "resnet14t": "resnet14t", "resnet18_224": "resnet18_224",
    "resnet34_224": "resnet34_224",
    "resnet50_224": "resnet50_224", "wrn50_2_224": "wrn50_2_224",
}

# Methods available from the CLI: the paper's three methods, its ablations, and the InfoMax baseline.
VERIFIED_METHODS = {
    "standard", "infobatch", "ib_prw",
    "ib_noanneal", "ib_earlystop_cooled",
    "wlib_loss", "wlib_shuffle", "wlib_hardconst",
    "infomax",  # InfoMax static coreset: standard training on a precomputed coreset (coreset-index file)
}

_ALIASES = {
    "wlib": "ib_prw", "ib": "infobatch",
    "ib_earlystop": "ib_earlystop_cooled",
}


def canonical_method(method: str) -> str:
    m = str(method).strip().lower().replace("-", "_")
    return _ALIASES.get(m, m)


def _detect_precision() -> str:
    if not torch.cuda.is_available():
        return "float32"
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float32"


def _configure(cfg, spec: DatasetSpec, model_token: str, d: dict, data_root: str):
    """Apply the dataset spec and per-dataset defaults to ``cfg``."""
    cfg.data_root_dir = data_root
    cfg.dataset_name = spec.name
    cfg.batch_size = d["batch_size"]
    cfg.num_classes = spec.num_classes
    cfg.learning_rate = d["learning_rate"]
    cfg.weight_decay = d["weight_decay"]
    cfg.momentum = d["momentum"]
    cfg.optimizer_name = d.get("optimizer", "sgd")
    cfg.model_name = model_token
    cfg.num_epochs = d["num_epochs"]
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(getattr(cfg, "training_precision", "auto")).lower() == "auto":
        cfg.training_precision = _detect_precision()
    cfg.use_compile = d["use_compile"]
    cfg.base_seed = d["base_seed"]
    cfg.seed_list = [d["base_seed"]]
    cfg.train_aug = d["train_aug"]
    cfg.altflip = d["altflip"]
    cfg.noise_type = "clean"
    cfg.is_loo = False
    cfg.is_loo_with_partial_subsets = False
    cfg.sample_at_epoch = []
    cfg.evaluate_every_epoch = True
    cfg.log_sample_evolution = False
    cfg.log_evolution_activations = False
    cfg.log_evolution_grad_norms = False
    cfg.use_fixed_indices_for_analysis = False
    cfg.log_noise_splits = False
    cfg.log_wandb = False
    cfg.save_outputs = False
    cfg.save_best_checkpoint = False
    cfg.save_last_checkpoint = False
    cfg.save_epoch_checkpoints = False
    cfg.use_ib_sampler = False
    cfg.ratio = d["infobatch_ratio"]
    cfg.delta = d["infobatch_delta"]
    cfg.experiment_name = None
    cfg.non_loo_root = None
    cfg.loo_root = None
    cfg.loo_output_path = None
    # image datasets (non-CIFAR): loader backend + normalization + input size + pretrained flag
    cfg.loader_backend = spec.loader_backend
    cfg.norm_mean = spec.mean
    cfg.norm_std = spec.std
    cfg.input_size = spec.input_size
    cfg.loader_blob_device = spec.loader_blob_device or None
    cfg.pretrained = d.get("pretrained", False)
    return cfg


def build_method_config(spec: DatasetSpec, model: str, method: str, data_root: str, **overrides):
    """Return a fully-configured ExptConfig instance for (dataset, model, method)."""
    method = canonical_method(method)
    if spec.name not in TRAIN_DEFAULTS:
        raise NotImplementedError(
            f"No training defaults for dataset {spec.name!r}. "
            f"Have: {sorted(TRAIN_DEFAULTS)}."
        )
    if method not in VERIFIED_METHODS:
        raise KeyError(f"Unknown method {method!r}. Known: {sorted(VERIFIED_METHODS)}.")

    model_key = str(model).strip().lower()
    if model_key not in _MODEL_TOKEN:
        raise KeyError(f"Unknown model {model!r}. Known: {sorted(set(_MODEL_TOKEN))}.")
    model_token = _MODEL_TOKEN[model_key]
    d = dict(TRAIN_DEFAULTS[spec.name])

    # base config class per method
    if method == "standard":
        cfg = Cifar10StandardConfig()
    elif method == "infobatch":
        cfg = Cifar10IBConfig()
    elif method == "ib_noanneal":
        cfg = Cifar10IBConfig()
    elif method == "ib_earlystop_cooled":
        cfg = Cifar10IBConfig()
    elif method == "ib_prw":
        cfg = Cifar10IBPRWConfig()
    elif method in {"wlib_loss", "wlib_shuffle", "wlib_hardconst"}:
        cfg = Cifar10IBPRWConfig()
    elif method == "infomax":
        cfg = Cifar10StandardConfig()  # standard training; the coreset restricts the training subset
    else:  # unreachable (guarded above)
        raise KeyError(method)

    _configure(cfg, spec, model_token, d, data_root)
    if method == "infomax":
        # coreset-index file (global indices into the train order); applied by train_full / the attack
        # runner (subset = coreset, or coreset ∩ membership pool for RMIA).
        cfg.infomax_coreset_path = os.environ.get(
            "PRUNEPRIV_INFOMAX_CORESET",
            str(paths.results_dir("studies", "infomax", f"coreset_index_{spec.name}.npy")))

    # method-specific tweaks
    if method == "ib_noanneal":
        cfg.delta = 1.0
    elif method == "ib_earlystop_cooled":
        cfg.num_epochs = int(d["num_epochs"] * d["infobatch_delta"]) + 1  # e.g. 88 (epochs 0..87)
        cfg.delta = 1.0
    elif method in {"wlib_loss", "wlib_shuffle", "wlib_hardconst"}:
        cfg.prw_weight_mode = {
            "wlib_loss": "loss_final",
            "wlib_shuffle": "shuffle",
            "wlib_hardconst": "hard_tercile_const",
        }[method]

    # caller overrides (epochs, seeds, ...) applied last
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
