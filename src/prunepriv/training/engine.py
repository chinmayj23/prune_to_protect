import argparse
import os
import random
import time
from math import ceil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.amp import autocast
from torchmetrics import Accuracy

from prunepriv.configs.models.resnet20 import ResNet20
from prunepriv.utils.create_subsets import generate_partial_subsets, generate_subsets

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
except ImportError:
    Console = None
    Progress = None
    SpinnerColumn = None
    BarColumn = None
    TextColumn = None
    TimeRemainingColumn = None
    Table = None

try:
    import wandb
except ImportError:
    wandb = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SCRIPT_DIR = Path(__file__).resolve().parent

CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR10_STD = torch.tensor((0.2470, 0.2435, 0.2616))
CIFAR100_MEAN = torch.tensor((0.5071, 0.4867, 0.4408))
CIFAR100_STD = torch.tensor((0.2675, 0.2565, 0.2761))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def TriangularSchedule(total_epochs, optimizer, steps_per_epoch=1):
    total_train_steps = total_epochs * steps_per_epoch
    lr_schedule = np.interp(
        np.arange(1 + total_train_steps),
        [0, int(0.2 * total_train_steps), total_train_steps],
        [0.2, 1, 0],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    return scheduler


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r + sy:r + sy + crop_size, r + sx:r + sx + crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size + 2 * r), device=images.device, dtype=images.dtype)
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            images_tmp[mask] = images[mask, :, r + s:r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            images_out[mask] = images_tmp[mask, :, :, r + s:r + s + crop_size]
    return images_out


def make_random_square_masks(inputs, size):
    n, _, h, w = inputs.shape
    corner_y = torch.randint(0, h - size + 1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w - size + 1, size=(n,), device=inputs.device)
    corner_y_dists = torch.arange(h, device=inputs.device).view(1, 1, h, 1) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(1, 1, 1, w) - corner_x.view(-1, 1, 1, 1)
    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)
    return mask_y * mask_x


def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)


def make_metrics_table(title):
    if Table is None:
        return None
    table = Table(title=title)
    table.add_column("Epoch", justify="center")
    table.add_column("Train Loss", justify="center")
    table.add_column("Test Loss", justify="center")
    table.add_column("Train Accuracy", justify="center")
    table.add_column("Test Accuracy", justify="center")
    return table


def compute_effective_pruning_rate(total_samples: int, active_samples: int) -> float:
    return 1.0 - (active_samples / max(1, total_samples))


def resolve_training_method(cfg) -> str:
    method = str(getattr(cfg, "training_method", "auto")).lower()
    if method == "auto":
        return "ib" if getattr(cfg, "use_ib", False) else "standard"
    return method


def build_config():
    from cifar_configs import (
        Cifar10EHConfig,
        Cifar10IBFocalConfig,
        Cifar10IBFocalLateConfig,
        Cifar10IB3SegConfig,
        Cifar10IB3SegStaticConfig,
        Cifar10IBRisk3Config,
        Cifar10IBStdConfig,
        Cifar10StandardConfig,
        Cifar10IBConfig,
        Cifar10IBPRWConfig,
        Cifar10IB3SegPRWNormConfig,
        Cifar10StandardLOOConfig,
        Cifar10IBLOOConfig,
        Cifar10EHLOOConfig,
        Cifar10IBFocalLOOConfig,
        Cifar10IBFocalLateLOOConfig,
        Cifar10IB3SegLOOConfig,
        Cifar10IB3SegStaticLOOConfig,
        Cifar10IBRisk3LOOConfig,
        Cifar10IBStdLOOConfig,
        Cifar10IBPRWLOOConfig,
        Cifar10IB3SegPRWNormLOOConfig,
    )

    # expt_config = Cifar10EHConfig()
    # expt_config = Cifar10IBFocalConfig()
    # expt_config = Cifar10IBFocalLateConfig()
    # expt_config = Cifar10IB3SegConfig()
    # expt_config = Cifar10IB3SegStaticConfig()
    # expt_config = Cifar10IBRisk3Config()
    # expt_config = Cifar10IBStdConfig()
    # expt_config = Cifar10IBConfig()
    # expt_config = Cifar10StandardConfig()
    # expt_config = Cifar10StandardLOOConfig()
    expt_config = Cifar10IBLOOConfig()
    # expt_config = Cifar10EHLOOConfig()
    # expt_config = Cifar10IBFocalLOOConfig()
    # expt_config = Cifar10IBFocalLateLOOConfig()
    # expt_config = Cifar10IB3SegLOOConfig()
    # expt_config = Cifar10IB3SegPRWNormConfig()
    # expt_config = Cifar10IB3SegPRWNormLOOConfig()
    # expt_config = Cifar10IB3SegStaticLOOConfig()
    # expt_config = Cifar10IBRisk3LOOConfig()
    # expt_config = Cifar10IBFocalLateLOOConfig()
    # expt_config = Cifar10IBStdLOOConfig()
    # expt_config = Cifar10IBPRWConfig()
    # expt_config = Cifar10IBPRWLOOConfig()
    expt_config.dataset_name = str(getattr(expt_config, "dataset_name", "cifar10")).lower()
    expt_config.num_classes = 10 if expt_config.dataset_name == "cifar10" else 100

    if not hasattr(expt_config, "data_root_dir") or expt_config.data_root_dir is None:
        expt_config.data_root_dir = "data_cifar10" if expt_config.dataset_name == "cifar10" else "data_cifar100"

    if not hasattr(expt_config, "base_seed"):
        expt_config.base_seed = 42
    if not hasattr(expt_config, "seed_list"):
        expt_config.seed_list = [expt_config.base_seed]
    expt_config.seed_list = [int(seed) for seed in expt_config.seed_list]

    if not hasattr(expt_config, "train_aug"):
        expt_config.train_aug = {"flip": True, "translate": 2}
    if not hasattr(expt_config, "altflip"):
        expt_config.altflip = False
    if not hasattr(expt_config, "use_compile"):
        expt_config.use_compile = True
    if not hasattr(expt_config, "delta"):
        expt_config.delta = 0.875
    if not hasattr(expt_config, "grad_clip"):
        expt_config.grad_clip = False
    else:
        expt_config.grad_clip = bool(expt_config.grad_clip)
    if not hasattr(expt_config, "grad_clip_max_norm"):
        expt_config.grad_clip_max_norm = 1.0
    else:
        expt_config.grad_clip_max_norm = float(expt_config.grad_clip_max_norm)
    if expt_config.grad_clip_max_norm <= 0.0:
        raise ValueError("grad_clip_max_norm must be positive.")
    if not hasattr(expt_config, "ratio"):
        expt_config.ratio = float(getattr(expt_config, "ratio_easy", 0.5))
    else:
        expt_config.ratio = float(expt_config.ratio)
    if not hasattr(expt_config, "ratio_easy"):
        expt_config.ratio_easy = float(expt_config.ratio)
    else:
        expt_config.ratio_easy = float(expt_config.ratio_easy)
    if not hasattr(expt_config, "ratio_hard"):
        expt_config.ratio_hard = 1.0
    else:
        expt_config.ratio_hard = float(expt_config.ratio_hard)
    if not hasattr(expt_config, "ratio_std"):
        expt_config.ratio_std = float(getattr(expt_config, "ratio", 0.5))
    else:
        expt_config.ratio_std = float(expt_config.ratio_std)
    if not hasattr(expt_config, "top_keep_fraction"):
        expt_config.top_keep_fraction = float(getattr(expt_config, "std_factor", 0.55))
    else:
        expt_config.top_keep_fraction = float(expt_config.top_keep_fraction)
    if not hasattr(expt_config, "std_factor"):
        expt_config.std_factor = 1.0
    else:
        expt_config.std_factor = float(expt_config.std_factor)
    if not hasattr(expt_config, "segment_low_fraction"):
        expt_config.segment_low_fraction = 0.15
    else:
        expt_config.segment_low_fraction = float(expt_config.segment_low_fraction)
    if not hasattr(expt_config, "segment_high_fraction"):
        expt_config.segment_high_fraction = 0.15
    else:
        expt_config.segment_high_fraction = float(expt_config.segment_high_fraction)
    if not hasattr(expt_config, "segment_keep_low"):
        expt_config.segment_keep_low = 1.0
    else:
        expt_config.segment_keep_low = float(expt_config.segment_keep_low)
    if not hasattr(expt_config, "segment_keep_mid"):
        expt_config.segment_keep_mid = 0.5
    else:
        expt_config.segment_keep_mid = float(expt_config.segment_keep_mid)
    if not hasattr(expt_config, "segment_keep_high"):
        expt_config.segment_keep_high = 1.0
    else:
        expt_config.segment_keep_high = float(expt_config.segment_keep_high)
    if not hasattr(expt_config, "static_segment_epoch"):
        expt_config.static_segment_epoch = 10
    else:
        expt_config.static_segment_epoch = int(expt_config.static_segment_epoch)
    if not hasattr(expt_config, "risk_low_percentile"):
        expt_config.risk_low_percentile = 0.75
    else:
        expt_config.risk_low_percentile = float(expt_config.risk_low_percentile)
    if not hasattr(expt_config, "risk_high_percentile"):
        expt_config.risk_high_percentile = 0.95
    else:
        expt_config.risk_high_percentile = float(expt_config.risk_high_percentile)
    if not hasattr(expt_config, "risk_keep_low"):
        expt_config.risk_keep_low = float(getattr(expt_config, "ratio", 0.5))
    else:
        expt_config.risk_keep_low = float(expt_config.risk_keep_low)
    if not hasattr(expt_config, "risk_keep_mid"):
        expt_config.risk_keep_mid = 0.35
    else:
        expt_config.risk_keep_mid = float(expt_config.risk_keep_mid)
    if not hasattr(expt_config, "risk_keep_high"):
        expt_config.risk_keep_high = 0.85
    else:
        expt_config.risk_keep_high = float(expt_config.risk_keep_high)
    if not hasattr(expt_config, "risk_rescale_low"):
        expt_config.risk_rescale_low = True
    else:
        expt_config.risk_rescale_low = bool(expt_config.risk_rescale_low)
    if not hasattr(expt_config, "risk_rescale_mid"):
        expt_config.risk_rescale_mid = True
    else:
        expt_config.risk_rescale_mid = bool(expt_config.risk_rescale_mid)
    if not hasattr(expt_config, "risk_rescale_high"):
        expt_config.risk_rescale_high = True
    else:
        expt_config.risk_rescale_high = bool(expt_config.risk_rescale_high)
    if not hasattr(expt_config, "risk_warmup_epoch"):
        expt_config.risk_warmup_epoch = 10
    else:
        expt_config.risk_warmup_epoch = int(expt_config.risk_warmup_epoch)
    if not hasattr(expt_config, "risk_start_epoch"):
        expt_config.risk_start_epoch = 25
    else:
        expt_config.risk_start_epoch = int(expt_config.risk_start_epoch)
    if not (0.0 <= expt_config.risk_low_percentile <= expt_config.risk_high_percentile <= 1.0):
        raise ValueError("risk_low_percentile and risk_high_percentile must satisfy 0 <= low <= high <= 1.")
    if expt_config.risk_start_epoch < expt_config.risk_warmup_epoch:
        raise ValueError("risk_start_epoch must be greater than or equal to risk_warmup_epoch.")
    for keep_value, keep_name in (
        (expt_config.risk_keep_low, "risk_keep_low"),
        (expt_config.risk_keep_mid, "risk_keep_mid"),
        (expt_config.risk_keep_high, "risk_keep_high"),
    ):
        if keep_value < 0.0:
            raise ValueError(f"{keep_name} must be non-negative.")
    if not hasattr(expt_config, "use_ib"):
        expt_config.use_ib = False
    if not hasattr(expt_config, "noise_type"):
        expt_config.noise_type = "clean"
    if not hasattr(expt_config, "noise_file_path"):
        expt_config.noise_file_path = None
    if not hasattr(expt_config, "recompute_losses_before_ib"):
        expt_config.recompute_losses_before_ib = False
    if not hasattr(expt_config, "evolution_seed"):
        expt_config.evolution_seed = expt_config.base_seed
    if not hasattr(expt_config, "use_focal_loss"):
        expt_config.use_focal_loss = False
    if not hasattr(expt_config, "focal_gamma"):
        expt_config.focal_gamma = 0.0
    if not hasattr(expt_config, "focal_in_annealing_only"):
        expt_config.focal_in_annealing_only = False
    if not hasattr(expt_config, "train_loss"):
        expt_config.train_loss = "ce"
    if not hasattr(expt_config, "score_loss"):
        expt_config.score_loss = "ce"
    if not hasattr(expt_config, "is_loo"):
        expt_config.is_loo = False
    if not hasattr(expt_config, "is_loo_with_partial_subsets"):
        expt_config.is_loo_with_partial_subsets = False
    if expt_config.is_loo and expt_config.is_loo_with_partial_subsets:
        raise ValueError("Both is_loo and is_loo_with_partial_subsets cannot be True at the same time")
    if not hasattr(expt_config, "num_subsets"):
        expt_config.num_subsets = 200
    if not hasattr(expt_config, "sample_ratio_in_subsets"):
        expt_config.sample_ratio_in_subsets = 0.7
    if not hasattr(expt_config, "partial_subset_size"):
        expt_config.partial_subset_size = 5000
    if not hasattr(expt_config, "evaluate_every_epoch"):
        expt_config.evaluate_every_epoch = True
    if not hasattr(expt_config, "save_outputs"):
        expt_config.save_outputs = False
    if not hasattr(expt_config, "save_best_checkpoint"):
        expt_config.save_best_checkpoint = False
    if not hasattr(expt_config, "save_last_checkpoint"):
        expt_config.save_last_checkpoint = False
    if not hasattr(expt_config, "save_epoch_checkpoints"):
        expt_config.save_epoch_checkpoints = False
    if not hasattr(expt_config, "sample_at_epoch"):
        expt_config.sample_at_epoch = [10, 18, 25, 50, 57, 73, 75, 87, 88, 99]
    if not hasattr(expt_config, "log_sample_evolution"):
        expt_config.log_sample_evolution = False
    if not hasattr(expt_config, "log_evolution_activations"):
        expt_config.log_evolution_activations = True
    if not hasattr(expt_config, "log_evolution_grad_norms"):
        expt_config.log_evolution_grad_norms = True
    if not hasattr(expt_config, "use_fixed_indices_for_analysis"):
        expt_config.use_fixed_indices_for_analysis = False
    if not hasattr(expt_config, "log_noise_splits"):
        expt_config.log_noise_splits = True
    if not hasattr(expt_config, "log_wandb"):
        expt_config.log_wandb = False
    if not hasattr(expt_config, "wandb_project"):
        expt_config.wandb_project = f"{expt_config.dataset_name}_new"
    if not hasattr(expt_config, "wandb_entity"):
        expt_config.wandb_entity = None
    if not hasattr(expt_config, "wandb_group"):
        expt_config.wandb_group = None
    if not hasattr(expt_config, "wandb_tags"):
        expt_config.wandb_tags = None
    if not hasattr(expt_config, "loo_output_path"):
        expt_config.loo_output_path = None
    if not hasattr(expt_config, "experiment_name"):
        expt_config.experiment_name = None
    if not hasattr(expt_config, "non_loo_root"):
        expt_config.non_loo_root = None
    if not hasattr(expt_config, "loo_root"):
        expt_config.loo_root = None
    if not hasattr(expt_config, "device"):
        expt_config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(expt_config.device, str):
        expt_config.device = torch.device(expt_config.device)

    expt_config.training_method = resolve_training_method(expt_config)
    if expt_config.training_method == "standard":
        expt_config.use_ib = False
    elif expt_config.training_method == "ib":
        expt_config.use_ib = True
    elif expt_config.training_method == "eh":
        expt_config.use_ib = True
    elif expt_config.training_method == "ib_std":
        expt_config.use_ib = True
    elif expt_config.training_method == "ib_3seg":
        expt_config.use_ib = True
    elif expt_config.training_method == "ib_3seg_static":
        expt_config.use_ib = True
    elif expt_config.training_method == "ib_risk3":
        expt_config.use_ib = True
    elif expt_config.training_method == "ib_focal":
        expt_config.use_ib = True
        expt_config.train_loss = "focal"
        expt_config.score_loss = "focal"
        expt_config.use_focal_loss = True
        expt_config.focal_in_annealing_only = False
    elif expt_config.training_method == "ib_focal_late":
        expt_config.use_ib = True
        expt_config.train_loss = "focal"
        expt_config.score_loss = "focal"
        expt_config.use_focal_loss = True
        expt_config.focal_in_annealing_only = True
    elif expt_config.training_method == "ib_u_batch":
        expt_config.use_ib = True
    elif expt_config.training_method == "ib_prw":
        expt_config.use_ib = True
    elif expt_config.training_method == "ib_3seg_prw_norm":
        expt_config.use_ib = True
    else:
        raise ValueError("Supported training_method values are 'standard', 'ib', 'ib_prw', 'ib_3seg_prw_norm', 'ib_u_batch', 'eh', 'ib_std', 'ib_3seg', 'ib_3seg_static', 'ib_risk3', 'ib_focal', or 'ib_focal_late'.")

    if bool(getattr(expt_config, "use_ib_sampler", False)):
        raise ValueError("use_ib_sampler=True is not supported.")
    expt_config.train_loss = str(getattr(expt_config, "train_loss", "ce")).lower()
    expt_config.score_loss = str(getattr(expt_config, "score_loss", "ce")).lower()
    if expt_config.train_loss not in {"ce", "focal"}:
        raise ValueError("train_loss must be 'ce' or 'focal'.")
    if expt_config.score_loss not in {"ce", "focal"}:
        raise ValueError("score_loss must be 'ce' or 'focal'.")
    expt_config.use_focal_loss = bool(getattr(expt_config, "use_focal_loss", False) or expt_config.train_loss == "focal" or expt_config.score_loss == "focal")

    if torch.cuda.is_available() and str(getattr(expt_config, "training_precision", "auto")).lower() == "auto":
        device_name = torch.cuda.get_device_name(0)
        expt_config.training_precision = "float32" if "T4" in device_name else "bfloat16"

    return expt_config


class CifarLoader:
    def __init__(self, path, dataset_name, train=True, batch_size=500, aug=None, drop_last=None, shuffle=None, altflip=False, noise_type="clean", noise_file_path=None, device="cuda"):
        self.device = torch.device(device)
        data_path = os.path.join(path, f"{dataset_name}_train.pt" if train else f"{dataset_name}_test.pt")
        if not os.path.exists(data_path):
            if dataset_name == "cifar10":
                dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            else:
                dset = torchvision.datasets.CIFAR100(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)
        data = torch.load(data_path, map_location=self.device)
        self.train = train

        self.epoch = 0
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]
        self.sample_ids = torch.arange(len(self.images), device=self.device)

        if self.train:
            self.clean_labels = self.labels.clone()
            self.noisy_labels = None
            self.noise_mask = None

        if self.train and noise_type != "clean":
            noise_path = noise_file_path or os.path.join(
                os.environ.get("PRUNEPRIV_NOISE_DIR", os.path.join(os.environ.get("PRUNEPRIV_DATA", "data"), "noisy")),
                "CIFAR-10_human.pt",
            )
            if os.path.exists(noise_path):
                print(f"Loading noisy labels ({noise_type}) from {noise_path}")
                noise_data = torch.load(noise_path, map_location="cpu", weights_only=False)
                if isinstance(noise_data, dict) and noise_type in noise_data:
                    if "clean_label" in noise_data:
                        clean_label = torch.as_tensor(noise_data["clean_label"]).reshape(-1)
                        if not torch.equal(self.labels.cpu(), clean_label):
                            print("Warning: Clean labels in noisy file do not match dataset labels.")
                    self.noisy_labels = torch.as_tensor(noise_data[noise_type]).reshape(-1).to(self.device)
                    self.labels = self.noisy_labels
                    self.noise_mask = self.noisy_labels != self.clean_labels
                else:
                    raise ValueError(f"Noise type '{noise_type}' not found in {noise_path}")
            else:
                raise FileNotFoundError(f"Noisy labels file not found at {noise_path}")

        self.images = (self.images / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
        self.normalize = T.Normalize(*((CIFAR10_MEAN, CIFAR10_STD) if dataset_name == "cifar10" else (CIFAR100_MEAN, CIFAR100_STD)))
        self.proc_images = {}

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate", "cutout"], f"Unrecognized key: {k}"

        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle
        self.altflip = altflip

        self.scores = torch.ones(len(self.images), device=self.device) * 3
        self.weights = torch.ones(len(self.images), device=self.device)
        self.static_three_segment_bins = None

    def perform_infobatch(self, use_infobatch: bool = True, keep_ratio: float = 0.5):
        status_log = torch.zeros(len(self.images), dtype=torch.int8, device=self.device)
        if not use_infobatch:
            self.weights[:] = 1.0
            status_log[:] = 3
            return torch.randperm(len(self.images), device=self.device), status_log
        assert self.train, "InfoBatch can only be used during training."
        well_learned_mask = self.scores < self.scores.mean()
        well_learned_indices = torch.where(well_learned_mask)[0]
        remained_indices = torch.where(~well_learned_mask)[0]
        status_log[remained_indices] = 2
        self.weights[:] = 1.0
        num_to_select = int(keep_ratio * len(well_learned_indices))
        if num_to_select > 0:
            selected_indices = well_learned_indices[torch.randperm(len(well_learned_indices), device=self.device)[:num_to_select]]
            self.weights[selected_indices] = 1 / keep_ratio
            indices = torch.cat([remained_indices, selected_indices])
            indices = indices[torch.randperm(len(indices), device=self.device)]
            status_log[selected_indices] = 1
        else:
            indices = remained_indices[torch.randperm(len(remained_indices), device=self.device)]
        return indices, status_log

    def perform_random_prune(self, use_infobatch: bool = True, keep_ratio: float = 0.5):
        """RANDOM pruning at InfoBatch's per-epoch rate (ablation for the ib_random method).

        InfoBatch drops (1-keep_ratio) of the EASY samples each epoch; this drops the SAME COUNT but
        chosen uniformly at random from ALL samples, with an unbiased upweight on the kept set (total
        weighted scale = N, matching InfoBatch). Isolates the effect of *which* samples are pruned
        (selective-easy vs random) at identical data volume.
        """
        status_log = torch.zeros(len(self.images), dtype=torch.int8, device=self.device)
        if not use_infobatch:
            self.weights[:] = 1.0
            status_log[:] = 3
            return torch.randperm(len(self.images), device=self.device), status_log
        assert self.train, "ib_random can only be used during training."
        n = len(self.images)
        num_easy = int((self.scores < self.scores.mean()).sum())
        num_pruned = num_easy - int(keep_ratio * num_easy)   # == InfoBatch's pruned count this epoch
        self.weights[:] = 1.0
        perm = torch.randperm(n, device=self.device)
        pruned_idx, kept_idx = perm[:num_pruned], perm[num_pruned:]
        status_log[pruned_idx] = 0                            # pruned (counted by prune_counts)
        status_log[kept_idx] = 1                              # kept (randomly)
        if num_pruned > 0 and len(kept_idx) > 0:
            self.weights[kept_idx] = float(n) / float(len(kept_idx))   # unbiased rescale
        return kept_idx[torch.randperm(len(kept_idx), device=self.device)], status_log

    def perform_easy_hard(self, use_infobatch: bool = True, keep_ratio_easy: float = 0.5, keep_ratio_hard: float = 1.0):
        if keep_ratio_hard >= 1.0:
            return self.perform_infobatch(use_infobatch=use_infobatch, keep_ratio=keep_ratio_easy)
        status_log = torch.zeros(len(self.images), dtype=torch.int8, device=self.device)
        if not use_infobatch:
            self.weights[:] = 1.0
            status_log[:] = 3
            return torch.randperm(len(self.images), device=self.device), status_log
        assert self.train, "EH can only be used during training."

        easy_mask = self.scores < self.scores.mean()
        easy_indices = torch.where(easy_mask)[0]
        hard_indices = torch.where(~easy_mask)[0]
        self.weights[:] = 1.0

        num_easy = int(keep_ratio_easy * len(easy_indices))
        num_hard = int(keep_ratio_hard * len(hard_indices))

        if num_easy > 0:
            selected_easy = easy_indices[torch.randperm(len(easy_indices), device=self.device)[:num_easy]]
            self.weights[selected_easy] = 1 / keep_ratio_easy
            status_log[selected_easy] = 1
        else:
            selected_easy = easy_indices.new_empty((0,), dtype=torch.long)

        if num_hard > 0:
            selected_hard = hard_indices[torch.randperm(len(hard_indices), device=self.device)[:num_hard]]
            if keep_ratio_hard < 1.0:
                self.weights[selected_hard] = 1 / keep_ratio_hard
            status_log[selected_hard] = 2
        else:
            selected_hard = hard_indices.new_empty((0,), dtype=torch.long)

        indices = torch.cat([selected_hard, selected_easy])
        if len(indices) > 0:
            indices = indices[torch.randperm(len(indices), device=self.device)]
        return indices, status_log

    def perform_infobatch_std(self, use_infobatch: bool = True, keep_ratio: float = 0.35, top_keep_fraction: float = 0.55):
        status_log = torch.zeros(len(self.images), dtype=torch.int8, device=self.device)
        if not use_infobatch:
            self.weights[:] = 1.0
            status_log[:] = 3
            return torch.randperm(len(self.images), device=self.device), status_log
        assert self.train, "InfoBatch-std can only be used during training."

        shuffled_indices = torch.randperm(len(self.images), device=self.device)
        shuffled_scores = self.scores[shuffled_indices]
        sorted_order = torch.argsort(shuffled_scores, descending=True)
        ranked_indices = shuffled_indices[sorted_order]

        num_top = int(top_keep_fraction * len(ranked_indices))
        remained_indices = ranked_indices[:num_top]
        prunable_indices = ranked_indices[num_top:]
        status_log[remained_indices] = 2
        self.weights[:] = 1.0
        num_to_select = int(keep_ratio * len(prunable_indices))
        if num_to_select > 0:
            selected_indices = prunable_indices[torch.randperm(len(prunable_indices), device=self.device)[:num_to_select]]
            self.weights[selected_indices] = 1 / keep_ratio
            indices = torch.cat([remained_indices, selected_indices])
            indices = indices[torch.randperm(len(indices), device=self.device)]
            status_log[selected_indices] = 1
        else:
            indices = remained_indices[torch.randperm(len(remained_indices), device=self.device)]
        return indices, status_log

    def _ranked_three_segment_bins(self, low_fraction: float, high_fraction: float):
        shuffled_indices = torch.randperm(len(self.images), device=self.device)
        shuffled_scores = self.scores[shuffled_indices]
        sorted_order = torch.argsort(shuffled_scores, descending=True)
        ranked_indices = shuffled_indices[sorted_order]

        total = len(ranked_indices)
        num_high = min(max(int(high_fraction * total), 0), total)
        num_low = min(max(int(low_fraction * total), 0), max(0, total - num_high))

        high_indices = ranked_indices[:num_high]
        low_indices = ranked_indices[total - num_low:] if num_low > 0 else ranked_indices.new_empty((0,), dtype=torch.long)
        mid_end = total - num_low if num_low > 0 else total
        mid_indices = ranked_indices[num_high:mid_end]
        return low_indices, mid_indices, high_indices

    def _sample_three_segment_bins(
        self,
        low_indices: torch.Tensor,
        mid_indices: torch.Tensor,
        high_indices: torch.Tensor,
        keep_low: float,
        keep_mid: float,
        keep_high: float,
        rescale_low: bool = True,
        rescale_mid: bool = True,
        rescale_high: bool = True,
    ):
        status_log = torch.zeros(len(self.images), dtype=torch.int8, device=self.device)
        self.weights[:] = 1.0

        def select(indices: torch.Tensor, keep_ratio: float, status_value: int, use_rescaling: bool):
            if len(indices) == 0 or keep_ratio <= 0.0:
                return indices.new_empty((0,), dtype=torch.long)
            if keep_ratio >= 1.0:
                selected = indices
            else:
                num_to_select = int(keep_ratio * len(indices))
                if num_to_select <= 0:
                    return indices.new_empty((0,), dtype=torch.long)
                selected = indices[torch.randperm(len(indices), device=self.device)[:num_to_select]]
                if use_rescaling:
                    self.weights[selected] = 1.0 / keep_ratio
            status_log[selected] = status_value
            return selected

        selected_low = select(low_indices, keep_low, 1, rescale_low)
        selected_mid = select(mid_indices, keep_mid, 2, rescale_mid)
        selected_high = select(high_indices, keep_high, 2, rescale_high)

        indices = torch.cat([selected_high, selected_mid, selected_low])
        if len(indices) > 0:
            indices = indices[torch.randperm(len(indices), device=self.device)]
        return indices, status_log

    def perform_three_segment(
        self,
        use_infobatch: bool = True,
        low_fraction: float = 0.15,
        high_fraction: float = 0.15,
        keep_low: float = 1.0,
        keep_mid: float = 0.5,
        keep_high: float = 1.0,
    ):
        status_log = torch.zeros(len(self.images), dtype=torch.int8, device=self.device)
        if not use_infobatch:
            self.weights[:] = 1.0
            status_log[:] = 3
            return torch.randperm(len(self.images), device=self.device), status_log
        assert self.train, "3-segment selection can only be used during training."

        low_indices, mid_indices, high_indices = self._ranked_three_segment_bins(
            low_fraction=low_fraction,
            high_fraction=high_fraction,
        )
        return self._sample_three_segment_bins(
            low_indices=low_indices,
            mid_indices=mid_indices,
            high_indices=high_indices,
            keep_low=keep_low,
            keep_mid=keep_mid,
            keep_high=keep_high,
        )

    def perform_static_three_segment(
        self,
        epoch: int,
        use_infobatch: bool = True,
        freeze_epoch: int = 10,
        low_fraction: float = 0.20,
        high_fraction: float = 0.20,
        keep_low: float = 0.85,
        keep_mid: float = 0.5,
        keep_high: float = 0.85,
    ):
        status_log = torch.zeros(len(self.images), dtype=torch.int8, device=self.device)
        if not use_infobatch or epoch < freeze_epoch:
            self.weights[:] = 1.0
            status_log[:] = 3
            return torch.randperm(len(self.images), device=self.device), status_log
        assert self.train, "Static 3-segment selection can only be used during training."

        if self.static_three_segment_bins is None:
            self.static_three_segment_bins = self._ranked_three_segment_bins(
                low_fraction=low_fraction,
                high_fraction=high_fraction,
            )

        low_indices, mid_indices, high_indices = self.static_three_segment_bins
        return self._sample_three_segment_bins(
            low_indices=low_indices,
            mid_indices=mid_indices,
            high_indices=high_indices,
            keep_low=keep_low,
            keep_mid=keep_mid,
            keep_high=keep_high,
        )

    def perform_risk_three_band(
        self,
        use_infobatch: bool = True,
        low_percentile: float = 0.75,
        high_percentile: float = 0.95,
        keep_low: float = 0.5,
        keep_mid: float = 0.35,
        keep_high: float = 0.85,
        rescale_low: bool = True,
        rescale_mid: bool = True,
        rescale_high: bool = True,
    ):
        status_log = torch.zeros(len(self.images), dtype=torch.int8, device=self.device)
        if not use_infobatch:
            self.weights[:] = 1.0
            status_log[:] = 3
            return torch.randperm(len(self.images), device=self.device), status_log
        assert self.train, "IB-risk3 selection can only be used during training."

        shuffled_indices = torch.randperm(len(self.images), device=self.device)
        shuffled_scores = self.scores[shuffled_indices]
        sorted_order = torch.argsort(shuffled_scores, descending=False)
        ranked_indices = shuffled_indices[sorted_order]

        total = len(ranked_indices)
        low_cut = min(max(int(low_percentile * total), 0), total)
        high_cut = min(max(int(high_percentile * total), low_cut), total)

        low_indices = ranked_indices[:low_cut]
        mid_indices = ranked_indices[low_cut:high_cut]
        high_indices = ranked_indices[high_cut:]

        return self._sample_three_segment_bins(
            low_indices=low_indices,
            mid_indices=mid_indices,
            high_indices=high_indices,
            keep_low=keep_low,
            keep_mid=keep_mid,
            keep_high=keep_high,
            rescale_low=rescale_low,
            rescale_mid=rescale_mid,
            rescale_high=rescale_high,
        )


    def perform_function_pruning(
        self,
        use_infobatch: bool,
        prob_fn,
        weight_fn,
        beta: float = 1.0
    ):
        status_log = torch.zeros(len(self.images), dtype=torch.int8, device=self.device)
        if not use_infobatch:
            self.weights[:] = 1.0
            status_log[:] = 3
            return torch.randperm(len(self.images), device=self.device), status_log
            
        assert self.train, "Function pruning can only be used during training."

        # 1. Rank the cached losses (0.0 is easiest, 1.0 is hardest)
        shuffled_indices = torch.randperm(len(self.images), device=self.device)
        shuffled_scores = self.scores[shuffled_indices]
        sorted_order = torch.argsort(shuffled_scores, descending=False) # Ascending: easy to hard
        ranked_indices = shuffled_indices[sorted_order]
        
        total = len(ranked_indices)
        ranks = torch.arange(total, device=self.device, dtype=torch.float32) / max(1, total - 1)
        
        # 2. Map ranks to probabilities using your custom function
        probs = prob_fn(ranks)
        
        # 3. Stochastic Drop
        random_rolls = torch.rand(total, device=self.device)
        keep_mask = random_rolls < probs
        
        survivor_indices = ranked_indices[keep_mask]
        survivor_probs = probs[keep_mask]
        
        # 4. Apply annealed weights
        self.weights[:] = 1.0
        if len(survivor_indices) > 0:
            self.weights[survivor_indices] = weight_fn(survivor_probs, beta)
            
        # 5. Populate status log for WandB (0: pruned, 1: easy kept, 2: hard kept)
        pruned_indices = ranked_indices[~keep_mask]
        status_log[pruned_indices] = 0
        
        easy_survivors = ranked_indices[keep_mask & (ranks < 0.5)]
        hard_survivors = ranked_indices[keep_mask & (ranks >= 0.5)]
        status_log[easy_survivors] = 1
        status_log[hard_survivors] = 2

        # 6. Return randomized survivors
        if len(survivor_indices) > 0:
            survivor_indices = survivor_indices[torch.randperm(len(survivor_indices), device=self.device)]
            
        return survivor_indices, status_log

    def update_losses(self, values: torch.Tensor, indices: torch.Tensor, scores: Optional[torch.Tensor] = None):
        if scores is not None:
            loss_val = scores.detach()
        else:
            loss_val = values.detach().clone()
        self.scores[indices] = loss_val
        values = values * self.weights[indices]
        return values.mean()

    def batches_for_indices(self, indices: Optional[torch.Tensor] = None) -> int:
        if indices is None:
            total_samples = len(self.images)
        else:
            total_samples = len(indices)
        return (total_samples + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return len(self.images) // self.batch_size if self.drop_last else ceil(len(self.images) / self.batch_size)

    def __setattr__(self, k, v):
        if k in ("images", "labels"):
            assert self.epoch == 0, "Changing images or labels is only unsupported before iteration."
        super().__setattr__(k, v)

    def iterate(self, indices: Optional[torch.Tensor]):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(input=images, pad=(pad, pad, pad, pad), mode="reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]

        if self.aug.get("flip", False):
            if self.altflip:
                if self.epoch % 2 == 1:
                    images = images.flip(-1)
            else:
                images = batch_flip_lr(images)
        if self.aug.get("cutout", 0) > 0:
            images = batch_cutout(images, self.aug["cutout"])

        self.epoch += 1

        if self.train:
            if indices is None:
                indices = torch.randperm(len(self.images), device=self.device) if self.shuffle else torch.arange(len(self.images), device=self.device)
            n_batches = self.batches_for_indices(indices)
            for i in range(n_batches):
                idxs = indices[i * self.batch_size:(i + 1) * self.batch_size]
                yield images[idxs], self.labels[idxs], idxs
        else:
            indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
            for i in range(len(self)):
                idxs = indices[i * self.batch_size:(i + 1) * self.batch_size]
                yield images[idxs], self.labels[idxs]

    def iterate_wo_aug(self, indices: Optional[torch.Tensor]):
        images = self.proc_images["norm"]
        if self.train:
            if indices is None:
                indices = torch.randperm(len(self.images), device=self.device) if self.shuffle else torch.arange(len(self.images), device=self.device)
            n_batches = self.batches_for_indices(indices)
            for i in range(n_batches):
                idxs = indices[i * self.batch_size:(i + 1) * self.batch_size]
                yield images[idxs], self.labels[idxs], idxs
        else:
            indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
            for i in range(len(self)):
                idxs = indices[i * self.batch_size:(i + 1) * self.batch_size]
                yield images[idxs], self.labels[idxs]

    def __iter__(self):
        return self.iterate(None)

    def apply_subset(self, subset_indices: torch.Tensor):
        self.images = self.images[subset_indices]
        self.labels = self.labels[subset_indices]
        self.sample_ids = self.sample_ids[subset_indices]
        self.scores = self.scores[subset_indices]
        self.weights = self.weights[subset_indices]
        if self.clean_labels is not None:
            self.clean_labels = self.clean_labels[subset_indices]
        if self.noisy_labels is not None:
            self.noisy_labels = self.noisy_labels[subset_indices]
        if self.noise_mask is not None:
            self.noise_mask = self.noise_mask[subset_indices]
        self.epoch = 0
        self.proc_images = {}
        self.static_three_segment_bins = None


class TrainingHarness:
    def __init__(self, expt_config, subset_indices: Optional[torch.Tensor] = None):
        self.expt_config = expt_config
        self.criterion = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=float(getattr(expt_config, "label_smoothing", 0.0)))
        self.device = self.expt_config.device

        self.setup_model()
        if self.expt_config.use_compile:
            self.model = torch.compile(self.model)

        if getattr(self.expt_config, "log_sample_evolution", False) and getattr(self.expt_config, "log_evolution_activations", True):
            self.setup_hooks()
        self.setup_loaders(subset_indices)
        self.setup_optimizer()
        self.precision, self.use_amp = self.get_dtype_amp()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.expt_config.num_classes).to(self.device)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.expt_config.num_classes).to(self.device)
        self.switch_epoch = int(self.expt_config.num_epochs * self.expt_config.delta)
        if getattr(self.expt_config, "log_sample_evolution", False):
            if subset_indices is not None:
                self.held_out_indices_to_log = torch.arange(len(self.train_loader.images), device=self.device)
            elif getattr(self.expt_config, "use_fixed_indices_for_analysis", False):
                try:
                    fixed_indices_file = f"fixed_indices_cifar_{self.expt_config.num_subsets}_subsets.pt"
                    fixed_indices = torch.load(fixed_indices_file)
                    all_indices_mask = torch.ones(50000, dtype=torch.bool)
                    all_indices_mask[fixed_indices] = False
                    self.held_out_indices_to_log = torch.where(all_indices_mask)[0].to(self.device)
                    print(f"Loaded {len(self.held_out_indices_to_log)} held-out indices for evolution tracking.")
                except FileNotFoundError:
                    print(f"Warning: Could not find {fixed_indices_file}.")
                    print("Logging evolution for ALL 50,000 samples.")
                    self.held_out_indices_to_log = torch.arange(50000, device=self.device)
            else:
                self.held_out_indices_to_log = torch.arange(50000, device=self.device)
        else:
            print("No sample evolution data is recorded.")
            self.held_out_indices_to_log = None

    def get_dtype_amp(self):
        dtype_map = {
            "bfloat16": (torch.bfloat16, True),
            "float16": (torch.float16, True),
            "float32": (torch.float32, False),
        }
        return dtype_map.get(str(self.expt_config.training_precision).lower(), (torch.float32, False))

    def setup_hooks(self):
        self.captured_activations = None
        self.penultimate_features = None

        def hook_fn(module, input, output):
            self.captured_activations = input[0].detach()

        model_name = self.expt_config.model_name
        if model_name in ["r18", "r20", "r50"]:
            layer_to_hook = self.model.fc
            self.penultimate_features = layer_to_hook.in_features
            layer_to_hook.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Model name {model_name} not recognized for hooks.")

    def get_evolution_data(self, epoch):
        from torch._functorch import config

        need_activations = bool(getattr(self.expt_config, "log_evolution_activations", True))
        need_grads = bool(getattr(self.expt_config, "log_evolution_grad_norms", True) and epoch in self.expt_config.sample_at_epoch)
        # Per-sample gradient-DIRECTION sketch (for per-group gradient alignment/cosine): a fixed
        # Johnson-Lindenstrauss random projection of each sample's full-parameter gradient to
        # grad_sketch_dim dims (cosines are preserved), unit-normalised. Piggybacks on the per-sample
        # autograd already done for the norm, so it adds negligible compute. Default off.
        need_sketch = bool(getattr(self.expt_config, "log_evolution_grad_sketch", False) and need_grads)
        sketch_dim = int(getattr(self.expt_config, "grad_sketch_dim", 512))

        self.model.eval()

        all_losses = torch.zeros(len(self.train_loader.images), device=self.device)
        all_entropies = torch.zeros(len(self.train_loader.images), device=self.device)
        all_grad_norms = torch.full((len(self.train_loader.images),), float("nan"), device=self.device) if need_grads else None
        all_grad_sketches = torch.full((len(self.train_loader.images), sketch_dim), float("nan"), device=self.device) if need_sketch else None
        all_activations = None
        if need_activations:
            all_activations = torch.zeros(
                len(self.train_loader.images),
                self.penultimate_features,
                device=self.device,
                dtype=self.precision,
            )

        for inputs, targets, indices in self.train_loader.iterate_wo_aug(None):
            self.model.zero_grad(set_to_none=True)
            ctx = torch.enable_grad() if need_grads else torch.no_grad()
            with ctx, autocast("cuda", dtype=self.precision, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            if need_grads:
                batch_grad_norms = []
                batch_sketches = [] if need_sketch else None
                for j, sample_loss in enumerate(loss):
                    retain = j < len(loss) - 1
                    with config.patch(donated_buffer=False):
                        grads = torch.autograd.grad(
                            sample_loss,
                            self.model.parameters(),
                            retain_graph=retain,
                            allow_unused=True,
                        )
                    flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
                    batch_grad_norms.append(flat.pow(2).sum())
                    if need_sketch:
                        proj = self._get_grad_proj(flat.numel(), sketch_dim, flat.dtype)
                        s = flat.float() @ proj                       # [k] projected gradient direction
                        n = s.norm()
                        batch_sketches.append(s / n if n > 0 else torch.zeros_like(s))
                all_grad_norms[indices] = torch.stack(batch_grad_norms)
                if need_sketch:
                    all_grad_sketches[indices] = torch.stack(batch_sketches)

            all_losses[indices] = loss
            if all_activations is not None and self.captured_activations is not None:
                all_activations[indices] = self.captured_activations
            all_entropies[indices] = entropies

        return all_losses, all_activations, all_grad_norms, all_entropies, all_grad_sketches

    def _get_grad_proj(self, dim: int, sketch_dim: int, dtype):
        """Fixed (seeded) Gaussian random projection matrix [dim, sketch_dim], built once and reused
        across all samples/epochs so gradient directions are comparable. Scaled by 1/sqrt(sketch_dim)
        so the projection approximately preserves inner products / cosines (JL)."""
        proj = getattr(self, "_grad_proj", None)
        if proj is None or proj.shape[0] != dim or proj.shape[1] != sketch_dim:
            gen = torch.Generator(device=self.device).manual_seed(int(getattr(self.expt_config, "grad_sketch_seed", 20240726)))
            proj = torch.randn(dim, sketch_dim, generator=gen, device=self.device, dtype=torch.float32) / (sketch_dim ** 0.5)
            self._grad_proj = proj
        return proj

    def setup_loaders(self, subset_indices: Optional[torch.Tensor]):
        backend = getattr(self.expt_config, "loader_backend", "fast_pt")
        if backend == "image_pt":
            from prunepriv.data.image_loader import ImageLoader  # lazy (avoids circular import)
            common = dict(path=self.expt_config.data_root_dir, dataset_name=self.expt_config.dataset_name,
                          batch_size=self.expt_config.batch_size, device=self.device,
                          blob_device=getattr(self.expt_config, "loader_blob_device", None),
                          mean=getattr(self.expt_config, "norm_mean", None),
                          std=getattr(self.expt_config, "norm_std", None),
                          crop_size=getattr(self.expt_config, "input_size", 224))
            self.train_loader = ImageLoader(train=True, aug=self.expt_config.train_aug,
                                            altflip=self.expt_config.altflip, **common)
            self.test_loader = ImageLoader(train=False, **common)
        else:
            self.train_loader = CifarLoader(
                path=self.expt_config.data_root_dir, dataset_name=self.expt_config.dataset_name,
                train=True, batch_size=self.expt_config.batch_size, aug=self.expt_config.train_aug,
                altflip=self.expt_config.altflip,
                noise_type=getattr(self.expt_config, "noise_type", "clean"),
                noise_file_path=getattr(self.expt_config, "noise_file_path", None), device=self.device)
            self.test_loader = CifarLoader(
                path=self.expt_config.data_root_dir, dataset_name=self.expt_config.dataset_name,
                train=False, batch_size=self.expt_config.batch_size, device=self.device)
        if subset_indices is not None:
            self.train_loader.apply_subset(subset_indices.to(self.train_loader.device))

    def setup_model(self):
        if self.expt_config.model_name == "r18":
            print("Using ResNet18 model.")
            self.model = torchvision.models.resnet18(num_classes=self.expt_config.num_classes, weights=None)
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.maxpool = nn.Identity()
        elif self.expt_config.model_name == "r50":
            print("Using ResNet50 model.")
            self.model = torchvision.models.resnet50(num_classes=self.expt_config.num_classes, weights=None)
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.maxpool = nn.Identity()
        elif self.expt_config.model_name == "r20":
            print("Using ResNet20 model.")
            self.model = ResNet20(num_classes=self.expt_config.num_classes)
        elif self.expt_config.model_name == "wrn28_2":
            print("Using WideResNet-28-2 model.")
            from prunepriv.configs.models.wideresnet import WideResNet
            self.model = WideResNet(depth=28, num_classes=self.expt_config.num_classes, widen_factor=2)
        else:
            # Any other registered model via the factory. Honors `pretrained`.
            from prunepriv.configs.models import build_model
            print(f"Using {self.expt_config.model_name} via build_model "
                  f"(pretrained={getattr(self.expt_config, 'pretrained', False)}).")
            self.model = build_model(self.expt_config.model_name,
                                     num_classes=self.expt_config.num_classes,
                                     pretrained=getattr(self.expt_config, "pretrained", False))
        self.model.to(self.expt_config.device)

    def setup_optimizer(self):
        opt_name = getattr(self.expt_config, "optimizer_name", "sgd").lower()
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.expt_config.learning_rate,
                weight_decay=self.expt_config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.expt_config.learning_rate,
                momentum=self.expt_config.momentum,
                weight_decay=self.expt_config.weight_decay,
                nesterov=True,
            )
        self.scheduler = TriangularSchedule(self.expt_config.num_epochs, self.optimizer, steps_per_epoch=1)

    def use_focal_this_epoch(self, epoch: int) -> bool:
        if not self.expt_config.use_focal_loss:
            return False
        if getattr(self.expt_config, "focal_in_annealing_only", False):
            return epoch > self.switch_epoch
        return True

    def compute_train_and_score_losses(self, outputs: torch.Tensor, targets: torch.Tensor, epoch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ce_loss = self.criterion(outputs, targets)
        if not self.use_focal_this_epoch(epoch):
            return ce_loss, ce_loss

        gamma = float(getattr(self.expt_config, "focal_gamma", 0.0))
        probs = F.softmax(outputs, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal_loss = ((1 - p_t) ** gamma) * ce_loss

        train_values = focal_loss if self.expt_config.train_loss == "focal" else ce_loss
        score_values = focal_loss if self.expt_config.score_loss == "focal" else ce_loss
        return train_values, score_values

    def train_one_epoch(self, epoch: int, active_indices: Optional[torch.Tensor]) -> Tuple[float, float]:
        self.model.train()
        train_loss = 0.0
        self.train_accuracy.reset()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeRemainingColumn(elapsed_when_finished=True),
        )

        with progress:
            task = progress.add_task(f"[cyan]Epoch {epoch}", total=len(self.train_loader))
            for inputs, targets, batch_indices in self.train_loader.iterate(indices=active_indices):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", dtype=self.precision, enabled=self.use_amp):
                    outputs = self.model(inputs)
                    if self.expt_config.use_focal_loss:
                        train_values, score_values = self.compute_train_and_score_losses(outputs, targets, epoch)
                        loss = self.train_loader.update_losses(values=train_values, indices=batch_indices, scores=score_values)
                    else:
                        loss = self.criterion(outputs, targets)
                        loss = self.train_loader.update_losses(values=loss, indices=batch_indices)
                loss.backward()
                if getattr(self.expt_config, "grad_clip", False):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.expt_config.grad_clip_max_norm)
                self.optimizer.step()

                self._global_step = getattr(self, "_global_step", 0) + 1
                _sset = getattr(self, "_save_at_steps", None)
                if _sset and self._global_step in _sset and getattr(self, "_step_ckpt_dir", None) is not None:
                    torch.save(self.model.state_dict(), self._step_ckpt_dir / f"step_{self._global_step}.pt")

                train_loss += loss.item()
                self.train_accuracy.update(outputs, targets)
                progress.update(task, advance=1)

            train_loss /= len(self.train_loader)

        self.scheduler.step()
        accuracy = self.train_accuracy.compute().item() * 100
        return train_loss, accuracy

    def test(self) -> Tuple[float, float]:
        self.model.eval()
        test_loss = 0.0
        self.test_accuracy.reset()

        with torch.no_grad(), autocast("cuda", dtype=self.precision, enabled=self.use_amp):
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = torch.mean(loss)
                test_loss += loss.item()
                self.test_accuracy.update(outputs, targets)

        test_loss /= len(self.test_loader)
        accuracy = self.test_accuracy.compute().item() * 100
        return test_loss, accuracy

    @torch.no_grad()
    def recompute_all_losses(self, run_checks_and_print=False):
        self.model.eval()
        if run_checks_and_print:
            assert not self.model.training, "Model is still in train() mode during recomputation!"
        if "norm" not in self.train_loader.proc_images:
            next(self.train_loader.iterate(indices=torch.arange(1, device=self.device)))
            self.train_loader.epoch = 0

        total_samples = len(self.train_loader.images)
        if run_checks_and_print:
            old_scores = self.train_loader.scores.clone()
            touched_mask = torch.zeros(total_samples, dtype=torch.bool, device=self.device)
            print(f"\n[OHEM] Recomputing exact losses for all {total_samples} samples...")

        for inputs, targets, indices in self.train_loader.iterate_wo_aug(None):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with autocast("cuda", dtype=self.precision, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.train_loader.scores[indices] = loss.detach()
                if run_checks_and_print:
                    touched_mask[indices] = True

        if run_checks_and_print:
            assert touched_mask.all(), f"Coverage failure! Only {touched_mask.sum().item()} / {total_samples} samples updated."
            assert not torch.isnan(self.train_loader.scores).any(), "NaN values detected in recomputed losses!"
            assert not torch.isinf(self.train_loader.scores).any(), "Infinity values detected in recomputed losses!"
            if len(torch.unique(old_scores)) > 1:
                differences = torch.sum(self.train_loader.scores != old_scores).item()
                assert differences > 0, "Losses did not change at all after recomputation. Something is frozen!"
            print(f"[OHEM] Recompute Check -> Mean Loss: {self.train_loader.scores.mean().item():.4f}")

    def eval_noise_splits(self) -> Optional[dict]:
        if not hasattr(self.train_loader, "noise_mask") or self.train_loader.noise_mask is None:
            return None

        self.model.eval()
        clean_correct = 0
        clean_total = 0
        noisy_correct = 0
        noisy_total = 0
        restored_correct = 0

        with torch.no_grad(), autocast("cuda", dtype=self.precision, enabled=self.use_amp):
            for inputs, _, indices in self.train_loader.iterate_wo_aug(None):
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)

                clean_labels = self.train_loader.clean_labels[indices]
                noisy_labels = self.train_loader.noisy_labels[indices]
                noise_mask = self.train_loader.noise_mask[indices]

                clean_mask = ~noise_mask
                if clean_mask.any():
                    clean_correct += (preds[clean_mask] == clean_labels[clean_mask]).sum().item()
                    clean_total += clean_mask.sum().item()

                if noise_mask.any():
                    noisy_correct += (preds[noise_mask] == noisy_labels[noise_mask]).sum().item()
                    restored_correct += (preds[noise_mask] == clean_labels[noise_mask]).sum().item()
                    noisy_total += noise_mask.sum().item()

        clean_acc = (clean_correct / clean_total) * 100 if clean_total > 0 else float("nan")
        noisy_acc = (noisy_correct / noisy_total) * 100 if noisy_total > 0 else float("nan")
        restored_acc = (restored_correct / noisy_total) * 100 if noisy_total > 0 else float("nan")

        return {
            "clean_acc": clean_acc,
            "noisy_acc": noisy_acc,
            "restored_acc": restored_acc,
        }


def make_subset_path(expt_config) -> Path:
    # LOO/partial subset caches live under PRUNEPRIV_DATA (env-driven; no hardcoded absolutes).
    root = Path(os.environ.get("PRUNEPRIV_DATA", "data"))
    stem = "subsets_cifar" if expt_config.is_loo else "partial_subsets_cifar"
    return root / f"{stem}_{expt_config.num_subsets}.pt"


def load_subsets(expt_config):
    subset_path = make_subset_path(expt_config)
    if not subset_path.exists():
        if expt_config.is_loo:
            generate_subsets(expt_config.num_subsets, 50000, expt_config.sample_ratio_in_subsets, expt_config.base_seed, save_path=str(subset_path))
        else:
            fixed_indices = generate_partial_subsets(
                expt_config.num_subsets,
                50000,
                expt_config.partial_subset_size,
                expt_config.sample_ratio_in_subsets,
                expt_config.base_seed,
                save_path=str(subset_path),
            )
            torch.save(fixed_indices, f"fixed_indices_cifar_{expt_config.num_subsets}_subsets.pt")
    return torch.load(subset_path)


def maybe_make_output_dir(expt_config, seed: Optional[int] = None, run_id: Optional[int] = None) -> Optional[Path]:
    is_loo_run = expt_config.is_loo or expt_config.is_loo_with_partial_subsets
    needs_dir = (
        expt_config.save_outputs
        or (getattr(expt_config, "log_wandb", False) and not is_loo_run)
        or (getattr(expt_config, "log_sample_evolution", False) and not is_loo_run)
        or (getattr(expt_config, "log_noise_splits", True) and not is_loo_run)
    )
    if not needs_dir:
        return None

    experiment_name = str(getattr(expt_config, "experiment_name", "") or make_run_name(expt_config))

    if expt_config.is_loo or expt_config.is_loo_with_partial_subsets:
        base = Path(getattr(expt_config, "loo_root", None) or (SCRIPT_DIR / "LOO")) / experiment_name
        if expt_config.loo_output_path is not None:
            base = Path(expt_config.loo_output_path)
        base.mkdir(parents=True, exist_ok=True)
        assert run_id is not None
        output_dir = base / f"run_{run_id}"
    else:
        base = Path(getattr(expt_config, "non_loo_root", None) or (SCRIPT_DIR / "NON_LOO")) / experiment_name
        output_dir = base
        if seed is not None:
            output_dir = output_dir / f"seed_{seed}"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def print_metrics(console, table, unit_name, epoch, train_loss, test_loss, train_accuracy, test_accuracy):
    if table is not None and console is not None:
        table.add_row(
            str(epoch),
            f"{train_loss:.4f}",
            f"{test_loss:.4f}",
            f"{train_accuracy:.2f}%",
            f"{test_accuracy:.2f}%",
        )
        console.print(table)
    else:
        print(
            f"[{unit_name}] epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"train_acc={train_accuracy:.2f}% test_acc={test_accuracy:.2f}%"
        )


def select_active_indices(train_harness: TrainingHarness, expt_config, use_infobatch: bool, epoch: int):
    if expt_config.training_method == "eh":
        return train_harness.train_loader.perform_easy_hard(
            use_infobatch=use_infobatch,
            keep_ratio_easy=expt_config.ratio_easy,
            keep_ratio_hard=expt_config.ratio_hard,
        )
    if expt_config.training_method == "ib_std":
        return train_harness.train_loader.perform_infobatch_std(
            use_infobatch=use_infobatch,
            keep_ratio=expt_config.ratio_std,
            top_keep_fraction=expt_config.top_keep_fraction,
        )
    if expt_config.training_method in {"ib_3seg", "ib_3seg_prw_norm"}:
        return train_harness.train_loader.perform_three_segment(
            use_infobatch=use_infobatch,
            low_fraction=expt_config.segment_low_fraction,
            high_fraction=expt_config.segment_high_fraction,
            keep_low=expt_config.segment_keep_low,
            keep_mid=expt_config.segment_keep_mid,
            keep_high=expt_config.segment_keep_high,
        )
    if expt_config.training_method == "ib_3seg_static":
        return train_harness.train_loader.perform_static_three_segment(
            epoch=epoch,
            use_infobatch=use_infobatch,
            freeze_epoch=expt_config.static_segment_epoch,
            low_fraction=expt_config.segment_low_fraction,
            high_fraction=expt_config.segment_high_fraction,
            keep_low=expt_config.segment_keep_low,
            keep_mid=expt_config.segment_keep_mid,
            keep_high=expt_config.segment_keep_high,
        )
    if expt_config.training_method == "ib_risk3":
        if epoch < expt_config.risk_start_epoch:
            return train_harness.train_loader.perform_infobatch(
                use_infobatch=use_infobatch,
                keep_ratio=expt_config.ratio,
            )
        return train_harness.train_loader.perform_risk_three_band(
            use_infobatch=use_infobatch,
            low_percentile=expt_config.risk_low_percentile,
            high_percentile=expt_config.risk_high_percentile,
            keep_low=expt_config.risk_keep_low,
            keep_mid=expt_config.risk_keep_mid,
            keep_high=expt_config.risk_keep_high,
            rescale_low=expt_config.risk_rescale_low,
            rescale_mid=expt_config.risk_rescale_mid,
            rescale_high=expt_config.risk_rescale_high,
        )

    if expt_config.training_method == "ib_func_std":
        # Standard IB via functions: 50% keep for bottom half, 100% keep for top half
        prob_fn = lambda r: torch.where(r < 0.5, torch.tensor(0.5, device=r.device), torch.tensor(1.0, device=r.device))
        weight_fn = lambda p, b: 1.0 / p
        return train_harness.train_loader.perform_function_pruning(
            use_infobatch=use_infobatch, prob_fn=prob_fn, weight_fn=weight_fn, beta=1.0
        )

    if expt_config.training_method == "ib_u_batch":
        # U-Curve: rank 0 (easy) -> 0.5, rank 0.5 (mid) -> 0.25, rank 1.0 (hard) -> 1.0
        def u_curve_prob(r):
            p = 2.0 * (r ** 2) - 1.5 * r + 0.5
            return torch.clamp(p, min=0.01, max=1.0)
            
        weight_fn = lambda p, b: (1.0 / p) ** b

        # Phase 1 / Phase 2 Beta Scheduling for Triangular LR
        peak_epoch = int(0.2 * expt_config.num_epochs)
        if epoch <= peak_epoch:
            current_beta = 0.0 # Warmup: biased, safe gradients
        else:
            # Cooldown: linearly scale from 0.0 to 1.0 by the switch_epoch
            progress = (epoch - peak_epoch) / max(1, train_harness.switch_epoch - peak_epoch)
            current_beta = min(1.0, max(0.0, progress))

        return train_harness.train_loader.perform_function_pruning(
            use_infobatch=use_infobatch, prob_fn=u_curve_prob, weight_fn=weight_fn, beta=current_beta
        )

    if expt_config.training_method == "ib_random":
        return train_harness.train_loader.perform_random_prune(
            use_infobatch=use_infobatch,
            keep_ratio=expt_config.ratio,
        )

    if expt_config.training_method == "ib_prw":
        return train_harness.train_loader.perform_infobatch(
            use_infobatch=use_infobatch,
            keep_ratio=expt_config.ratio,
        )

    return train_harness.train_loader.perform_infobatch(
        use_infobatch=use_infobatch,
        keep_ratio=expt_config.ratio,
    )


def should_use_infobatch(epoch: int, train_harness: TrainingHarness, expt_config) -> bool:
    if not expt_config.use_ib:
        return False
    if expt_config.training_method in {"eh", "ib_std", "ib_3seg", "ib_3seg_prw_norm"}:
        return 0 < epoch <= train_harness.switch_epoch
    if expt_config.training_method == "ib_3seg_static":
        return expt_config.static_segment_epoch <= epoch <= train_harness.switch_epoch
    if expt_config.training_method == "ib_risk3":
        return expt_config.risk_warmup_epoch <= epoch <= train_harness.switch_epoch
    return epoch <= train_harness.switch_epoch


def get_status_map(expt_config):
    if expt_config.training_method in {"eh","ib_u_batch"}:
        return {0: "pruned", 1: "easy_kept", 2: "hard_kept", 3: "full_set_kept"}
    if expt_config.training_method in {"ib_3seg", "ib_3seg_static", "ib_3seg_prw_norm"}:
        return {0: "segment_pruned", 1: "low_kept", 2: "mid_or_high_kept", 3: "full_set_kept"}
    if expt_config.training_method == "ib_risk3":
        return {0: "risk_pruned", 1: "low_band_kept", 2: "risk_or_extreme_kept", 3: "full_set_kept"}
    return {0: "easy_pruned", 1: "easy_kept", 2: "hard_kept", 3: "full_set_kept"}


def get_status_field_names(expt_config):
    if expt_config.training_method in {"ib_3seg", "ib_3seg_static", "ib_3seg_prw_norm"}:
        return {
            0: "pruned_count",
            1: "low_kept_count",
            2: "mid_or_high_kept_count",
            3: "full_set_kept_count",
        }
    if expt_config.training_method == "ib_risk3":
        return {
            0: "pruned_count",
            1: "low_band_kept_count",
            2: "risk_or_extreme_kept_count",
            3: "full_set_kept_count",
        }
    return {
        0: "pruned_count",
        1: "easy_kept_count",
        2: "hard_kept_count",
        3: "full_set_kept_count",
    }


def get_status_metric_names(expt_config):
    if expt_config.training_method in {"ib_3seg", "ib_3seg_static", "ib_3seg_prw_norm"}:
        return {
            0: "status/pruned",
            1: "status/low_kept",
            2: "status/mid_or_high_kept",
            3: "status/full_set_kept",
        }
    if expt_config.training_method == "ib_risk3":
        return {
            0: "status/pruned",
            1: "status/low_band_kept",
            2: "status/risk_or_extreme_kept",
            3: "status/full_set_kept",
        }
    return {
        0: "status/pruned",
        1: "status/easy_kept",
        2: "status/hard_kept",
        3: "status/full_set_kept",
    }


def get_selection_score_metric_names(expt_config):
    if expt_config.training_method in {"ib_3seg", "ib_3seg_static", "ib_3seg_prw_norm"}:
        return {
            0: "selection/score_mean_pruned",
            1: "selection/score_mean_low_kept",
            2: "selection/score_mean_mid_or_high_kept",
        }
    if expt_config.training_method == "ib_risk3":
        return {
            0: "selection/score_mean_pruned",
            1: "selection/score_mean_low_band_kept",
            2: "selection/score_mean_risk_or_extreme_kept",
        }
    return {
        0: "selection/score_mean_pruned",
        1: "selection/score_mean_easy_kept",
        2: "selection/score_mean_hard_kept",
    }


def get_status_frac_names(expt_config):
    if expt_config.training_method in {"ib_3seg", "ib_3seg_static", "ib_3seg_prw_norm"}:
        return {
            0: "pruned_frac",
            1: "low_kept_frac",
            2: "mid_or_high_kept_frac",
            3: "full_set_kept_frac",
        }
    if expt_config.training_method == "ib_risk3":
        return {
            0: "pruned_frac",
            1: "low_band_kept_frac",
            2: "risk_or_extreme_kept_frac",
            3: "full_set_kept_frac",
        }
    return {
        0: "pruned_frac",
        1: "easy_kept_frac",
        2: "hard_kept_frac",
        3: "full_set_kept_frac",
    }


def get_noise_status_names(expt_config):
    if expt_config.training_method in {"ib_3seg", "ib_3seg_static", "ib_3seg_prw_norm"}:
        return {
            1: "low_kept",
            2: "mid_or_high_kept",
        }
    if expt_config.training_method == "ib_risk3":
        return {
            1: "low_band_kept",
            2: "risk_or_extreme_kept",
        }
    return {
        1: "easy_kept",
        2: "hard_kept",
    }


def make_run_name(expt_config, seed: Optional[int] = None, average: bool = False) -> str:
    base_name = str(getattr(expt_config, "experiment_name", None) or "")
    if not base_name:
        parts = [
            expt_config.dataset_name,
            expt_config.model_name,
            expt_config.training_method,
            expt_config.noise_type,
        ]
        if expt_config.training_method == "eh":
            parts.extend([f"easy_{expt_config.ratio_easy}", f"hard_{expt_config.ratio_hard}"])
        else:
            parts.append(f"ratio_{expt_config.ratio}")
        parts.append(str(expt_config.num_epochs))
        base_name = "_".join(parts)
    if seed is not None:
        base_name = f"{base_name}_seed_{seed}"
    if average:
        base_name = f"{base_name}_avg"
    return base_name


def compute_prw_weights(mode, prune_counts, scores, switch_epoch, expt_config, seed):
    """Per-sample annealing weights for a PRW method (computed once at the switch).

    mode:
      "pruning_rate"       (default, == WLIB): normalize(clamp(prune_counts/switch_epoch, eps))
      "loss_final"         : per-sample loss at the start of annealing (low loss -> high weight),
                             same clamp/normalize
      "shuffle"            : the pruning_rate weights, randomly permuted across samples (seeded)
      "hard_tercile_const" : terciles of cumulative pruning rate; lowest (hardest) -> eps, rest -> 1.0

    All modes keep WLIB's clamp(eps) floor and (optional) mean-normalization, so only the *signal*
    that produces the weight changes.
    """
    eps = float(getattr(expt_config, "prw_epsilon", 0.05))
    normalize = bool(getattr(expt_config, "prw_normalize_weights", False))
    pruning_rates = prune_counts / max(1.0, float(switch_epoch))

    if mode == "loss_final":
        s = scores.detach().float()
        s_min, s_max = s.min(), s.max()
        easiness = 1.0 - (s - s_min) / (s_max - s_min + 1e-12)   # low loss -> ~1 (like high prune rate)
        weights = torch.clamp(easiness, min=eps)
    elif mode == "shuffle":
        base = torch.clamp(pruning_rates, min=eps)
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + 90_000)
        perm = torch.randperm(base.numel(), generator=gen).to(base.device)
        weights = base[perm]
    elif mode == "hard_tercile_const":
        pr = pruning_rates.float()
        q1 = torch.quantile(pr, 1.0 / 3.0)                      # split cumulative prune rate into 3 tiers
        weights = torch.ones_like(pr)
        weights[pr <= q1] = eps                                 # lowest tercile (rarely pruned = hardest)
    else:  # "pruning_rate" == WLIB
        weights = torch.clamp(pruning_rates, min=eps)

    if normalize:
        weights = weights / weights.mean()
    print(f"[PRW] weights mode={mode} eps={eps} normalized={normalize}: "
          f"mean={weights.mean():.4f}, min={weights.min():.4f}, max={weights.max():.4f}")
    return weights


def run_single_training(expt_config, seed: int, subset_indices: Optional[torch.Tensor], unit_name: str, output_dir: Optional[Path]):
    set_seed(seed)
    train_harness = TrainingHarness(expt_config=expt_config, subset_indices=subset_indices)
    # Step-milestone checkpointing (apples-to-apples across methods that prune different #steps/epoch):
    # save step_<N>.pt when the global optimizer-step counter hits a milestone in cfg.save_at_steps.
    train_harness._global_step = 0
    train_harness._save_at_steps = set(int(s) for s in (getattr(expt_config, "save_at_steps", []) or []))
    train_harness._step_ckpt_dir = output_dir
    console = Console() if Console is not None else None
    table = make_metrics_table(f"Training Metrics for {unit_name}")
    is_loo_run = expt_config.is_loo or expt_config.is_loo_with_partial_subsets
    enable_wandb = bool(getattr(expt_config, "log_wandb", False) and not is_loo_run)
    enable_evolution = bool(getattr(expt_config, "log_sample_evolution", False) and not is_loo_run)
    enable_noise_logs = bool(getattr(expt_config, "log_noise_splits", True) and not is_loo_run)

    best_accuracy = 0.0
    history = []
    # Optional ablation hook (default None => no effect on any existing config):
    # halt the non-LOO training loop right after this epoch index, keeping the
    # original (full-length) LR schedule. Used by the "uncooled" early-stop
    # ablation to capture the model at the switch epoch before annealing, with
    # LR not yet cooled.
    early_stop_epoch = getattr(expt_config, "early_stop_epoch", None)
    total_samples = len(train_harness.train_loader.images)
    status_map = get_status_map(expt_config)
    status_field_names = get_status_field_names(expt_config)
    status_metric_names = get_status_metric_names(expt_config)
    selection_score_metric_names = get_selection_score_metric_names(expt_config)
    status_frac_names = get_status_frac_names(expt_config)
    noise_status_names = get_noise_status_names(expt_config)
    status_log = torch.full((total_samples,), 3, dtype=torch.int8, device=train_harness.device)
    status_counts_log = []
    noise_counts_log = []
    log_evolution_this_seed = bool(enable_evolution and seed == getattr(expt_config, "evolution_seed", expt_config.base_seed))
    evolution_log = []
    grad_sketch_store = {}          # epoch -> [N, k] fp16 per-sample gradient-direction sketches (non-LOO)
    evolution_status_counts = None
    evolution_transition_counts = None
    evolution_prev_status = None
    seed_train_loss = []
    seed_train_acc = []
    seed_test_loss = []
    seed_test_acc = []
    seed_clean_acc = []
    seed_noisy_acc = []
    seed_restored_acc = []
    start_time = time.time()

    if enable_wandb:
        if wandb is None:
            raise ImportError("log_wandb=True requires wandb to be installed.")
        wandb.init(
            project=getattr(expt_config, "wandb_project", f"{expt_config.dataset_name}_new"),
            entity=getattr(expt_config, "wandb_entity", None),
            group=getattr(expt_config, "wandb_group", None),
            tags=getattr(expt_config, "wandb_tags", None),
            config=vars(expt_config),
            name=make_run_name(expt_config, seed=seed),
            save_code=False,
            dir=str(output_dir if output_dir is not None else SCRIPT_DIR),
        )

    if not expt_config.use_ib_sampler:
        init_indices = torch.arange(len(train_harness.train_loader.images), device=train_harness.train_loader.device)
        init_indices = init_indices[torch.randperm(len(init_indices), device=train_harness.train_loader.device)]
        active_indices = init_indices.clone()

    is_prw_method = expt_config.training_method in {"ib_prw", "ib_3seg_prw_norm"}
    if is_prw_method:
        prune_counts = torch.zeros(total_samples, device=train_harness.device)
        prw_weights = None

    if log_evolution_this_seed and train_harness.held_out_indices_to_log is not None:
        held_out_indices = train_harness.held_out_indices_to_log
        held_out_labels = train_harness.train_loader.labels[held_out_indices].cpu().numpy()
        evolution_status_counts = torch.zeros(
            (len(held_out_indices), 4),
            dtype=torch.int32,
            device=train_harness.device,
        )
        evolution_transition_counts = torch.zeros(
            (len(held_out_indices),),
            dtype=torch.int32,
            device=train_harness.device,
        )
        evolution_prev_status = torch.full(
            (len(held_out_indices),),
            -1,
            dtype=torch.int16,
            device=train_harness.device,
        )
    else:
        held_out_indices = None
        held_out_labels = None

    if expt_config.is_loo or expt_config.is_loo_with_partial_subsets:
        for epoch in range(expt_config.num_epochs):
            log_dict = {}
            if expt_config.use_ib_sampler:
                raise ValueError("use_ib_sampler=True is not supported.")
            use_infobatch = should_use_infobatch(epoch, train_harness, expt_config)

            if not use_infobatch and not expt_config.use_ib_sampler:
                active_indices = init_indices.clone()
                active_indices = active_indices[torch.randperm(len(active_indices), device=train_harness.train_loader.device)]

            if use_infobatch and getattr(expt_config, "recompute_losses_before_ib", False):
                train_harness.recompute_all_losses()

            active_indices, status_log = select_active_indices(train_harness, expt_config, use_infobatch, epoch)

            if is_prw_method:
                if use_infobatch:
                    prune_counts += (status_log == 0).float()
                elif epoch > train_harness.switch_epoch:
                    if prw_weights is None:
                        prw_weights = compute_prw_weights(
                            getattr(expt_config, "prw_weight_mode", "pruning_rate"),
                            prune_counts, train_harness.train_loader.scores,
                            train_harness.switch_epoch, expt_config, seed)
                    train_harness.train_loader.weights[:] = prw_weights

            train_loss, train_accuracy = train_harness.train_one_epoch(epoch, active_indices)
            test_loss, test_accuracy = train_harness.test() if expt_config.evaluate_every_epoch else (float("nan"), float("nan"))
            noise_metrics = train_harness.eval_noise_splits() if enable_noise_logs else None

            effective_pruning_rate = compute_effective_pruning_rate(total_samples, int(active_indices.numel())) if use_infobatch else 0.0
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "effective_pruning_rate": effective_pruning_rate,
            }
            history.append(row)
            seed_train_loss.append(train_loss)
            seed_train_acc.append(train_accuracy)
            seed_test_loss.append(test_loss)
            seed_test_acc.append(test_accuracy)
            log_dict.update(
                {
                    "train/loss": train_loss,
                    "train/accuracy": train_accuracy,
                    "test/loss": test_loss,
                    "test/accuracy": test_accuracy,
                    "epoch": epoch,
                    "lr": train_harness.scheduler.get_last_lr()[0],
                    "pruning/effective_rate": effective_pruning_rate,
                }
            )
            if noise_metrics is not None:
                seed_clean_acc.append(noise_metrics["clean_acc"])
                seed_noisy_acc.append(noise_metrics["noisy_acc"])
                seed_restored_acc.append(noise_metrics["restored_acc"])
                log_dict.update(
                    {
                        "train/clean_accuracy": noise_metrics["clean_acc"],
                        "train/noisy_accuracy": noise_metrics["noisy_acc"],
                        "train/restored_accuracy": noise_metrics["restored_acc"],
                    }
                )

            counts = torch.bincount(status_log, minlength=4)
            counts_dict = {"epoch": epoch}
            for idx in range(4):
                counts_dict[status_field_names[idx]] = counts[idx].item()
            status_counts_log.append(counts_dict)
            log_dict.update({status_metric_names[idx]: counts_dict[status_field_names[idx]] for idx in range(4)})
            if not expt_config.use_ib_sampler:
                scores = train_harness.train_loader.scores
                active_ratio = int(active_indices.numel()) / max(1, total_samples) if use_infobatch else 1.0
                log_dict.update(
                    {
                        "selection/active_ratio": active_ratio,
                        selection_score_metric_names[1]: float(scores[status_log == 1].mean().item()) if (status_log == 1).any() else float("nan"),
                        selection_score_metric_names[2]: float(scores[status_log == 2].mean().item()) if (status_log == 2).any() else float("nan"),
                        selection_score_metric_names[0]: float(scores[status_log == 0].mean().item()) if (status_log == 0).any() else float("nan"),
                    }
                )

            if log_evolution_this_seed and held_out_indices is not None:
                all_losses, all_activations, all_grad_norms, all_entropies, _all_grad_sketches = train_harness.get_evolution_data(epoch)
                log_losses = all_losses[held_out_indices].cpu()
                log_activations = all_activations[held_out_indices].float().cpu() if all_activations is not None else None
                log_grad_norms = all_grad_norms[held_out_indices].cpu() if all_grad_norms is not None else None
                log_statuses = status_log[held_out_indices].cpu()
                log_entropies = all_entropies[held_out_indices].cpu()
                for i, s in enumerate(log_statuses):
                    si = int(s.item())
                    evolution_status_counts[i, si] += 1
                    prev = int(evolution_prev_status[i].item())
                    if prev != -1 and prev != si:
                        evolution_transition_counts[i] += 1
                    evolution_prev_status[i] = si
                status_fracs = (evolution_status_counts.float() / float(epoch + 1)).cpu()
                sample_ids = train_harness.train_loader.sample_ids[held_out_indices].cpu().numpy()
                for i, sample_idx in enumerate(sample_ids):
                    record = {
                        "epoch": epoch,
                        "sample_index": int(sample_idx),
                        "loss": log_losses[i].item(),
                        "uncertainty_entropy": log_entropies[i].item(),
                        "ib_status": status_map[log_statuses[i].item()],
                        "ib_status_frac": {status_frac_names[idx]: float(status_fracs[i, idx].item()) for idx in range(4)},
                        "ib_status_transitions": int(evolution_transition_counts[i].item()),
                    }
                    if log_activations is not None:
                        record["activation"] = log_activations[i].numpy()
                    if log_grad_norms is not None:
                        record["grad_norm_sq"] = log_grad_norms[i].item()
                    evolution_log.append(record)
                if enable_wandb:
                    log_dict["held_out/loss_histogram"] = wandb.Histogram(log_losses.detach().numpy())

            if enable_wandb:
                wandb.log(log_dict, step=epoch)
            print_metrics(console, table, unit_name, epoch, train_loss, test_loss, train_accuracy, test_accuracy)

            if output_dir is not None and expt_config.save_best_checkpoint and test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(train_harness.model.state_dict(), output_dir / "best_epoch.pt")
            if output_dir is not None and expt_config.save_epoch_checkpoints and epoch in expt_config.sample_at_epoch:
                torch.save(train_harness.model.state_dict(), output_dir / f"epoch_{epoch}.pt")
    else:
        for epoch in range(expt_config.num_epochs):
            log_dict = {}

            use_infobatch = should_use_infobatch(epoch, train_harness, expt_config)

            if not use_infobatch and not expt_config.use_ib_sampler:
                active_indices = init_indices.clone()
                active_indices = active_indices[torch.randperm(len(active_indices), device=train_harness.train_loader.device)]

            if expt_config.use_ib_sampler:
                raise ValueError("use_ib_sampler=True is not supported.")

            if use_infobatch and getattr(expt_config, "recompute_losses_before_ib", False):
                train_harness.recompute_all_losses()

            active_indices, status_log = select_active_indices(train_harness, expt_config, use_infobatch, epoch)

            if is_prw_method:
                if use_infobatch:
                    prune_counts += (status_log == 0).float()
                elif epoch > train_harness.switch_epoch:
                    if prw_weights is None:
                        prw_weights = compute_prw_weights(
                            getattr(expt_config, "prw_weight_mode", "pruning_rate"),
                            prune_counts, train_harness.train_loader.scores,
                            train_harness.switch_epoch, expt_config, seed)
                    train_harness.train_loader.weights[:] = prw_weights

            train_loss, train_accuracy = train_harness.train_one_epoch(epoch, active_indices)
            test_loss, test_accuracy = train_harness.test() if expt_config.evaluate_every_epoch else (float("nan"), float("nan"))
            noise_metrics = train_harness.eval_noise_splits() if enable_noise_logs else None

            effective_pruning_rate = compute_effective_pruning_rate(total_samples, int(active_indices.numel())) if use_infobatch else 0.0
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "effective_pruning_rate": effective_pruning_rate,
            }
            history.append(row)
            seed_train_loss.append(train_loss)
            seed_train_acc.append(train_accuracy)
            seed_test_loss.append(test_loss)
            seed_test_acc.append(test_accuracy)
            log_dict.update(
                {
                    "train/loss": train_loss,
                    "train/accuracy": train_accuracy,
                    "test/loss": test_loss,
                    "test/accuracy": test_accuracy,
                    "epoch": epoch,
                    "lr": train_harness.scheduler.get_last_lr()[0],
                    "pruning/effective_rate": effective_pruning_rate,
                }
            )
            if noise_metrics is not None:
                seed_clean_acc.append(noise_metrics["clean_acc"])
                seed_noisy_acc.append(noise_metrics["noisy_acc"])
                seed_restored_acc.append(noise_metrics["restored_acc"])
                log_dict.update(
                    {
                        "train/clean_accuracy": noise_metrics["clean_acc"],
                        "train/noisy_accuracy": noise_metrics["noisy_acc"],
                        "train/restored_accuracy": noise_metrics["restored_acc"],
                    }
                )

            noise_mask = getattr(train_harness.train_loader, "noise_mask", None)
            if noise_mask is not None:
                noisy_count = int(noise_mask[active_indices].sum().item())
                clean_count = int(active_indices.numel() - noisy_count)
                total_noisy = int(noise_mask.sum().item())
                total_clean = int(noise_mask.numel() - total_noisy)
                noisy_status2 = int((noise_mask & (status_log == 2)).sum().item())
                noisy_status1 = int((noise_mask & (status_log == 1)).sum().item())
                clean_status2 = int(((~noise_mask) & (status_log == 2)).sum().item())
                clean_status1 = int(((~noise_mask) & (status_log == 1)).sum().item())
                noise_counts_log.append(
                    {
                        "epoch": epoch,
                        "noisy_active": noisy_count,
                        "clean_active": clean_count,
                        "active_total": int(active_indices.numel()),
                        "noisy_total": total_noisy,
                        "clean_total": total_clean,
                        f"noisy_{noise_status_names[2]}": noisy_status2,
                        f"noisy_{noise_status_names[1]}": noisy_status1,
                        f"clean_{noise_status_names[2]}": clean_status2,
                        f"clean_{noise_status_names[1]}": clean_status1,
                    }
                )
                log_dict.update(
                    {
                        "selection/noisy_active": noisy_count,
                        "selection/clean_active": clean_count,
                        "selection/active_total": int(active_indices.numel()),
                        "selection/noisy_total": total_noisy,
                        "selection/clean_total": total_clean,
                        f"selection/noisy_{noise_status_names[2]}": noisy_status2,
                        f"selection/noisy_{noise_status_names[1]}": noisy_status1,
                        f"selection/clean_{noise_status_names[2]}": clean_status2,
                        f"selection/clean_{noise_status_names[1]}": clean_status1,
                    }
                )

            counts = torch.bincount(status_log, minlength=4)
            counts_dict = {"epoch": epoch}
            for idx in range(4):
                counts_dict[status_field_names[idx]] = counts[idx].item()
            status_counts_log.append(counts_dict)
            log_dict.update({status_metric_names[idx]: counts_dict[status_field_names[idx]] for idx in range(4)})
            if not expt_config.use_ib_sampler:
                scores = train_harness.train_loader.scores
                active_ratio = int(active_indices.numel()) / max(1, total_samples) if use_infobatch else 1.0
                log_dict.update(
                    {
                        "selection/active_ratio": active_ratio,
                        selection_score_metric_names[1]: float(scores[status_log == 1].mean().item()) if (status_log == 1).any() else float("nan"),
                        selection_score_metric_names[2]: float(scores[status_log == 2].mean().item()) if (status_log == 2).any() else float("nan"),
                        selection_score_metric_names[0]: float(scores[status_log == 0].mean().item()) if (status_log == 0).any() else float("nan"),
                    }
                )

            if log_evolution_this_seed and held_out_indices is not None:
                all_losses, all_activations, all_grad_norms, all_entropies, all_grad_sketches = train_harness.get_evolution_data(epoch)
                log_losses = all_losses[held_out_indices].cpu()
                log_activations = all_activations[held_out_indices].float().cpu() if all_activations is not None else None
                log_grad_norms = all_grad_norms[held_out_indices].cpu() if all_grad_norms is not None else None
                log_statuses = status_log[held_out_indices].cpu()
                log_entropies = all_entropies[held_out_indices].cpu()
                if all_grad_sketches is not None:
                    # per-sample gradient-direction sketch at this (sampled) epoch, for per-group alignment
                    grad_sketch_store[epoch] = all_grad_sketches[held_out_indices].to(torch.float16).cpu().numpy()
                for i, s in enumerate(log_statuses):
                    si = int(s.item())
                    evolution_status_counts[i, si] += 1
                    prev = int(evolution_prev_status[i].item())
                    if prev != -1 and prev != si:
                        evolution_transition_counts[i] += 1
                    evolution_prev_status[i] = si
                status_fracs = (evolution_status_counts.float() / float(epoch + 1)).cpu()
                sample_ids = train_harness.train_loader.sample_ids[held_out_indices].cpu().numpy()
                for i, sample_idx in enumerate(sample_ids):
                    record = {
                        "epoch": epoch,
                        "sample_index": int(sample_idx),
                        "loss": log_losses[i].item(),
                        "uncertainty_entropy": log_entropies[i].item(),
                        "ib_status": status_map[log_statuses[i].item()],
                        "ib_status_frac": {status_frac_names[idx]: float(status_fracs[i, idx].item()) for idx in range(4)},
                        "ib_status_transitions": int(evolution_transition_counts[i].item()),
                    }
                    if log_activations is not None:
                        record["activation"] = log_activations[i].numpy()
                    if log_grad_norms is not None:
                        record["grad_norm_sq"] = log_grad_norms[i].item()
                    evolution_log.append(record)
                if enable_wandb:
                    log_dict["held_out/loss_histogram"] = wandb.Histogram(log_losses.detach().numpy())

            if enable_wandb:
                wandb.log(log_dict, step=epoch)
            print_metrics(console, table, unit_name, epoch, train_loss, test_loss, train_accuracy, test_accuracy)

            if output_dir is not None and expt_config.save_best_checkpoint and test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(train_harness.model.state_dict(), output_dir / "best_epoch.pt")
            if output_dir is not None and expt_config.save_epoch_checkpoints and epoch in expt_config.sample_at_epoch:
                torch.save(train_harness.model.state_dict(), output_dir / f"epoch_{epoch}.pt")

            if early_stop_epoch is not None and epoch >= early_stop_epoch:
                # Uncooled early-stop ablation: halt before annealing; last_epoch.pt
                # below is saved from this (switch-epoch) model, LR not yet cooled.
                print(f"[{unit_name}] early stop after epoch {epoch} (early_stop_epoch={early_stop_epoch}).")
                break

    if output_dir is not None and expt_config.save_last_checkpoint:
        torch.save(train_harness.model.state_dict(), output_dir / "last_epoch.pt")

    # Save the per-sample cumulative pruning rate (ρ_i signal) for PRW methods — used by the
    # hard-group subgroup analysis to bin samples by hardness.
    if output_dir is not None and (not is_loo_run) and is_prw_method:
        prune_rate = (prune_counts / max(1.0, float(train_harness.switch_epoch))).detach().cpu().numpy()
        np.save(output_dir / "prune_rate.npy", prune_rate)

    if output_dir is not None and (not is_loo_run) and log_evolution_this_seed and evolution_log:
        torch.save(evolution_log, output_dir / f"cifar_evolution_log_{expt_config.training_method}.pt")
    if output_dir is not None and (not is_loo_run) and grad_sketch_store:
        # Per-sample gradient-direction sketches: [N, n_sampled_epochs, k] fp16 + the epoch order.
        eps_sorted = sorted(grad_sketch_store.keys())
        sketch_arr = np.stack([grad_sketch_store[e] for e in eps_sorted], axis=1).astype(np.float16)
        np.save(output_dir / f"grad_sketch_{expt_config.training_method}.npy", sketch_arr)
        np.save(output_dir / f"grad_sketch_{expt_config.training_method}_epochs.npy", np.array(eps_sorted, dtype=np.int64))
    if output_dir is not None and (not is_loo_run) and status_counts_log:
        torch.save(status_counts_log, output_dir / f"cifar_status_counts_log_{expt_config.training_method}_seed_{seed}.pt")
    if output_dir is not None and (not is_loo_run) and noise_counts_log:
        torch.save(noise_counts_log, output_dir / f"cifar_noise_counts_log_{expt_config.training_method}_seed_{seed}.pt")

    final_prune = history[-1]["effective_pruning_rate"] if history else 0.0
    avg_prune = float(np.mean([row["effective_pruning_rate"] for row in history])) if history else 0.0
    total_time = time.time() - start_time
    if console is not None:
        console.print(f"Final effective pruning rate for {unit_name}: {final_prune * 100:.2f}% (avg {avg_prune * 100:.2f}%)")
        console.print(f"Total Training Time: {total_time:.2f} seconds for {expt_config.num_epochs} epochs.")
    else:
        print(f"[{unit_name}] final_effective_pruning_rate={final_prune * 100:.2f}% avg_effective_pruning_rate={avg_prune * 100:.2f}%")
        print(f"[{unit_name}] total_training_time={total_time:.2f}s")

    if enable_wandb:
        wandb.finish()

    return {
        "history": history,
        "train_loss": seed_train_loss,
        "train_accuracy": seed_train_acc,
        "test_loss": seed_test_loss,
        "test_accuracy": seed_test_acc,
        "effective_pruning_rate": [row["effective_pruning_rate"] for row in history],
        "final_effective_pruning_rate": final_prune,
        "avg_effective_pruning_rate": avg_prune,
        "clean_accuracy": seed_clean_acc,
        "noisy_accuracy": seed_noisy_acc,
        "restored_accuracy": seed_restored_acc,
    }


def run_standard_or_ib(expt_config):
    all_train_loss = []
    all_train_acc = []
    all_test_loss = []
    all_test_acc = []
    all_pruning_rate = []
    all_clean_acc = []
    all_noisy_acc = []
    all_restored_acc = []
    root_output_dir = maybe_make_output_dir(expt_config)

    for seed in expt_config.seed_list:
        output_dir = maybe_make_output_dir(expt_config, seed=seed)
        seed_result = run_single_training(expt_config, seed=seed, subset_indices=None, unit_name=f"seed-{seed}", output_dir=output_dir)
        all_train_loss.append(seed_result["train_loss"])
        all_train_acc.append(seed_result["train_accuracy"])
        all_test_loss.append(seed_result["test_loss"])
        all_test_acc.append(seed_result["test_accuracy"])
        all_pruning_rate.append(seed_result["effective_pruning_rate"])
        if seed_result["clean_accuracy"]:
            all_clean_acc.append(seed_result["clean_accuracy"])
            all_noisy_acc.append(seed_result["noisy_accuracy"])
            all_restored_acc.append(seed_result["restored_accuracy"])

    if not all_train_loss:
        return

    avg_train_loss = np.mean(np.stack(all_train_loss), axis=0)
    avg_train_acc = np.mean(np.stack(all_train_acc), axis=0)
    avg_test_loss = np.mean(np.stack(all_test_loss), axis=0)
    avg_test_acc = np.mean(np.stack(all_test_acc), axis=0)
    avg_pruning_rate = np.mean(np.stack(all_pruning_rate), axis=0)
    avg_clean_acc = np.mean(np.stack(all_clean_acc), axis=0) if all_clean_acc else None
    avg_noisy_acc = np.mean(np.stack(all_noisy_acc), axis=0) if all_noisy_acc else None
    avg_restored_acc = np.mean(np.stack(all_restored_acc), axis=0) if all_restored_acc else None
    avg_best_test_acc = float(np.mean([max(x) for x in all_test_acc]))
    avg_last_test_acc = float(np.mean([x[-1] for x in all_test_acc]))
    avg_final_pruning_rate = float(np.mean([x[-1] for x in all_pruning_rate]))
    avg_mean_pruning_rate = float(np.mean([np.mean(x) for x in all_pruning_rate]))

    summary = {
        "seed_list": expt_config.seed_list,
        "train_loss": all_train_loss,
        "train_accuracy": all_train_acc,
        "test_loss": all_test_loss,
        "test_accuracy": all_test_acc,
        "effective_pruning_rate": all_pruning_rate,
        "avg/train_loss": avg_train_loss,
        "avg/train_accuracy": avg_train_acc,
        "avg/test_loss": avg_test_loss,
        "avg/test_accuracy": avg_test_acc,
        "avg/effective_pruning_rate": avg_pruning_rate,
        "avg/best_test_accuracy": avg_best_test_acc,
        "avg/last_test_accuracy": avg_last_test_acc,
        "avg/final_effective_pruning_rate": avg_final_pruning_rate,
        "avg/mean_effective_pruning_rate": avg_mean_pruning_rate,
    }
    if all_clean_acc:
        summary.update(
            {
                "train/clean_accuracy": all_clean_acc,
                "train/noisy_accuracy": all_noisy_acc,
                "train/restored_accuracy": all_restored_acc,
                "avg/train/clean_accuracy": avg_clean_acc,
                "avg/train/noisy_accuracy": avg_noisy_acc,
                "avg/train/restored_accuracy": avg_restored_acc,
            }
        )
    if root_output_dir is not None:
        torch.save(summary, root_output_dir / "seed_avg.pt")

    if getattr(expt_config, "log_wandb", False):
        if wandb is None:
            raise ImportError("log_wandb=True requires wandb to be installed.")
        wandb.init(
            project=getattr(expt_config, "wandb_project", f"{expt_config.dataset_name}_new"),
            entity=getattr(expt_config, "wandb_entity", None),
            group=getattr(expt_config, "wandb_group", None),
            tags=getattr(expt_config, "wandb_tags", None),
            config=vars(expt_config),
            name=make_run_name(expt_config, average=True),
            save_code=False,
            dir=str(root_output_dir if root_output_dir is not None else SCRIPT_DIR),
        )
        for epoch in range(len(avg_train_loss)):
            avg_log = {
                "avg/train_loss": avg_train_loss[epoch],
                "avg/train_accuracy": avg_train_acc[epoch],
                "avg/test_loss": avg_test_loss[epoch],
                "avg/test_accuracy": avg_test_acc[epoch],
                "avg/pruning/effective_rate": avg_pruning_rate[epoch],
                "epoch": epoch,
            }
            if avg_clean_acc is not None:
                avg_log.update(
                    {
                        "avg/train/clean_accuracy": avg_clean_acc[epoch],
                        "avg/train/noisy_accuracy": avg_noisy_acc[epoch],
                        "avg/train/restored_accuracy": avg_restored_acc[epoch],
                    }
                )
            wandb.log(avg_log, step=epoch)
        wandb.log(
            {
                "avg/best_test_accuracy": avg_best_test_acc,
                "avg/last_test_accuracy": avg_last_test_acc,
                "avg/final_effective_pruning_rate": avg_final_pruning_rate,
                "avg/mean_effective_pruning_rate": avg_mean_pruning_rate,
            }
        )
        wandb.finish()


def run_loo(expt_config, start_run: int, end_run: Optional[int]):
    subsets = load_subsets(expt_config)
    end_run = expt_config.num_subsets if end_run is None else min(end_run, expt_config.num_subsets)
    for run_id in range(start_run, end_run):
        subset_indices = subsets[run_id].long()
        output_dir = maybe_make_output_dir(expt_config, run_id=run_id)
        run_single_training(
            expt_config,
            seed=expt_config.base_seed + run_id,
            subset_indices=subset_indices,
            unit_name=f"run-{run_id}",
            output_dir=output_dir,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-run", "--start_run", dest="start_run", type=int, default=0, help="First run_id to execute")
    parser.add_argument("--end-run", "--end_run", dest="end_run", type=int, default=None, help="Last run_id to execute (exclusive)")
    args = parser.parse_args()

    expt_config = build_config()
    expt_config.num_classes = 10 if expt_config.dataset_name == "cifar10" else 100
    print(expt_config)

    if expt_config.is_loo or expt_config.is_loo_with_partial_subsets:
        run_loo(expt_config, start_run=args.start_run, end_run=args.end_run)
    else:
        run_standard_or_ib(expt_config)


if __name__ == "__main__":
    main()
