import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = os.environ.get("PRUNEPRIV_DATA", "data")
DEFAULT_NOISE_FILE = ""


def _clean_token(value: str) -> str:
    return str(value).replace("/", "-").replace(" ", "-")


def build_experiment_name(cfg) -> str:
    parts = [cfg.dataset_name, cfg.model_name, cfg.training_method]
    parts.append(f"bs{cfg.batch_size}")
    parts.append(f"lr{cfg.learning_rate:g}")
    if cfg.training_method == "eh":
        parts.append(f"easy{cfg.ratio_easy:.3f}")
        parts.append(f"hard{cfg.ratio_hard:.3f}")
    elif cfg.training_method == "ib_std":
        parts.append(f"ratio{cfg.ratio_std:.3f}")
        parts.append(f"top{cfg.top_keep_fraction:.3f}")
    elif cfg.training_method in {"ib_3seg", "ib_3seg_static", "ib_3seg_prw_norm"}:
        parts.append(f"lowf{cfg.segment_low_fraction:.3f}")
        parts.append(f"highf{cfg.segment_high_fraction:.3f}")
        parts.append(f"keepl{cfg.segment_keep_low:.3f}")
        parts.append(f"keepm{cfg.segment_keep_mid:.3f}")
        parts.append(f"keeph{cfg.segment_keep_high:.3f}")
        if cfg.training_method == "ib_3seg_static":
            parts.append(f"fixep{cfg.static_segment_epoch}")
    elif cfg.training_method == "ib_risk3":
        parts.append(f"lowp{cfg.risk_low_percentile:.3f}")
        parts.append(f"highp{cfg.risk_high_percentile:.3f}")
        parts.append(f"keepl{cfg.risk_keep_low:.3f}")
        parts.append(f"keepr{cfg.risk_keep_mid:.3f}")
        parts.append(f"keeph{cfg.risk_keep_high:.3f}")
        parts.append(f"rsl{int(cfg.risk_rescale_low)}")
        parts.append(f"rsm{int(cfg.risk_rescale_mid)}")
        parts.append(f"rsh{int(cfg.risk_rescale_high)}")
        parts.append(f"warm{cfg.risk_warmup_epoch}")
        parts.append(f"start{cfg.risk_start_epoch}")
    elif cfg.training_method in {"ib", "ib_focal", "ib_focal_late", "ib_prw"}:
        parts.append(f"ratio{cfg.ratio:.3f}")
    parts.append(f"delta{cfg.delta:.3f}")
    parts.append(f"ep{cfg.num_epochs}")
    if getattr(cfg, "grad_clip", False):
        parts.append(f"clip{cfg.grad_clip_max_norm:g}")
    if cfg.use_focal_loss or cfg.train_loss == "focal" or cfg.score_loss == "focal":
        parts.append(f"train{cfg.train_loss}")
        parts.append(f"score{cfg.score_loss}")
        parts.append(f"gamma{cfg.focal_gamma:g}")
        if cfg.focal_in_annealing_only:
            parts.append("focalanneal")
    if getattr(cfg, "prw_normalize_weights", False):
        parts.append("prwnorm")
    if cfg.noise_type != "clean":
        parts.append(_clean_token(cfg.noise_type))
    if cfg.is_loo:
        parts.append(f"loo{cfg.num_subsets}")
        parts.append(f"sr{cfg.sample_ratio_in_subsets:.2f}")
    elif cfg.is_loo_with_partial_subsets:
        parts.append(f"ploo{cfg.num_subsets}")
        parts.append(f"ss{cfg.partial_subset_size}")
        parts.append(f"sr{cfg.sample_ratio_in_subsets:.2f}")
    return "_".join(parts)


@dataclass
class Cifar10BaseConfig:
    data_root_dir: str = DEFAULT_DATA_ROOT
    dataset_name: str = "cifar10"
    batch_size: int = 512
    num_classes: int = 10
    train_aug: dict = field(default_factory=lambda: {"flip": True, "translate": 2})
    altflip: bool = False

    learning_rate: float = 0.2
    weight_decay: float = 5e-4
    momentum: float = 0.9
    optimizer_name: str = "sgd"

    model_name: str = "r20"
    num_epochs: int = 100
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    training_precision: str = "auto"
    use_compile: bool = True

    base_seed: int = 3407 #42 was used for loo
    seed_list: List[int] = field(default_factory=lambda: [0, 1, 42, 1234, 3407])
    evolution_seed: int = 3407

    noise_type: str = "clean"
    noise_file_path: str = DEFAULT_NOISE_FILE

    training_method: str = "standard"
    use_ib: bool = False
    delta: float = 1 #0.875
    grad_clip: bool = False
    grad_clip_max_norm: float = 1.0
    ratio: float = 0.5
    ratio_easy: float = 0.5
    ratio_hard: float = 1.0
    ratio_std: float = 0.35
    top_keep_fraction: float = 0.55
    std_factor: float = 1.0
    segment_low_fraction: float = 0.15
    segment_high_fraction: float = 0.15
    segment_keep_low: float = 1.0
    segment_keep_mid: float = 0.5
    segment_keep_high: float = 1.0
    static_segment_epoch: int = 10
    risk_low_percentile: float = 0.75
    risk_high_percentile: float = 0.95
    risk_keep_low: float = 0.5
    risk_keep_mid: float = 0.35
    risk_keep_high: float = 0.85
    risk_rescale_low: bool = True
    risk_rescale_mid: bool = True
    risk_rescale_high: bool = True
    risk_warmup_epoch: int = 10
    risk_start_epoch: int = 25
    recompute_losses_before_ib: bool = False

    train_loss: str = "ce"
    score_loss: str = "ce"
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_in_annealing_only: bool = False

    num_subsets: int = 200
    sample_ratio_in_subsets: float = 0.7
    partial_subset_size: int = 5000
    is_loo: bool = False
    is_loo_with_partial_subsets: bool = False

    # sample_at_epoch: List[int] = field(default_factory=lambda: [10, 18, 25, 50, 57, 73, 75, 87, 88, 99])
    # evaluate_every_epoch: bool = True
    # save_outputs: bool = False #True
    # save_best_checkpoint: bool = False #True
    # save_last_checkpoint: bool = False #True
    # save_epoch_checkpoints: bool = False #True

    # log_sample_evolution: bool = False #True
    # log_evolution_activations: bool = False
    # log_evolution_grad_norms: bool = False
    # use_fixed_indices_for_analysis: bool = False
    # log_noise_splits: bool = False #True
    # log_wandb: bool = True
    # wandb_project: str = "cifar_new"
    # wandb_entity: Optional[str] = None
    # wandb_group: Optional[str] = None
    # wandb_tags: Optional[List[str]] = None

    sample_at_epoch: List[int] = field(default_factory=lambda: [10, 18, 25, 50, 57, 73, 75, 87, 88, 99])
    evaluate_every_epoch: bool = True
    save_outputs: bool = True
    save_best_checkpoint: bool = True
    save_last_checkpoint: bool = True
    save_epoch_checkpoints: bool = True

    log_sample_evolution: bool = True
    log_evolution_activations: bool = False
    log_evolution_grad_norms: bool = False
    use_fixed_indices_for_analysis: bool = False
    log_noise_splits: bool = True
    log_wandb: bool = True
    wandb_project: str = "cifar_new"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None

    use_ib_sampler: bool = False

    experiment_name: Optional[str] = None
    loo_output_path: Optional[str] = None
    non_loo_root: Optional[str] = None
    loo_root: Optional[str] = None

    def __post_init__(self):
        if self.is_loo and self.is_loo_with_partial_subsets:
            raise ValueError("Both is_loo and is_loo_with_partial_subsets cannot be True at the same time.")
        self.num_classes = 10
        self.ratio = float(self.ratio)
        self.grad_clip = bool(self.grad_clip)
        self.grad_clip_max_norm = float(self.grad_clip_max_norm)
        if self.grad_clip_max_norm <= 0.0:
            raise ValueError("grad_clip_max_norm must be positive.")
        self.ratio_easy = float(self.ratio_easy)
        self.ratio_hard = float(self.ratio_hard)
        self.ratio_std = float(self.ratio_std)
        self.top_keep_fraction = float(self.top_keep_fraction)
        self.segment_low_fraction = float(self.segment_low_fraction)
        self.segment_high_fraction = float(self.segment_high_fraction)
        self.segment_keep_low = float(self.segment_keep_low)
        self.segment_keep_mid = float(self.segment_keep_mid)
        self.segment_keep_high = float(self.segment_keep_high)
        self.static_segment_epoch = int(self.static_segment_epoch)
        self.risk_low_percentile = float(self.risk_low_percentile)
        self.risk_high_percentile = float(self.risk_high_percentile)
        self.risk_keep_low = float(self.risk_keep_low)
        self.risk_keep_mid = float(self.risk_keep_mid)
        self.risk_keep_high = float(self.risk_keep_high)
        self.risk_rescale_low = bool(self.risk_rescale_low)
        self.risk_rescale_mid = bool(self.risk_rescale_mid)
        self.risk_rescale_high = bool(self.risk_rescale_high)
        self.risk_warmup_epoch = int(self.risk_warmup_epoch)
        self.risk_start_epoch = int(self.risk_start_epoch)
        if not (0.0 <= self.risk_low_percentile <= self.risk_high_percentile <= 1.0):
            raise ValueError("risk_low_percentile and risk_high_percentile must satisfy 0 <= low <= high <= 1.")
        if self.risk_start_epoch < self.risk_warmup_epoch:
            raise ValueError("risk_start_epoch must be greater than or equal to risk_warmup_epoch.")
        for keep_value, keep_name in (
            (self.risk_keep_low, "risk_keep_low"),
            (self.risk_keep_mid, "risk_keep_mid"),
            (self.risk_keep_high, "risk_keep_high"),
        ):
            if keep_value < 0.0:
                raise ValueError(f"{keep_name} must be non-negative.")
        self.experiment_name = self.experiment_name or build_experiment_name(self)
        self.non_loo_root = self.non_loo_root or str(SCRIPT_DIR / "NON_LOO")
        self.loo_root = self.loo_root or str(SCRIPT_DIR / "LOO")
        if self.loo_output_path is None and (self.is_loo or self.is_loo_with_partial_subsets):
            self.loo_output_path = str(Path(self.loo_root) / self.experiment_name)
        if self.wandb_group is None:
            self.wandb_group = self.experiment_name
        if self.wandb_tags is None:
            self.wandb_tags = [self.dataset_name, self.model_name, self.training_method]
            if self.noise_type != "clean":
                self.wandb_tags.append(_clean_token(self.noise_type))


@dataclass
class Cifar10StandardConfig(Cifar10BaseConfig):
    training_method: str = "standard"
    use_ib: bool = False


@dataclass
class Cifar10IBConfig(Cifar10BaseConfig):
    training_method: str = "ib"
    use_ib: bool = True
    ratio: float = 0.5
    ratio_easy: float = 0.5
    ratio_hard: float = 1.0


@dataclass
class Cifar10EHConfig(Cifar10BaseConfig):
    training_method: str = "eh"
    use_ib: bool = True
    ratio: float = 0.5
    ratio_easy: float = 0.55
    ratio_hard: float = 0.75


@dataclass
class Cifar10IBFocalConfig(Cifar10IBConfig):
    training_method: str = "ib_focal"
    use_ib: bool = True
    train_loss: str = "focal"
    score_loss: str = "focal"
    use_focal_loss: bool = True
    focal_in_annealing_only: bool = False


@dataclass
class Cifar10IBFocalLateConfig(Cifar10IBConfig):
    training_method: str = "ib_focal_late"
    use_ib: bool = True
    train_loss: str = "focal"
    score_loss: str = "focal"
    use_focal_loss: bool = True
    focal_in_annealing_only: bool = True


@dataclass
class Cifar10IBStdConfig(Cifar10IBConfig):
    training_method: str = "ib_std"
    use_ib: bool = True
    ratio_std: float = 0.15
    top_keep_fraction: float = 0.52


@dataclass
class Cifar10IB3SegConfig(Cifar10IBConfig):
    training_method: str = "ib_3seg"
    use_ib: bool = True
    segment_low_fraction: float = 0.0
    segment_high_fraction: float = 0.60
    segment_keep_low: float = 0.0
    segment_keep_mid: float = 0.0
    segment_keep_high: float = 1.0


@dataclass
class Cifar10IB3SegStaticConfig(Cifar10IBConfig):
    training_method: str = "ib_3seg_static"
    use_ib: bool = True
    segment_low_fraction: float = 0.15
    segment_high_fraction: float = 0.35
    segment_keep_low: float = 0.65
    segment_keep_mid: float = 0.2
    segment_keep_high: float = 1.0
    static_segment_epoch: int = 10


@dataclass
class Cifar10IBRisk3Config(Cifar10IBConfig):
    training_method: str = "ib_risk3"
    use_ib: bool = True
    grad_clip: bool = False
    grad_clip_max_norm: float = 1.0
    risk_low_percentile: float = 0.7
    risk_high_percentile: float = 0.95
    risk_keep_low: float = 0.65
    risk_keep_mid: float = 0.2
    risk_keep_high: float = 0.95
    risk_rescale_low: bool = True
    risk_rescale_mid: bool = False
    risk_rescale_high: bool = True
    risk_warmup_epoch: int = 0
    risk_start_epoch: int = 10

#### A CONFIG THAT WORKS SOMEWHAT ###
# @dataclass
# class Cifar10IBRisk3Config(Cifar10IBConfig):
#     training_method: str = "ib_risk3"
#     use_ib: bool = True
#     risk_low_percentile: float = 0.75
#     risk_high_percentile: float = 0.95
#     risk_keep_low: float = 0.6
#     risk_keep_mid: float = 0.25
#     risk_keep_high: float = 0.95
#     risk_warmup_epoch: int = 0
#     risk_start_epoch: int = 25

###########################################

@dataclass
class Cifar10IBPRWConfig(Cifar10IBConfig):
    training_method: str = "ib_prw"
    use_ib: bool = True
    ratio: float = 0.5
    prw_epsilon: float = 0.05
    prw_normalize_weights: bool = True


@dataclass
class Cifar10IB3SegPRWNormConfig(Cifar10IB3SegConfig):
    training_method: str = "ib_3seg_prw_norm"
    use_ib: bool = True
    segment_low_fraction: float = 0.0
    segment_high_fraction: float = 0.60
    segment_keep_low: float = 0.0
    segment_keep_mid: float = 0.0
    segment_keep_high: float = 1.0
    prw_epsilon: float = 0.05
    prw_normalize_weights: bool = True


@dataclass
class Cifar10IB3SegPRWNormLOOConfig(Cifar10IB3SegPRWNormConfig):
    is_loo_with_partial_subsets: bool = False
    is_loo: bool = True
    num_subsets: int = 400
    log_sample_evolution: bool = False
    log_noise_splits: bool = False
    log_wandb: bool = False
    save_best_checkpoint: bool = False
    save_last_checkpoint: bool = False
    save_epoch_checkpoints: bool = True


@dataclass
class Cifar10StandardLOOConfig(Cifar10StandardConfig):
    is_loo_with_partial_subsets: bool = False #True
    is_loo: bool = True
    num_subsets: int = 400
    log_sample_evolution: bool = False
    log_noise_splits: bool = False
    log_wandb: bool = False
    save_best_checkpoint: bool = False
    save_last_checkpoint: bool = False
    save_epoch_checkpoints: bool = True


@dataclass
class Cifar10IBLOOConfig(Cifar10IBConfig):
    is_loo_with_partial_subsets: bool = False #True
    is_loo: bool = True
    num_subsets: int = 400
    log_sample_evolution: bool = False
    log_noise_splits: bool = False
    log_wandb: bool = False
    save_best_checkpoint: bool = False
    save_last_checkpoint: bool = False
    save_epoch_checkpoints: bool = True


@dataclass
class Cifar10EHLOOConfig(Cifar10EHConfig):
    is_loo_with_partial_subsets: bool = True
    log_sample_evolution: bool = False
    log_noise_splits: bool = False
    log_wandb: bool = False
    save_best_checkpoint: bool = False
    save_last_checkpoint: bool = False
    save_epoch_checkpoints: bool = True


@dataclass
class Cifar10IBFocalLOOConfig(Cifar10IBFocalConfig):
    is_loo_with_partial_subsets: bool = True
    log_sample_evolution: bool = False
    log_noise_splits: bool = False
    log_wandb: bool = False
    save_best_checkpoint: bool = False
    save_last_checkpoint: bool = False
    save_epoch_checkpoints: bool = True


@dataclass
class Cifar10IBFocalLateLOOConfig(Cifar10IBFocalLateConfig):
    is_loo_with_partial_subsets: bool = True
    log_sample_evolution: bool = False
    log_noise_splits: bool = False
    log_wandb: bool = False
    save_best_checkpoint: bool = False
    save_last_checkpoint: bool = False
    save_epoch_checkpoints: bool = True


@dataclass
class Cifar10IBStdLOOConfig(Cifar10IBStdConfig):
    is_loo_with_partial_subsets: bool = True
    log_sample_evolution: bool = False
    log_noise_splits: bool = False
    log_wandb: bool = False
    save_best_checkpoint: bool = False
    save_last_checkpoint: bool = False
    save_epoch_checkpoints: bool = True


@dataclass
class Cifar10IB3SegLOOConfig(Cifar10IB3SegConfig):
    is_loo_with_partial_subsets: bool = False
    is_loo: bool = True
    num_subsets: int = 400
    log_sample_evolution: bool = False
    log_noise_splits: bool = False
    log_wandb: bool = False
    save_best_checkpoint: bool = False
    save_last_checkpoint: bool = False
    save_epoch_checkpoints: bool = True


@dataclass
class Cifar10IB3SegStaticLOOConfig(Cifar10IB3SegStaticConfig):
    is_loo_with_partial_subsets: bool = True
    log_sample_evolution: bool = False
    log_noise_splits: bool = False
    log_wandb: bool = False
    save_best_checkpoint: bool = False
    save_last_checkpoint: bool = False
    save_epoch_checkpoints: bool = True


@dataclass
class Cifar10IBRisk3LOOConfig(Cifar10IBRisk3Config):
    is_loo_with_partial_subsets: bool = True
    log_sample_evolution: bool = False
    log_noise_splits: bool = False
    log_wandb: bool = False
    save_best_checkpoint: bool = False
    save_last_checkpoint: bool = False
    save_epoch_checkpoints: bool = True


@dataclass
class Cifar10IBPRWLOOConfig(Cifar10IBPRWConfig):
    is_loo_with_partial_subsets: bool = False #True
    is_loo: bool = True
    num_subsets: int = 400
    log_sample_evolution: bool = False
    log_noise_splits: bool = False
    log_wandb: bool = False
    save_best_checkpoint: bool = False
    save_last_checkpoint: bool = False
    save_epoch_checkpoints: bool = True
