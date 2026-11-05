"""DatasetSpec — intrinsic, method-independent facts about a dataset.

This is the single source of truth for "what dataset". Training hyper-parameters (batch size, lr,
epochs, pruning delta, ...) are NOT here — they live in the base ``ExptConfig`` and per-dataset
defaults resolved by ``build_config``. Keeping the two separate is what lets one training engine and
one attack runner serve every dataset.

Normalization constants are the [0,1]-scale mean/std used by the fast CIFAR loader
(``T.Normalize(mean, std)`` after dividing pixels by 255). ImageNet's FFCV pipeline uses the same
values on a 0-255 scale (mean*255, std*255) — handled by the FFCV loader, not here.

Add a dataset by dropping a ``DatasetSpec`` into a module under ``configs/datasets/`` and registering
it in ``configs/datasets/__init__.py``. Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    name: str                       # canonical key, e.g. "cifar10"
    num_classes: int
    input_size: int                 # square image side (px)
    channels: int = 3
    train_size: int = 0             # number of training examples (canonical sample-id space)
    test_size: int = 0
    mean: tuple[float, ...] = ()     # per-channel, [0,1] scale
    std: tuple[float, ...] = ()
    loader_backend: str = "fast_pt"  # "fast_pt" (GPU-resident CIFAR loader) | "ffcv" (ImageNet)
    default_model: str = ""          # model used when --model is omitted
    data_subpath: str = ""           # location under PRUNEPRIV_DATA (blobs / betons)
    num_groups: int = 0              # subgroups for fairness analysis (e.g. Waterbirds: 4)
    loader_blob_device: str = ""     # image_pt blob location: "" -> GPU-resident; "cpu" -> per-batch move (large sets)
    # models known-good for this dataset (advisory; build_model still accepts others)
    models: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.mean and len(self.mean) != self.channels:
            raise ValueError(f"{self.name}: mean has {len(self.mean)} values, expected {self.channels}")
        if self.std and len(self.std) != self.channels:
            raise ValueError(f"{self.name}: std has {len(self.std)} values, expected {self.channels}")
