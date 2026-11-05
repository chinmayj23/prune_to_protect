from __future__ import annotations

from .spec import DatasetSpec

# ImageNet-1k via FFCV betons (see imagenet/write_indexed_betons.py). Standard ImageNet
# normalization; the FFCV pipeline applies it on the 0-255 scale (mean*255, std*255).
SPEC = DatasetSpec(
    name="imagenet",
    num_classes=1000,
    input_size=224,
    channels=3,
    train_size=1_281_167,
    test_size=50_000,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    loader_backend="ffcv",
    default_model="resnet50",
    data_subpath="imagenet",
    models=("resnet50", "deit_small", "deit_tiny"),
)
