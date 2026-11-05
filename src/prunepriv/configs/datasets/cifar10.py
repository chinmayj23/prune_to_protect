from __future__ import annotations

from .spec import DatasetSpec

SPEC = DatasetSpec(
    name="cifar10",
    num_classes=10,
    input_size=32,
    channels=3,
    train_size=50_000,
    test_size=10_000,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2435, 0.2616),
    loader_backend="fast_pt",
    default_model="resnet20",
    data_subpath="cifar10",
    models=("resnet20", "resnet18"),
)
