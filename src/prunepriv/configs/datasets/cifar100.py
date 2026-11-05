from __future__ import annotations

from .spec import DatasetSpec

# Constants match cifar100/fast_cifar100_training_EH.py (CIFAR100_MEAN/STD).
SPEC = DatasetSpec(
    name="cifar100",
    num_classes=100,
    input_size=32,
    channels=3,
    train_size=50_000,
    test_size=10_000,
    mean=(0.5071, 0.4867, 0.4408),
    std=(0.2675, 0.2565, 0.2761),
    loader_backend="fast_pt",
    default_model="wrn28_2",
    data_subpath="cifar100",
    models=("wrn28_2", "resnet18", "resnet20"),
)
