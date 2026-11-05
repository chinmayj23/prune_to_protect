from __future__ import annotations

from .spec import DatasetSpec

# Waterbirds (HF grodino/waterbirds). Binary landbird/waterbird with a spurious background (place).
# 4 groups = label x place (minority = landbird/water, waterbird/land). 224px, ImageNet norm, image_pt.
SPEC = DatasetSpec(
    name="waterbirds",
    num_classes=2,
    input_size=224,
    channels=3,
    train_size=4795,
    test_size=5794,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    loader_backend="image_pt",
    default_model="resnet18_224",
    data_subpath="waterbirds",
    models=("resnet10t", "resnet14t", "resnet18_224"),
    num_groups=4,
)
