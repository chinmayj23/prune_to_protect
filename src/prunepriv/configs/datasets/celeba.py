from __future__ import annotations

from .spec import DatasetSpec

# CelebA (HF tpremoli/CelebA-attrs). Group-robustness setup: predict Blond_Hair with Male as the
# spurious attribute. 4 groups = 2*Blond_Hair + Male (minority = blonde males, ~1387 train, 0.85%).
# Large (162k train) -> CPU-resident blob (per-batch GPU move). Stored 128px, 112px train/eval crop.
SPEC = DatasetSpec(
    name="celeba",
    num_classes=2,               # Blond_Hair (0/1)
    input_size=112,              # crop size (blob stored at 128)
    channels=3,
    train_size=162770,
    test_size=19867,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    loader_backend="image_pt",
    default_model="resnet18_224",
    data_subpath="celeba",
    num_groups=4,                # 0 non-blonde/female, 1 non-blonde/male, 2 blonde/female, 3 blonde/male
    loader_blob_device="",       # GPU-resident (128px blob ~7.5GB fits) -> ~15s/epoch vs 23s on CPU
    models=("resnet10t", "resnet18_224", "resnet34_224"),
)
