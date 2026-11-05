"""
FFCV loaders for the indexed ImageNet betons.

Every loader returns (image, label, index) — the third field is the canonical
per-sample row id from the beton. Downstream code must treat this as ground
truth; do NOT reconstruct per-batch indices from step * batch_size.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256


def _index_pipeline(device: torch.device):
    return [IntDecoder(), ToTensor(), Squeeze(), ToDevice(device, non_blocking=True)]


def _label_pipeline(device: torch.device):
    return [IntDecoder(), ToTensor(), Squeeze(), ToDevice(device, non_blocking=True)]


def _train_image_pipeline(device: torch.device):
    return [
        RandomResizedCropRGBImageDecoder((224, 224)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
    ]


def _val_image_pipeline(device: torch.device):
    return [
        CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
    ]


def build_train_loader(
    data_root: str,
    beton_name: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    distributed: bool,
    indices: Sequence[int] | None,
    os_cache: bool = True,
    drop_last: bool = True,
    order: OrderOption = OrderOption.RANDOM,
) -> Loader:
    pipelines = {
        "image": _train_image_pipeline(device),
        "label": _label_pipeline(device),
        "index": _index_pipeline(device),
    }
    return Loader(
        os.path.join(data_root, beton_name),
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        os_cache=os_cache,
        drop_last=drop_last,
        pipelines=pipelines,
        distributed=distributed,
        seed=seed,
        indices=list(indices) if indices is not None else None,
    )


def build_test_loader(
    data_root: str,
    beton_name: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    distributed: bool = True,
) -> Loader:
    pipelines = {
        "image": _val_image_pipeline(device),
        "label": _label_pipeline(device),
        "index": _index_pipeline(device),
    }
    return Loader(
        os.path.join(data_root, beton_name),
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.SEQUENTIAL,
        drop_last=False,
        pipelines=pipelines,
        distributed=distributed,
        seed=seed,
    )


def build_analysis_loader(
    data_root: str,
    beton_name: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    indices: Sequence[int],
) -> Loader:
    pipelines = {
        "image": _val_image_pipeline(device),
        "label": _label_pipeline(device),
        "index": _index_pipeline(device),
    }
    return Loader(
        os.path.join(data_root, beton_name),
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.SEQUENTIAL,
        os_cache=True,
        drop_last=False,
        pipelines=pipelines,
        distributed=False,
        indices=list(indices),
    )
