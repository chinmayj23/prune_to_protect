"""Dataset registry. Add a dataset: create ``<name>.py`` with a ``SPEC`` and register it here."""

from __future__ import annotations

from .spec import DatasetSpec
from . import cifar10, cifar100, imagenet, waterbirds, celeba

_REGISTRY: dict[str, DatasetSpec] = {
    cifar10.SPEC.name: cifar10.SPEC,
    cifar100.SPEC.name: cifar100.SPEC,
    imagenet.SPEC.name: imagenet.SPEC,
    waterbirds.SPEC.name: waterbirds.SPEC,
    celeba.SPEC.name: celeba.SPEC,
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}. Known: {sorted(_REGISTRY)}") from None


def available_datasets() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["DatasetSpec", "get_dataset_spec", "available_datasets"]
