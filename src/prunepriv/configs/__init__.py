"""Config registry entry point.

``build_config(dataset, model, method, **overrides)`` composes:
    DatasetSpec (configs/datasets)  +  model choice  +  method (configs/methods/build)
into the fully-configured ExptConfig consumed by training / attacks / memorization.
"""

from __future__ import annotations

from .. import paths
from .datasets import DatasetSpec, get_dataset_spec, available_datasets
from .models import build_model, register_model, available_models
from .methods import build_method_config, canonical_method, available_methods


def build_config(dataset: str, model: str | None = None, method: str = "standard", **overrides):
    """Build the ExptConfig for a (dataset, model, method) triple.

    ``model`` defaults to the dataset's default model. Data lives under
    ``PRUNEPRIV_DATA/<dataset.data_subpath>`` (auto-downloaded by the loader on first run).
    Extra kwargs (e.g. ``num_epochs``, ``base_seed``) override config fields last.
    """
    spec = get_dataset_spec(dataset)
    model = model or spec.default_model
    data_root = str(paths.ensure(paths.data_dir(spec.data_subpath)))
    return build_method_config(spec, model, method, data_root=data_root, **overrides)


__all__ = [
    "build_config", "DatasetSpec", "get_dataset_spec", "available_datasets",
    "build_model", "register_model", "available_models",
    "build_method_config", "canonical_method", "available_methods",
]
