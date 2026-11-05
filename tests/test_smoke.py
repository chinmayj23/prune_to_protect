"""Smoke tests: registry coherence + config correctness (no GPU, no data, no training).

Full accuracy/attack reproduction needs GPUs: see docs/REPRODUCE.md.
"""

from __future__ import annotations

import pytest


def test_package_imports():
    import prunepriv
    assert prunepriv.__version__


def test_registries_populated():
    from prunepriv.configs import available_datasets, available_models, available_methods
    assert {"cifar10", "cifar100", "imagenet"} <= set(available_datasets())
    assert {"resnet20", "resnet18", "resnet50", "wrn28_2", "deit_small", "deit_tiny"} <= set(available_models())
    assert {"standard", "infobatch", "ib_prw", "wlib_loss", "ib_noanneal"} <= set(available_methods())


def test_dataset_specs():
    from prunepriv.configs import get_dataset_spec
    assert get_dataset_spec("cifar10").num_classes == 10
    assert get_dataset_spec("cifar100").num_classes == 100
    assert get_dataset_spec("imagenet").loader_backend == "ffcv"
    c10 = get_dataset_spec("cifar10")
    assert len(c10.mean) == 3 and len(c10.std) == 3


def test_method_aliases():
    from prunepriv.configs import canonical_method
    assert canonical_method("ib_prw") == "wlib" or canonical_method("wlib") == "ib_prw"
    assert canonical_method("ib") == "infobatch"


@pytest.mark.parametrize("dataset,model,expect_classes,expect_bs,expect_lr", [
    ("cifar10", "resnet20", 10, 512, 0.2),
    ("cifar100", "wrn28_2", 100, 128, 0.1),
])
def test_build_config_cifar(dataset, model, expect_classes, expect_bs, expect_lr):
    """build_config composes dataset+model+method into the right ExptConfig (CPU-only, no training)."""
    from prunepriv.configs import build_config
    for method, tm, use_ib in [("standard", "standard", False),
                               ("infobatch", "ib", True),
                               ("wlib", "ib_prw", True)]:
        cfg = build_config(dataset, model, method)
        assert cfg.num_classes == expect_classes
        assert cfg.batch_size == expect_bs
        assert abs(cfg.learning_rate - expect_lr) < 1e-9
        assert cfg.training_method == tm
        assert bool(getattr(cfg, "use_ib", False)) == use_ib
        assert cfg.dataset_name == dataset
    # WLIB signature: normalized PRW weights
    assert getattr(build_config(dataset, model, "wlib"), "prw_normalize_weights", False) is True


def test_build_model_resnet20_cpu():
    """The one model with no heavy deps builds on CPU."""
    from prunepriv.configs import build_model
    m = build_model("resnet20", num_classes=10)
    assert sum(p.numel() for p in m.parameters()) > 0


def test_unknown_names_are_loud():
    from prunepriv.configs import build_config, build_model
    with pytest.raises(KeyError):
        build_config("nope")
    with pytest.raises(KeyError):
        build_model("no_such_model", num_classes=10)


def test_clis_import():
    from prunepriv.cli import train, attack, memorize
    for mod in (train, attack, memorize):
        assert callable(mod.main) and mod.build_parser() is not None
