"""Model registry / factory.

``build_model(name, num_classes, **kw)`` is the ONE place model choice happens. The training engine
also dispatches on the ``model_name`` token (r20/r18/r50/...) internally; this factory mirrors those
builders so tests and non-engine callers share the same definitions.

Add a model: register a builder ``(num_classes, **kw) -> nn.Module`` in ``_BUILDERS``.
"""

from __future__ import annotations

from typing import Callable

_BUILDERS: dict[str, Callable[..., object]] = {}


def register_model(name: str, builder: Callable[..., object]) -> None:
    _BUILDERS[name] = builder


def build_model(name: str, num_classes: int, **kw):
    key = str(name).strip().lower()
    if key not in _BUILDERS:
        raise KeyError(f"Model {name!r} not registered (available: {sorted(_BUILDERS)}).")
    return _BUILDERS[key](num_classes=num_classes, **kw)


def available_models() -> list[str]:
    return sorted(_BUILDERS)


def _register_builtin_models() -> None:
    # resnet20: CIFAR ResNet with mask-capable convolutions.
    def _resnet20(num_classes: int, **kw):
        from .resnet20 import ResNet20
        return ResNet20(num_classes=num_classes)

    # resnet18 / resnet50 via torchvision WITH the CIFAR stem surgery (3x3 conv1, no maxpool) that the
    # training engine applies — so build_model matches how the engine trains these on 32x32 inputs.
    def _resnet18(num_classes: int, **kw):
        import torch.nn as nn
        import torchvision
        m = torchvision.models.resnet18(num_classes=num_classes, weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m

    def _resnet50(num_classes: int, **kw):
        import torch.nn as nn
        import torchvision
        m = torchvision.models.resnet50(num_classes=num_classes, weights=None)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m

    def _wrn28_2(num_classes: int, **kw):
        from .wideresnet import WideResNet
        return WideResNet(depth=28, num_classes=num_classes, widen_factor=2)

    # DeiT (timm VisionTransformer), defined in imagenet/utils/deit.py. Lazy import so the CIFAR path
    # never needs timm. pretrained=True loads ImageNet weights and swaps the head to num_classes
    # (fine-tune); pretrained=False builds fresh at num_classes (from-scratch).
    def _deit(builder_name):
        def _build(num_classes: int, **kw):
            import torch.nn as nn
            from prunepriv.imagenet.utils import deit as _deit_mod
            builder = getattr(_deit_mod, builder_name)
            if kw.get("pretrained", False):
                net = builder(pretrained=True, num_classes=1000)   # load ImageNet-1k weights
                net.head = nn.Linear(net.head.in_features, num_classes)  # fresh classification head
                return net
            return builder(pretrained=False, num_classes=num_classes)
        return _build

    _deit_tiny = _deit("deit_tiny_patch16_224")
    _deit_small = _deit("deit_small_patch16_224")

    # Light 224px ResNets via timm (ImageNet stems, not the CIFAR stem). Used for CelebA and Waterbirds.
    def _timm(name):
        def _b(num_classes: int, **kw):
            import timm
            return timm.create_model(name, pretrained=kw.get("pretrained", False), num_classes=num_classes)
        return _b

    _resnet10t = _timm("resnet10t")   # 4.9M
    _resnet14t = _timm("resnet14t")   # 8.0M
    _resnet18_224 = _timm("resnet18")  # 11.2M, standard 224 stem (distinct from the CIFAR resnet18)
    _resnet34_224 = _timm("resnet34")  # 21.8M
    _resnet50_224 = _timm("resnet50")  # 25.6M, standard 224 stem
    _wrn50_2_224 = _timm("wide_resnet50_2")  # 68.9M

    for alias, builder in [
        ("resnet20", _resnet20), ("r20", _resnet20),
        ("resnet18", _resnet18), ("r18", _resnet18),
        ("resnet50", _resnet50), ("r50", _resnet50),
        ("wrn28_2", _wrn28_2),
        ("deit_small", _deit_small), ("deit_tiny", _deit_tiny),
        ("resnet10t", _resnet10t), ("resnet14t", _resnet14t), ("resnet18_224", _resnet18_224),
        ("resnet34_224", _resnet34_224),
        ("resnet50_224", _resnet50_224), ("wrn50_2_224", _wrn50_2_224),
    ]:
        register_model(alias, builder)


_register_builtin_models()

__all__ = ["build_model", "register_model", "available_models"]
