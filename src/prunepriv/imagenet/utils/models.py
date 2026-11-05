"""Model factory: torchvision ResNet50 or timm DeiT-small."""
import torch.nn as nn
import torchvision.models as tv_models


def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    name = model_name.lower()
    if name in {"resnet50", "r50"}:
        return tv_models.resnet50(num_classes=num_classes, weights=None if not pretrained else "DEFAULT")
    if name in {"deit_small", "deit_small_patch16_224"}:
        from .deit import deit_small_patch16_224
        return deit_small_patch16_224(pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unsupported model_name: {model_name!r}")
