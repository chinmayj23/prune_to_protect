# Third-party code

The MIT license in `LICENSE` covers the original work in this repository. The code below comes from or
builds on other projects, which keep their own copyright and licenses.

| Code | Location | Source | License |
|---|---|---|---|
| CIFAR data loader and augmentations | `src/prunepriv/training/engine.py` | [cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench) | MIT |
| Training pipeline, masked layers | `src/prunepriv/training/`, `src/prunepriv/imagenet/`, `src/prunepriv/utils/mask_layers.py` | [TurboPrune](https://github.com/nelaturuharsha/TurboPrune) | Apache-2.0 |
| DeiT models | `src/prunepriv/imagenet/utils/deit.py` | [DeiT](https://github.com/facebookresearch/deit), Copyright (c) 2015-present, Facebook, Inc. | Apache-2.0 |
| ResNet-20 | `src/prunepriv/configs/models/resnet20.py` | [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10) | BSD-2-Clause |
| WideResNet | `src/prunepriv/configs/models/wideresnet.py` | [WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch) | MIT |
| LiRA scoring | `src/prunepriv/attacks/rmia/` | [carlini/privacy](https://github.com/carlini/privacy/tree/better-mi/research/mi_lira_2021) | Apache-2.0 |
| RMIA attack | `src/prunepriv/attacks/rmia/` | [ml_privacy_meter](https://github.com/privacytrustlab/ml_privacy_meter) | MIT |
