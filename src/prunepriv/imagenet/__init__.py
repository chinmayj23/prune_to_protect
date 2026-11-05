"""ImageNet pipeline (FFCV + DDP + fastargs).

Self-contained and separate from the CIFAR engine. Requires the FFCV conda env
(env/setup_imagenet.sh); imports of ffcv/timm are deferred so importing the rest of `prunepriv` never
needs them. Covers full-data and pruned training plus offline RMIA. See README.md.
"""
