#!/usr/bin/env bash
# Set up the ImageNet/FFCV environment. FFCV needs compiled deps
# (numba/opencv/pkg-config/libturbojpeg) that don't co-exist cleanly with the CIFAR venv, so this is
# a SEPARATE conda env. Requires conda/mamba on PATH.
# Usage:  bash env/setup_imagenet.sh [env_name]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-prunepriv_imagenet}"

echo ">> creating conda env $ENV_NAME (python 3.10)"
conda create -y -n "$ENV_NAME" python=3.10 cupy pkg-config libjpeg-turbo opencv numba \
    -c conda-forge -c pytorch

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ">> installing torch 2.5.1 + torchvision 0.20.1 (CUDA 12.1)"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

echo ">> installing ffcv + imagenet extras"
pip install ffcv
pip install -r "$REPO_ROOT/requirements-imagenet.txt"
pip install -e "$REPO_ROOT"

echo ">> verifying"
python - <<'PY'
import torch, ffcv, timm
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("ffcv OK; timm", timm.__version__)
PY
echo ">> done. next: build betons -> python -m prunepriv.imagenet.write_indexed_betons --help"
