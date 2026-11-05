#!/usr/bin/env bash
# Set up the CIFAR + analysis environment. Python 3.10.
# Usage:  bash env/setup_cifar.sh [venv_dir]
# Creates a venv, installs CUDA-12.1 torch + the pinned requirements, and editable-installs prunepriv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-$REPO_ROOT/env/venv}"

echo ">> creating venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel

echo ">> installing torch 2.5.1 + torchvision 0.20.1 (CUDA 12.1)"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

echo ">> installing pinned requirements"
pip install -r "$REPO_ROOT/requirements.txt"

echo ">> editable-installing prunepriv"
pip install -e "$REPO_ROOT"

echo ">> verifying"
python - <<'PY'
import torch, prunepriv
from prunepriv.configs import available_datasets
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("prunepriv", prunepriv.__version__, "datasets", available_datasets())
PY
echo ">> done. activate with:  source $VENV_DIR/bin/activate"
