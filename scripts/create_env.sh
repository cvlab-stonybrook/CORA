#!/usr/bin/env bash
# Creates the dedicated CORA environment: a conda environment with its own Python 3.11 and the pinned
# packages of requirements.txt. It does not depend on any other environment.
#
#   bash scripts/create_env.sh [ENV_DIR]
#
# ENV_DIR defaults to $CORA_ROOT/envs/cora (CORA_ROOT defaults to the repository root). Package caches and
# temporary files use the locations of your own conda / pip / TMPDIR settings.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CORA_ROOT=${CORA_ROOT:-$REPO}
ENV_DIR=${1:-$CORA_ROOT/envs/cora}
CONDA=${CONDA_EXE:-$(command -v conda || true)}

export PIP_DISABLE_PIP_VERSION_CHECK=1

if [ -e "$ENV_DIR" ]; then
  echo "ERROR: $ENV_DIR already exists. Remove it first or pass another path." >&2
  exit 1
fi
[ -n "$CONDA" ] && [ -x "$CONDA" ] || { echo "ERROR: conda not found. Install Miniconda or Anaconda and make sure 'conda' is on PATH (or set CONDA_EXE)." >&2; exit 1; }
mkdir -p "$(dirname "$ENV_DIR")"

echo "[1/4] conda environment with Python 3.11 at $ENV_DIR"
"$CONDA" create -y -q -p "$ENV_DIR" -c conda-forge --override-channels python=3.11 pip

echo "[2/4] PyTorch 2.11.0 (CUDA 12.8 build)"
"$ENV_DIR/bin/python" -m pip install -q --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0

echo "[3/4] packages from requirements.txt"
"$ENV_DIR/bin/python" -m pip install -q -r "$REPO/requirements.txt"

echo "[4/4] verification"
"$ENV_DIR/bin/python" - <<'PY'
import importlib.metadata as m
import sys

import torch

print("python", sys.version.split()[0], "at", sys.prefix)
for name in ("torch", "torchvision", "transformers", "peft", "segment-anything", "numpy", "opencv-python-headless"):
    print(f"  {name}=={m.version(name)}")
import cv2, peft, pyarrow, segment_anything, transformers, yaml  # noqa: E401,F401  (import check)
print("  CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
PY
"$ENV_DIR/bin/python" -m pip check || echo "WARNING: pip check reported conflicts (see above)"
echo "Environment ready. Activate it with:  source $REPO/scripts/env.sh"
