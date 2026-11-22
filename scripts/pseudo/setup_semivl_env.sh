#!/usr/bin/env bash
# Creates SemiVL's own legacy environment and everything SemiVL needs at model construction, so that
# cora/pseudo/semivl_standalone.py can run (tools/make_pseudo_labels.py --method semivl).
#
#   bash scripts/pseudo/setup_semivl_env.sh [ENV_DIR]
#
#   ENV_DIR   conda environment, default $CORA_ROOT/envs/semivl
# Environment: SEMIVL_DIR (default $CORA_ROOT/third_party/semivl), SEMIVL_COMMIT (default 13ee65d6f27e197e0b55d987b346fc880fc811ca).
#
# What it does (every step is skipped when its result already exists, so the script can be re-run):
#   1. clone of google-research/semivl at the pinned commit
#   2. conda env with python 3.8 at ENV_DIR (SemiVL: torch 1.10.1+cu113, mmcv-full 1.4.4, mmsegmentation 0.24.0, timm 0.6.5)
#   3. pretrained/resnet101_v1c-e67eebb6.pth (open-mmlab ResNet-101-v1c, checked against the hash prefix in its name)
#   4. pretrained/clip2mmseg_ViT16_clip_backbone.pth: OpenAI CLIP ViT-B/16 (checked against the SHA-256 in its URL),
#      converted by SemiVL's own third_party/maskclip/convert_clip_weights.py
#   5. an import check of SemiVL's model builder
# It needs network access (conda, pip, GitHub, download.openmmlab.com, openaipublic.azureedge.net), about 12 GB of
# free disk space and no sudo. It does not touch any other environment.
set -euo pipefail

CORA_ROOT=${CORA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ENV_DIR=${1:-$CORA_ROOT/envs/semivl}
SEMIVL_DIR=${SEMIVL_DIR:-$CORA_ROOT/third_party/semivl}
SEMIVL_COMMIT=${SEMIVL_COMMIT:-13ee65d6f27e197e0b55d987b346fc880fc811ca}
CONDA=${CONDA_EXE:-$(command -v conda || true)}
MIN_FREE_GB=12

export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$(dirname "$ENV_DIR")" "$(dirname "$SEMIVL_DIR")"

CLIP_URL=https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
CLIP_SHA256=5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
R101_URL=https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth
R101_SHA256_PREFIX=e67eebb6

PY=$ENV_DIR/bin/python

# ---- 1. SemiVL source at the pinned commit (its requirements.txt feeds the environment) ------------
if [ ! -d "$SEMIVL_DIR/.git" ]; then
  echo "[1/5] cloning SemiVL"
  git clone -q https://github.com/google-research/semivl "$SEMIVL_DIR"
fi
if [ "$(git -C "$SEMIVL_DIR" rev-parse HEAD)" != "$SEMIVL_COMMIT" ]; then
  git -C "$SEMIVL_DIR" checkout -q "$SEMIVL_COMMIT"
fi
echo "[1/5] SemiVL at $(git -C "$SEMIVL_DIR" rev-parse --short HEAD) in $SEMIVL_DIR"
mkdir -p "$SEMIVL_DIR/pretrained"

# ---- 2. environment ------------------------------------------------------------------------------------------
if [ -x "$PY" ] && "$PY" -c "import torch, mmcv, mmseg, timm, clip, cv2, yaml, einops, PIL; assert torch.__version__.startswith('1.10.1'), torch.__version__" 2>/dev/null; then
  echo "[2/5] environment $ENV_DIR is complete"
else
  free_gb=$(df --output=avail -BG "$(dirname "$ENV_DIR")" | tail -n 1 | tr -dc '0-9')
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || { echo "ERROR: only ${free_gb} GB free next to $ENV_DIR, need $MIN_FREE_GB GB" >&2; exit 1; }
  [ -n "$CONDA" ] && [ -x "$CONDA" ] || { echo "ERROR: conda not found. Install Miniconda or Anaconda and make sure 'conda' is on PATH (or set CONDA_EXE)." >&2; exit 1; }
  if [ ! -x "$PY" ]; then
    [ ! -e "$ENV_DIR" ] || { echo "ERROR: $ENV_DIR exists but is not a conda environment with python" >&2; exit 1; }
    echo "[2/5] conda environment with python 3.8 at $ENV_DIR"
    "$CONDA" create -y -q -p "$ENV_DIR" -c conda-forge --override-channels python=3.8 pip
  fi
  export PATH="$ENV_DIR/bin:$PATH"
  "$PY" -m pip install -q "setuptools==59.5.0" "wheel"
  "$PY" -m pip install -q "torch==1.10.1+cu113" "torchvision==0.11.2+cu113" -f https://download.pytorch.org/whl/torch_stable.html
  "$PY" -m pip install -q "mmcv-full==1.4.4" -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
  # SemiVL's pinned requirements without what is not needed for inference (notebook, plotting, pandas, openmim,
  # tensorboard) and without the torch / mmcv lines installed above; the GUI build of OpenCV becomes the headless one.
  REQ=$(mktemp "${TMPDIR:-/tmp}/semivl_requirements.XXXXXX")
  grep -v -E '^(notebook|seaborn|pandas|matplotlib|openmim|tensorboard|TensorBoardX|torch|torchvision|mmcv-full|-f )' "$SEMIVL_DIR/requirements.txt" \
    | sed 's/^opencv-python==/opencv-python-headless==/' > "$REQ"
  "$PY" -m pip install -q -r "$REQ" "mmsegmentation==0.24.0"
  rm -f "$REQ"
fi

# ---- 3. ResNet-101-v1c ----------------------------------------------------------------------------------------
R101=$SEMIVL_DIR/pretrained/resnet101_v1c-e67eebb6.pth
if [ -s "$R101" ]; then
  echo "[3/5] $R101 present"
else
  echo "[3/5] downloading ResNet-101-v1c"
  curl -fSL --retry 5 --retry-delay 5 -o "$R101.part" "$R101_URL"
  got=$(sha256sum "$R101.part" | cut -c1-8)
  [ "$got" = "$R101_SHA256_PREFIX" ] || { rm -f "$R101.part"; echo "ERROR: ResNet-101-v1c hash prefix $got != $R101_SHA256_PREFIX" >&2; exit 1; }
  mv -f "$R101.part" "$R101"
fi

# ---- 4. CLIP ViT-B/16 backbone converted for mmseg ------------------------------------------------------------
CLIP_OUT=$SEMIVL_DIR/pretrained/clip2mmseg_ViT16_clip_backbone.pth
if [ -s "$CLIP_OUT" ]; then
  echo "[4/5] $CLIP_OUT present"
else
  echo "[4/5] downloading and converting CLIP ViT-B/16"
  # The conversion script calls clip.load('ViT-B/16'), which caches under ~/.cache/clip: give it a temporary HOME.
  CLIP_HOME=$(mktemp -d "${TMPDIR:-/tmp}/semivl_clip_home.XXXXXX")
  mkdir -p "$CLIP_HOME/.cache/clip"
  CLIP_FILE=$CLIP_HOME/.cache/clip/ViT-B-16.pt
  if [ ! -s "$CLIP_FILE" ] || [ "$(sha256sum "$CLIP_FILE" | cut -d' ' -f1)" != "$CLIP_SHA256" ]; then
    curl -fSL --retry 5 --retry-delay 5 -o "$CLIP_FILE.part" "$CLIP_URL"
    [ "$(sha256sum "$CLIP_FILE.part" | cut -d' ' -f1)" = "$CLIP_SHA256" ] || { rm -f "$CLIP_FILE.part"; echo "ERROR: CLIP ViT-B/16 SHA-256 mismatch" >&2; exit 1; }
    mv -f "$CLIP_FILE.part" "$CLIP_FILE"
  fi
  (cd "$SEMIVL_DIR" && HOME=$CLIP_HOME "$PY" third_party/maskclip/convert_clip_weights.py --model ViT16 --backbone)
  [ -s "$CLIP_OUT" ] || { echo "ERROR: conversion did not produce $CLIP_OUT" >&2; exit 1; }
  rm -rf -- "$CLIP_HOME"
fi

# ---- 5. import check ------------------------------------------------------------------------------------------
echo "[5/5] import check"
(cd "$SEMIVL_DIR" && PYTHONPATH=$SEMIVL_DIR "$PY" -c "import torch, mmcv, mmseg; from model.builder import build_model; print('torch', torch.__version__, '| mmcv', mmcv.__version__, '| mmseg', mmseg.__version__, '| build_model importable')")
echo "SemiVL environment ready: $PY"
echo "Use it with: python tools/make_pseudo_labels.py --method semivl --partition <partition> --semivl-python $PY ..."
