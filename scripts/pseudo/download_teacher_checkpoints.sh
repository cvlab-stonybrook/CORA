#!/usr/bin/env bash
# Downloads the public checkpoints of the semi-supervised segmenters that provide CORA's pseudo-labels
# on Cityscapes (paper Sec. 4.1: UniMatch and SemiVL, trained on the same partitions as CORA).
#
#   SemiVL   all four partitions (1/30, 1/16, 1/8, 1/4): release "checkpoints" of google-research/semivl
#   UniMatch ResNet-101 checkpoints for 1/16, 1/8 and 1/4 plus the ImageNet ResNet-101 backbone the model
#            constructor needs: Google Drive links of the UniMatch README.
#            UniMatch publishes NO 1/30 checkpoint; that teacher has to be trained (scripts/pseudo/).
#
#   bash scripts/pseudo/download_teacher_checkpoints.sh [semivl|unimatch|all] [DEST]
#
# DEST defaults to $CORA_ROOT/pretrained/teachers. Finished files are skipped on re-runs.
# The checkpoints are third-party Python pickles: load them only with torch.load(..., weights_only=True).
set -euo pipefail

WHAT=${1:-all}
ROOT=${CORA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DEST=${2:-$ROOT/pretrained/teachers}
PYTHON=${PYTHON:-python}

semivl() {
  local dir=$DEST/semivl split zip name
  mkdir -p "$dir"
  for split in 1_30 1_16 1_8 1_4; do
    name=semivl_cs_$split
    if [ -s "$dir/$name/best.pth" ] && [ -s "$dir/$name/config.yaml" ]; then
      echo "SemiVL $split: already present"
      continue
    fi
    zip=$dir/$name.zip
    echo "SemiVL $split: downloading"
    curl -fSL --retry 5 --retry-delay 5 -C - -o "$zip" \
      "https://github.com/google-research/semivl/releases/download/checkpoints/$name.zip"
    # Refuse archives with absolute or parent-directory paths before extracting anything.
    if unzip -Z1 "$zip" | grep -qE '^/|(^|/)\.\.(/|$)'; then
      echo "ERROR: $zip contains unsafe paths; not extracted" >&2
      exit 1
    fi
    unzip -q -o "$zip" "$name/best.pth" "$name/config.yaml" "$name/debug.log" "$name/preds.tar.gz" -d "$dir"
    rm -f -- "$zip"
    echo "SemiVL $split: $(du -h "$dir/$name/best.pth" | cut -f1) best.pth"
  done
}

unimatch() {
  local dir=$DEST/unimatch
  mkdir -p "$dir"
  "$PYTHON" -c "import gdown" 2>/dev/null || { echo "ERROR: gdown is not installed in $("$PYTHON" -c 'import sys; print(sys.prefix)')" >&2; exit 1; }
  # file name -> Google Drive id (UniMatch README, Cityscapes table, ResNet-101 row; backbone from "Pretrained Backbones")
  local -a files=(
    "cityscapes_r101_1_16.pth:1qmCBLC9aj57kz1_OptvK6YTo4GwxTsiK"
    "cityscapes_r101_1_8.pth:14LrPkWC8QIMO44da5pGflyOrW_Fdxo0U"
    "cityscapes_r101_1_4.pth:1cL-p2_FIwEe9Y4AapSjlLmt4hdAZzX7a"
    "resnet101.pth:1Rx0legsMolCWENpfvE2jUScT3ogalMO8"
  )
  local entry file id
  for entry in "${files[@]}"; do
    file=${entry%%:*}
    id=${entry##*:}
    if [ -s "$dir/$file" ]; then
      echo "UniMatch $file: already present"
      continue
    fi
    echo "UniMatch $file: downloading"
    "$PYTHON" -m gdown "https://drive.google.com/uc?id=$id" -O "$dir/$file.part"
    # A Drive quota or permission page is HTML, not a checkpoint.
    if head -c 512 "$dir/$file.part" | grep -qi '<html'; then
      echo "ERROR: Google Drive returned a web page instead of $file (quota or permission problem)" >&2
      rm -f -- "$dir/$file.part"
      exit 1
    fi
    mv -f -- "$dir/$file.part" "$dir/$file"
    echo "UniMatch $file: $(du -h "$dir/$file" | cut -f1)"
  done
  echo "NOTE: UniMatch has no public Cityscapes 1/30 checkpoint."
}

case $WHAT in
  semivl) semivl ;;
  unimatch) unimatch ;;
  all) semivl; unimatch ;;
  *) echo "usage: $0 [semivl|unimatch|all] [DEST]" >&2; exit 2 ;;
esac
echo "--- files in $DEST ---"
find "$DEST" -type f \( -name '*.pth' -o -name 'config.yaml' \) -exec ls -l {} \; | awk '{print $5, $9}'
