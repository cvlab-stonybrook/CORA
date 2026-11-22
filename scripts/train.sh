#!/usr/bin/env bash
# Trains CORA (the three stages of paper Sec. 3.3, Eq. 9) from one configuration file.
#
#   bash scripts/train.sh <config> [num_gpus] [overrides...] [--dry-run]
#
#   config      YAML file, for example configs/cityscapes/1_16.yaml (one file per partition in configs/cityscapes/)
#   num_gpus    processes on this machine, default 1. More than one starts
#               "python -m torch.distributed.run --standalone --nproc_per_node=N tools/train.py ...".
#               The paper trains on 4 GPUs with 2 images each (batch 8). With another count keep batch 8 by setting
#               train.grad_accum_steps (2 GPUs: 2, 1 GPU: 4); the script prints this hint but does not change anything.
#   overrides   key=value pairs of cora/config.py, for example  train.iters_per_stage=100  data.partition=1_16
#   --dry-run   print the command only
#
# Choose the GPUs with CUDA_VISIBLE_DEVICES, for example  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh ... 4
# Before it starts, the script shows the state of those GPUs (nvidia-smi) and stops when one of them is already in
# use (the machine may be shared). ALLOW_BUSY_GPUS=1 starts anyway.
# A finished stage is skipped on a re-run and an interrupted stage resumes from its last checkpoint, both under
# outputs/<train.run_name>/. When the overrides change data.partition or data.ssl_method but not train.run_name,
# the script appends them to the run name (config path below configs/ + partition + teacher), so that runs of
# different partitions never share a folder. Pass train.run_name=... to choose the name yourself.
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

POS=()
for arg in "$@"; do
  case $arg in
    -h | --help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    *) POS+=("$arg") ;;
  esac
done
if [ "${#POS[@]}" -lt 1 ]; then
  usage >&2
  exit 2
fi

CONFIG_GIVEN=${POS[0]}
NUM_GPUS=1
FIRST_OVERRIDE=1
if [ "${#POS[@]}" -ge 2 ] && [[ ${POS[1]} =~ ^[0-9]+$ ]]; then
  NUM_GPUS=${POS[1]}
  FIRST_OVERRIDE=2
  if [ "$NUM_GPUS" -lt 1 ]; then die "num_gpus must be at least 1"; fi
fi
OVERRIDES=()
for ((i = FIRST_OVERRIDE; i < ${#POS[@]}; i++)); do
  case ${POS[$i]} in
    *=*) OVERRIDES+=("${POS[$i]}") ;;
    *) die "'${POS[$i]}' is neither a number of GPUs (second argument) nor a key=value override" ;;
  esac
done

load_env
cd "$REPO"
CONFIG=$(resolve_config "$CONFIG_GIVEN")
RUN_NAME=$(run_name_override "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"})
if [ -n "$RUN_NAME" ]; then
  echo "run name: ${RUN_NAME#train.run_name=} (derived from the overrides; pass train.run_name=... to change it)"
  OVERRIDES+=("$RUN_NAME")
fi

if [ "$NUM_GPUS" -gt 1 ]; then
  CMD=("${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" tools/train.py)
else
  CMD=("${PYTHON:-python}" tools/train.py)
fi
CMD+=(--config "$CONFIG")
if [ "${#OVERRIDES[@]}" -gt 0 ]; then CMD+=("${OVERRIDES[@]}"); fi

if [ "$NUM_GPUS" -ne 4 ] && [ $((8 % (2 * NUM_GPUS))) -eq 0 ]; then
  echo "note: the paper uses 4 GPUs x 2 images (batch 8); with $NUM_GPUS GPU(s) add train.grad_accum_steps=$((8 / (2 * NUM_GPUS))) to keep batch 8"
fi
check_gpus_free "$NUM_GPUS"
run_cmd "${CMD[@]}"
