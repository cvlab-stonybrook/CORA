#!/usr/bin/env bash
# Evaluates a trained CORA model with cIoU and gIoU (paper Sec. 4.1) through tools/evaluate.py.
#
#   bash scripts/evaluate.sh <config> [tools/evaluate.py options] [overrides...] [--num-gpus N] [--dry-run]
#
#   config      the YAML file the model was trained with
#   options     everything of tools/evaluate.py, for example
#                 --reasonseg test        ReasonSeg test set, overall and long-query subset (Table 3)
#                 --sets conditional      only these validation instruction sets
#                 --mode generate         let the model write the answer (default: teacher forced)
#                 --checkpoint FILE       a stage's final.pt or last.pt (default: last finished stage of the run)
#                 --max-samples N
#   overrides   key=value pairs, repeated exactly as they were given to scripts/train.sh (the run folder must match);
#               like train.sh, the script derives the run name from data.partition / data.ssl_method overrides
#   --num-gpus  evaluate with torch.distributed.run on N processes (default 1)
#   --dry-run   print the command only
#
# Examples:
#   bash scripts/evaluate.sh configs/cityscapes/1_16.yaml
#   bash scripts/evaluate.sh configs/cityscapes/1_16.yaml --reasonseg test
#   bash scripts/evaluate.sh configs/cityscapes/1_16.yaml data.ssl_method=unimatch
# Results are printed and written to outputs/<run name>/eval/<name>_<mode>.json.
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

if [ "$#" -lt 1 ]; then
  usage >&2
  exit 2
fi
CONFIG_GIVEN=""
NUM_GPUS=1
OPTIONS=()
OVERRIDES=()
while [ "$#" -gt 0 ]; do
  case $1 in
    -h | --help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --num-gpus) shift; [ "$#" -gt 0 ] || die "--num-gpus needs a number"; NUM_GPUS=$1 ;;
    -*) OPTIONS+=("$1") ;;
    *=*) OVERRIDES+=("$1") ;;
    *)
      if [ -z "$CONFIG_GIVEN" ]; then
        CONFIG_GIVEN=$1
      else
        OPTIONS+=("$1")  # the value of an option such as --reasonseg test
      fi
      ;;
  esac
  shift
done
[ -n "$CONFIG_GIVEN" ] || die "no configuration file given"
[[ $NUM_GPUS =~ ^[0-9]+$ ]] && [ "$NUM_GPUS" -ge 1 ] || die "--num-gpus must be a positive integer"

load_env
cd "$REPO"
CONFIG=$(resolve_config "$CONFIG_GIVEN")
RUN_NAME=$(run_name_override "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"})
if [ -n "$RUN_NAME" ]; then
  echo "run name: ${RUN_NAME#train.run_name=} (derived from the overrides)"
  OVERRIDES+=("$RUN_NAME")
fi

if [ "$NUM_GPUS" -gt 1 ]; then
  CMD=("${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" tools/evaluate.py)
else
  CMD=("${PYTHON:-python}" tools/evaluate.py)
fi
CMD+=(--config "$CONFIG")
if [ "${#OPTIONS[@]}" -gt 0 ]; then CMD+=("${OPTIONS[@]}"); fi
# The overrides are positional; "--" keeps option lists such as "--sets a b" from taking them.
if [ "${#OVERRIDES[@]}" -gt 0 ]; then CMD+=(-- "${OVERRIDES[@]}"); fi
run_cmd "${CMD[@]}"
