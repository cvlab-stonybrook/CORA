#!/usr/bin/env bash
# Prepares everything that training on one Cityscapes partition needs, except the LLM-written instruction sets
# (attribute, conditional): those come with the provided instruction files, which you put in place afterwards.
#
#   bash scripts/prepare_cityscapes.sh <partition> <teacher> [options]
#
#   partition   1_30 | 1_16 | 1_8 | 1_4   (labeled fraction, paper Sec. 4.2)
#   teacher     unimatch | semivl         (source of the pseudo-labels, paper Sec. 4.1)
#
# Options:
#   --gpus LIST          GPU indices for the pseudo-label step, for example 0 or 0,1,2,3 (one process per GPU)
#   --workers N          processes for the CPU steps (default 8)
#   --date YYYY-MM-DD    date written into the pseudo-label manifest (default: today)
#   --set KEY=VALUE      config override passed to every tool, for example paths.data_dir=data_other (repeatable)
#   --skip-pseudo        do not make pseudo-labels (steps 1 to 3 only)
#   --skip-teacher-eval  do not measure the teacher's mIoU on the 500 validation images before writing pseudo-labels
#   --dry-run            print the commands and run none of them
#
# Steps (every tool resumes: finished files are kept):
#   1. check the Cityscapes folder and the partition lists            tools/prepare_cityscapes.py
#   2. region maps of the labeled images and of val                   tools/build_objects.py
#   3. rule-based sets for train and val (semantic, templated conditional) and the shipped query database D
#                                                                     tools/generate_rule_sets.py
#   4. teacher check (mIoU on val) and pseudo-labels of the unlabeled images   tools/make_pseudo_labels.py
#
# Teachers: UniMatch checkpoints for 1_16, 1_8, 1_4 come from scripts/pseudo/download_teacher_checkpoints.sh unimatch;
# UniMatch has no public 1_30 checkpoint: train one with UniMatch's own training code and point UNIMATCH_130_CKPT at
# the result (default: $CORA_ROOT/pretrained/teachers/unimatch/cityscapes_r101_1_30.pth).
# SemiVL runs in its own environment (scripts/pseudo/setup_semivl_env.sh) with the checkpoints from
# scripts/pseudo/download_teacher_checkpoints.sh semivl.
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

GPUS=""
WORKERS=8
DATE=$(date +%F)
SKIP_PSEUDO=0
SKIP_TEACHER_EVAL=0
SETS=()
POS=()
while [ "$#" -gt 0 ]; do
  case $1 in
    -h | --help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
    --skip-pseudo) SKIP_PSEUDO=1 ;;
    --skip-teacher-eval) SKIP_TEACHER_EVAL=1 ;;
    --gpus) shift; [ "$#" -gt 0 ] || die "--gpus needs a value"; GPUS=$1 ;;
    --workers) shift; [ "$#" -gt 0 ] || die "--workers needs a value"; WORKERS=$1 ;;
    --date) shift; [ "$#" -gt 0 ] || die "--date needs a value"; DATE=$1 ;;
    --set) shift; [ "$#" -gt 0 ] || die "--set needs KEY=VALUE"; SETS+=("$1") ;;
    -*) die "unknown option $1 (see --help)" ;;
    *) POS+=("$1") ;;
  esac
  shift
done
if [ "${#POS[@]}" -ne 2 ]; then
  usage >&2
  exit 2
fi
PARTITION=${POS[0]}
TEACHER=${POS[1]}
check_partition "$PARTITION"
case $TEACHER in unimatch | semivl) ;; *) die "unknown teacher '$TEACHER'; use unimatch or semivl" ;; esac
[[ $WORKERS =~ ^[0-9]+$ ]] && [ "$WORKERS" -ge 1 ] || die "--workers must be a positive integer"
[[ $DATE =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "--date must look like 2026-11-19"

load_env
cd "$REPO"
PY=${PYTHON:-python}
ROOT=${CORA_ROOT:-$REPO}

SET_ARGS=()
DATA_DIR=$ROOT/data
for item in ${SETS[@]+"${SETS[@]}"}; do
  SET_ARGS+=(--set "$item")
  case $item in
    paths.data_dir=*)
      DATA_DIR=${item#*=}
      case $DATA_DIR in /*) ;; *) DATA_DIR=$ROOT/$DATA_DIR ;; esac
      ;;
  esac
done
QUERY_DB=$DATA_DIR/cityscapes/query_db.json

# ---- the teacher must exist before any work is done --------------------------------------------------------------
CHECKPOINT=""
if [ "$SKIP_PSEUDO" = 0 ]; then
  if [ "$TEACHER" = unimatch ]; then
    if [ "$PARTITION" = 1_30 ]; then
      CHECKPOINT=${UNIMATCH_130_CKPT:-$ROOT/pretrained/teachers/unimatch/cityscapes_r101_1_30.pth}
      HINT="UniMatch publishes no 1/30 checkpoint: train one with UniMatch's own training code and set UNIMATCH_130_CKPT"
    else
      CHECKPOINT=$ROOT/pretrained/teachers/unimatch/cityscapes_r101_$PARTITION.pth
      HINT="download it with  bash scripts/pseudo/download_teacher_checkpoints.sh unimatch"
    fi
  else
    CHECKPOINT=$ROOT/pretrained/teachers/semivl/semivl_cs_$PARTITION/best.pth
    HINT="download it with  bash scripts/pseudo/download_teacher_checkpoints.sh semivl"
    SEMIVL_PY=$ROOT/envs/semivl/bin/python
    if [ ! -x "$SEMIVL_PY" ]; then
      if [ "$DRY_RUN" = 1 ]; then
        echo "WARNING: SemiVL's environment $SEMIVL_PY is missing; create it with  bash scripts/pseudo/setup_semivl_env.sh" >&2
      else
        die "SemiVL's environment $SEMIVL_PY is missing; create it with  bash scripts/pseudo/setup_semivl_env.sh"
      fi
    fi
  fi
  if [ ! -f "$CHECKPOINT" ]; then
    if [ "$DRY_RUN" = 1 ]; then
      echo "WARNING: teacher checkpoint $CHECKPOINT not found; $HINT" >&2
    else
      die "teacher checkpoint $CHECKPOINT not found; $HINT (or pass --skip-pseudo)"
    fi
  fi
fi

echo "== Cityscapes $PARTITION with $TEACHER pseudo-labels (data under $DATA_DIR) =="

echo "[1/4] Cityscapes folder and partition lists"
run_cmd "$PY" tools/prepare_cityscapes.py --partition "$PARTITION" --which labeled --workers "$WORKERS"

echo "[2/4] region maps (connected components, supplement Sec. 4) of the labeled images and of val"
run_cmd "$PY" tools/build_objects.py ${SET_ARGS[@]+"${SET_ARGS[@]}"} --partition "$PARTITION" --which labeled --workers "$WORKERS"
run_cmd "$PY" tools/build_objects.py ${SET_ARGS[@]+"${SET_ARGS[@]}"} --split val --workers "$WORKERS"

echo "[3/4] rule-based instruction sets and the query database D"
for split_args in "--split train --partition $PARTITION" "--split val"; do
  # shellcheck disable=SC2086
  run_cmd "$PY" tools/generate_rule_sets.py ${SET_ARGS[@]+"${SET_ARGS[@]}"} semantic $split_args
  # shellcheck disable=SC2086
  run_cmd "$PY" tools/generate_rule_sets.py ${SET_ARGS[@]+"${SET_ARGS[@]}"} templated-conditional $split_args
done
if [ -f "$QUERY_DB" ] && [ "$DRY_RUN" = 0 ]; then
  echo "query database $QUERY_DB exists; kept"
else
  run_cmd "$PY" tools/generate_rule_sets.py ${SET_ARGS[@]+"${SET_ARGS[@]}"} query-db
fi

echo "[4/4] pseudo-labels of the unlabeled images"
if [ "$SKIP_PSEUDO" = 1 ]; then
  echo "skipped (--skip-pseudo)"
else
  GPU_ARGS=()
  if [ -n "$GPUS" ]; then GPU_ARGS=(--gpus "$GPUS"); fi
  REPORT=$ROOT/outputs/pseudo_reports/${TEACHER}_${PARTITION}_val.json
  BASE=("$PY" tools/make_pseudo_labels.py ${SET_ARGS[@]+"${SET_ARGS[@]}"} --method "$TEACHER" --partition "$PARTITION" --date "$DATE" --checkpoint "$CHECKPOINT" ${GPU_ARGS[@]+"${GPU_ARGS[@]}"})
  if [ "$SKIP_TEACHER_EVAL" = 0 ]; then
    echo "teacher check on the 500 validation images: compare the mIoU with the published value of this teacher and partition"
    run_cmd "${BASE[@]}" --evaluate --report "$REPORT"
    run_cmd "${BASE[@]}" --eval-report "$REPORT"
  else
    run_cmd "${BASE[@]}"
  fi
fi

TRAIN_OVERRIDE=""
if [ "$TEACHER" != semivl ]; then TRAIN_OVERRIDE=" data.ssl_method=$TEACHER"; fi  # the configs name SemiVL as the teacher
cat <<NEXT

The LLM-written instruction sets (attribute, conditional) are not generated by this script. Put the provided
instruction files in place:
  $DATA_DIR/cityscapes/instructions/train/{attribute,conditional}.jsonl
  $DATA_DIR/cityscapes/instructions/val/{attribute,conditional}.jsonl
Then train:
  bash scripts/train.sh configs/cityscapes/${PARTITION}.yaml 4${TRAIN_OVERRIDE}
NEXT
