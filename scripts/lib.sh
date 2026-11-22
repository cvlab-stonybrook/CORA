# Helpers shared by the launcher scripts of this folder (prepare_cityscapes.sh, train.sh, evaluate.sh). Source it; do not run it.
#
# Conventions: every launcher accepts --dry-run, which prints the commands and runs none of them, and sources
# scripts/env.sh (the dedicated CORA environment) before it starts any work.

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DRY_RUN=0

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Prints a command shell-quoted, one line, prefixed with "+". Plain words stay unquoted.
show_cmd() {
  local word
  printf '+'
  for word in "$@"; do
    if [[ $word =~ ^[A-Za-z0-9_./:=@%+,-]+$ ]]; then
      printf ' %s' "$word"
    else
      printf ' %q' "$word"
    fi
  done
  printf '\n'
}

# Prints a command and runs it unless --dry-run was given.
run_cmd() {
  show_cmd "$@"
  if [ "$DRY_RUN" = 1 ]; then
    return 0
  fi
  "$@"
}

# Activates the CORA environment (scripts/env.sh). With --dry-run a missing environment is only a warning, so that the
# commands can be inspected on a machine without it.
load_env() {
  set +u
  # shellcheck disable=SC1091
  if ! source "$REPO/scripts/env.sh"; then
    set -u
    if [ "$DRY_RUN" = 1 ]; then
      echo "WARNING: the CORA environment is missing; --dry-run only prints the commands" >&2
      export CORA_ROOT=${CORA_ROOT:-$REPO}
      return 0
    fi
    exit 1
  fi
  set -u
}

# Shows the GPUs a job would use and stops when one of them is already in use (the machine may be shared).
#   check_gpus_free <number of processes>
# The GPUs are the first <number> ids of CUDA_VISIBLE_DEVICES, or 0 .. number-1 when it is unset. A GPU counts as in
# use when a compute process runs on it or more than GPU_BUSY_MIB MiB (default 1000) are allocated on it.
# ALLOW_BUSY_GPUS=1 turns the stop into a warning. With --dry-run the state is only shown. Without nvidia-smi the
# check is skipped with a note.
check_gpus_free() {
  local count=$1 ids id used procs busy=""
  if ! command -v nvidia-smi > /dev/null 2>&1; then
    echo "note: nvidia-smi not found, so the GPUs were not checked"
    return 0
  fi
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    ids=$(echo "${CUDA_VISIBLE_DEVICES//,/ }" | tr ' ' '\n' | head -n "$count" | tr '\n' ' ')
  else
    ids=$(seq 0 $((count - 1)) | tr '\n' ' ')
  fi
  for id in $ids; do
    used=$(nvidia-smi --id="$id" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | tr -d ' ')
    procs=$(nvidia-smi --id="$id" --query-compute-apps=pid --format=csv,noheader 2> /dev/null | grep -c . || true)
    if ! [[ $used =~ ^[0-9]+$ ]]; then
      die "GPU $id could not be queried with nvidia-smi (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
    fi
    echo "GPU $id: $used MiB in use, $procs compute process(es)"
    if [ "$used" -gt "${GPU_BUSY_MIB:-1000}" ] || [ "$procs" -gt 0 ]; then busy="$busy $id"; fi
  done
  if [ -n "$busy" ]; then
    if [ "$DRY_RUN" = 1 ]; then
      echo "note: GPU(s)$busy are in use; a real run would stop here"
    elif [ "${ALLOW_BUSY_GPUS:-0}" = 1 ]; then
      echo "WARNING: GPU(s)$busy are in use; continuing because ALLOW_BUSY_GPUS=1" >&2
    else
      die "GPU(s)$busy are in use. Choose free GPUs with CUDA_VISIBLE_DEVICES (see nvidia-smi), or set ALLOW_BUSY_GPUS=1 to start anyway."
    fi
  fi
}

check_partition() {
  case ${1:-} in
    1_30 | 1_16 | 1_8 | 1_4) ;;
    *) die "unknown partition '${1:-}'; use one of 1_30, 1_16, 1_8, 1_4" ;;
  esac
}

# Prints the path of a configuration file, relative to the repository when it lies inside it.
resolve_config() {
  local given=$1 path
  if [ -f "$given" ]; then
    path=$(cd "$(dirname "$given")" && pwd)/$(basename "$given")
  elif [ -f "$REPO/$given" ]; then
    path=$REPO/$given
  else
    die "configuration file '$given' not found"
  fi
  echo "${path#"$REPO"/}"
}

# Prints "train.run_name=<name>" when the given overrides change the partition or the teacher but not the run name.
# Without it a run of another partition would find the finished stages of an earlier run under the same name and
# silently skip training. The name is the config path below configs/ with "/" replaced by "_", plus the overridden
# partition and teacher. Prints nothing when train.run_name is overridden or neither is changed.
run_name_override() {
  local config=$1
  shift
  local item part="" teacher=""
  for item in "$@"; do
    case $item in
      train.run_name=*) return 0 ;;
      data.partition=*) part=${item#*=} ;;
      data.ssl_method=*) teacher=${item#*=} ;;
    esac
  done
  if [ -z "$part$teacher" ]; then
    return 0
  fi
  local stem=${config%.yaml}
  stem=${stem#*configs/}
  stem=${stem//\//_}
  stem=${stem#_}
  local name=$stem
  part=${part//[\"\']/}
  teacher=${teacher//[\"\']/}
  if [ -n "$part" ]; then name=${name}_$part; fi
  if [ -n "$teacher" ]; then name=${name}_$teacher; fi
  echo "train.run_name=$name"
}
