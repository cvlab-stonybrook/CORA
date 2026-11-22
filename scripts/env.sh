# Source this file to work with CORA:   source scripts/env.sh
# It activates the dedicated CORA environment (default $CORA_ROOT/envs/cora, made by scripts/create_env.sh).
# CORA_ROOT is the folder that holds data/, outputs/, pretrained/, datasets/, envs/ and third_party/;
# it defaults to the repository root.
_CORA_REPO=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)
export CORA_ROOT=${CORA_ROOT:-$_CORA_REPO}
export CORA_ENV=${CORA_ENV:-$CORA_ROOT/envs/cora}
export TOKENIZERS_PARALLELISM=false

if [ ! -x "$CORA_ENV/bin/python" ]; then
  echo "CORA environment not found at $CORA_ENV. Create it with: bash $_CORA_REPO/scripts/create_env.sh" >&2
  unset _CORA_REPO
  return 1 2>/dev/null || exit 1
fi
if command -v conda > /dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CORA_ENV"
else
  export PATH="$CORA_ENV/bin:$PATH"
fi
export PYTHONPATH="$_CORA_REPO${PYTHONPATH:+:$PYTHONPATH}"
unset _CORA_REPO
