#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 CHECKPOINT ENV_ID FINAL_TOTAL_STEPS LOG_DIR [extra train_star.py args...]" >&2
  exit 2
fi

checkpoint=$1
env_id=$2
final_steps=$3
log_dir=$4
shift 4

exec "$(dirname "$0")/train_star.py" \
  --resume "$checkpoint" \
  --checkpoint-replay \
  --env-id "$env_id" \
  --steps "$final_steps" \
  --log-dir "$log_dir" \
  "$@"
