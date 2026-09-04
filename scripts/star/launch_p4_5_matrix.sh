#!/usr/bin/env bash
# Launch the locked P4.5 4x1 then 2x1 single-GPU training schedule.

set -uo pipefail

SOURCE_DIR=${1:-$PWD}
RUN_ROOT=${2:-/root/autodl-tmp/SAFE/p4_5}
PYTHON_BIN=${3:-/root/autodl-tmp/SAFE/envs/star-p0-p1/bin/python}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment is unavailable: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    echo "Formal training requires a Git worktree: $SOURCE_DIR" >&2
    exit 2
fi
if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
    echo "Formal training requires a clean committed source tree" >&2
    git -C "$SOURCE_DIR" status --short >&2
    exit 2
fi

SOURCE_COMMIT=$(git -C "$SOURCE_DIR" rev-parse HEAD)
mkdir -p "$RUN_ROOT/runs"
{
    echo "source_commit=$SOURCE_COMMIT"
    echo "source_dir=$SOURCE_DIR"
    echo "schedule=batch1:C0-2027,C1-2027,C0-2028,C1-2028;batch2:C0-2029,C1-2029"
} >"$RUN_ROOT/launch_manifest.txt"

run_complete() {
    local receipt=$1
    [[ -f "$receipt" ]] || return 1
    "$PYTHON_BIN" - "$receipt" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding='utf-8'))
raise SystemExit(0 if payload.get('status') == 'passed' and payload.get('steps') == 100_000 else 1)
PY
}

run_one() {
    local arm=$1
    local seed=$2
    local config
    if [[ "$arm" == C0 ]]; then
        config=configs/star/p4_5_c0_unified_cost_geometry.yaml
    else
        config=configs/star/p4_5_c1_directional_loss.yaml
    fi
    local container="$RUN_ROOT/runs/${arm}-${seed}"
    local receipt="$container/receipt.json"
    mkdir -p "$container/training"
    if run_complete "$receipt"; then
        echo "$arm-$seed already completed; skipping"
        return 0
    fi
    echo "starting $arm-$seed at $(date -u +%FT%TZ)"
    (
        cd "$SOURCE_DIR" || exit 2
        export CUDA_VISIBLE_DEVICES=0
        export PYTHONPATH=$SOURCE_DIR
        export OMP_NUM_THREADS=1
        "$PYTHON_BIN" scripts/star/train_star.py \
            --config "$config" \
            --seed "$seed" \
            --device cuda:0 \
            --vector-envs 1 \
            --log-dir "$container/training" \
            --result "$receipt" \
            --no-tensorboard
    ) >"$container/train.log" 2>&1
    local status=$?
    echo "$status" >"$container/exit_code.txt"
    echo "finished $arm-$seed with exit $status at $(date -u +%FT%TZ)"
    return "$status"
}

wait_batch() {
    local label=$1
    shift
    local failures=0
    local pid
    for pid in "$@"; do
        if ! wait "$pid"; then
            failures=$((failures + 1))
        fi
    done
    echo "$label completed with $failures failed process(es) at $(date -u +%FT%TZ)"
    return "$failures"
}

echo "batch1 starting at $(date -u +%FT%TZ)"
run_one C0 2027 &
p1=$!
run_one C1 2027 &
p2=$!
run_one C0 2028 &
p3=$!
run_one C1 2028 &
p4=$!
batch1_failures=0
wait_batch batch1 "$p1" "$p2" "$p3" "$p4" || batch1_failures=$?

echo "batch2 starting at $(date -u +%FT%TZ)"
run_one C0 2029 &
p5=$!
run_one C1 2029 &
p6=$!
batch2_failures=0
wait_batch batch2 "$p5" "$p6" || batch2_failures=$?

total_failures=$((batch1_failures + batch2_failures))
echo "$total_failures" >"$RUN_ROOT/matrix_failure_count.txt"
if [[ "$total_failures" -eq 0 ]]; then
    touch "$RUN_ROOT/training_matrix_complete"
fi
exit "$total_failures"
