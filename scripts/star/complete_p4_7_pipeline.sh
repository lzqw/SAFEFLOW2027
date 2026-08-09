#!/usr/bin/env bash
# Wait for the formal P4.7 matrix, then run locked evaluation and finalization exactly once.

set -euo pipefail

SOURCE_DIR=${1:-$PWD}
RUN_ROOT=${2:-/root/autodl-tmp/SAFE/p4_7_safety_aware_geometry}
PYTHON_BIN=${3:-/root/autodl-tmp/SAFE/envs/star-p0-p1/bin/python}
P45_ROOT=${4:-/root/autodl-tmp/SAFE/p4_5_532d4b1/runs}
OUTPUT_ROOT=${5:-$RUN_ROOT/public_artifacts}

if [[ -e "$RUN_ROOT/pipeline_complete" ]]; then
    echo "P4.7 pipeline is already complete: $RUN_ROOT"
    exit 0
fi

while [[ ! -e "$RUN_ROOT/training_matrix_complete" ]]; do
    if [[ -e "$RUN_ROOT/matrix_failure_count.txt" ]] \
        && [[ "$(cat "$RUN_ROOT/matrix_failure_count.txt")" -ne 0 ]]; then
        echo "P4.7 training stopped with failed run segments" >&2
        exit 1
    fi
    if [[ -e "$RUN_ROOT/launcher.pid" ]] \
        && ! kill -0 "$(cat "$RUN_ROOT/launcher.pid")" 2>/dev/null \
        && [[ ! -e "$RUN_ROOT/matrix_failure_count.txt" ]]; then
        echo "P4.7 launcher disappeared without a completion record" >&2
        exit 1
    fi
    sleep 60
done

bash "$SOURCE_DIR/scripts/star/evaluate_p4_7_checkpoints.sh" \
    "$SOURCE_DIR" "$RUN_ROOT" "$PYTHON_BIN" "$P45_ROOT"

bash "$SOURCE_DIR/scripts/star/finalize_p4_7_safety_aware_geometry.sh" \
    "$SOURCE_DIR" "$RUN_ROOT" "$PYTHON_BIN" "$P45_ROOT" "$OUTPUT_ROOT"

touch "$RUN_ROOT/pipeline_complete"
echo "P4.7 training, evaluation, and finalization complete"
