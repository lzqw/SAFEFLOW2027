#!/usr/bin/env bash
# Run the locked common-curve and formal-final P4.7 checkpoint evaluations.

set -uo pipefail

SOURCE_DIR=${1:-$PWD}
RUN_ROOT=${2:-/root/autodl-tmp/SAFE/p4_7_safety_aware_geometry}
PYTHON_BIN=${3:-/root/autodl-tmp/SAFE/envs/star-p0-p1/bin/python}
P45_ROOT=${4:-/root/autodl-tmp/SAFE/p4_5_532d4b1/runs}
EVAL_SEEDS=22027,22028,22029,22030,22031

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment is unavailable: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -f "$RUN_ROOT/training_matrix_complete" ]]; then
    echo "P4.7 training matrix is incomplete: $RUN_ROOT" >&2
    exit 2
fi
if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
    echo "Formal evaluation requires a clean committed source tree" >&2
    git -C "$SOURCE_DIR" status --short >&2
    exit 2
fi

find_checkpoint() {
    local root=$1
    local name=$2
    local matches=()
    mapfile -t matches < <(find "$root" -path "*/checkpoints/$name" -type f -print | sort)
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "Expected one $name below $root, found ${#matches[@]}" >&2
        return 2
    fi
    printf '%s\n' "${matches[0]}"
}

append_checkpoint() {
    local -n destination=$1
    local root=$2
    local name=$3
    local checkpoint
    checkpoint=$(find_checkpoint "$root" "$name") || return 2
    destination+=("$checkpoint")
}

declare -a C0_INTERMEDIATE=()
declare -a C1_INTERMEDIATE=()
declare -a P47_INTERMEDIATE=()
declare -a C0_FORMAL=()
declare -a C1_FORMAL=()
declare -a P47_FORMAL=()

for seed in 2027 2028 2029; do
    c0_old="$P45_ROOT/C0-$seed"
    c1_old="$P45_ROOT/C1-$seed"
    c0_new="$RUN_ROOT/runs/C0-$seed"
    p47_new="$RUN_ROOT/runs/P47-$seed"

    append_checkpoint C0_INTERMEDIATE "$c0_old" milestone_50000.pt || exit 2
    append_checkpoint C0_INTERMEDIATE "$c0_old" milestone_100000.pt || exit 2
    append_checkpoint C0_INTERMEDIATE "$c0_new" milestone_150000.pt || exit 2
    append_checkpoint C0_INTERMEDIATE "$c0_new" milestone_200000.pt || exit 2
    append_checkpoint C1_INTERMEDIATE "$c1_old" milestone_50000.pt || exit 2
    append_checkpoint C1_INTERMEDIATE "$c1_old" milestone_100000.pt || exit 2
    append_checkpoint P47_INTERMEDIATE "$p47_new" milestone_50000.pt || exit 2
    append_checkpoint P47_INTERMEDIATE "$p47_new" milestone_100000.pt || exit 2
    append_checkpoint P47_INTERMEDIATE "$p47_new" milestone_150000.pt || exit 2
    append_checkpoint P47_INTERMEDIATE "$p47_new" milestone_200000.pt || exit 2

    append_checkpoint C0_FORMAL "$c0_new" milestone_200000.pt || exit 2
    append_checkpoint C1_FORMAL "$c1_old" milestone_100000.pt || exit 2
    append_checkpoint P47_FORMAL "$p47_new" milestone_200000.pt || exit 2
done

SOURCE_COMMIT=$(git -C "$SOURCE_DIR" rev-parse HEAD)
mkdir -p "$RUN_ROOT/evaluation/intermediate" "$RUN_ROOT/evaluation/formal" "$RUN_ROOT/evaluation/work"
{
    echo "protocol_version=p4.7-safety-aware-geometry-v1"
    echo "source_commit=$SOURCE_COMMIT"
    echo "base_seeds=$EVAL_SEEDS"
    echo "intermediate_episodes_per_base_seed=4"
    echo "formal_episodes_per_base_seed=20"
    echo "primary_only=true"
} >"$RUN_ROOT/evaluation/evaluation_manifest.txt"

run_evaluator() {
    local tier=$1
    local arm=$2
    local episodes_per_seed=$3
    shift 3
    local output="$RUN_ROOT/evaluation/$tier/$arm"
    local work="$RUN_ROOT/evaluation/work/$tier-$arm"
    mkdir -p "$output" "$work"
    echo "starting $tier $arm evaluation at $(date -u +%FT%TZ)"
    (
        cd "$SOURCE_DIR" || exit 2
        export CUDA_VISIBLE_DEVICES=0
        export PYTHONPATH=$SOURCE_DIR
        export OMP_NUM_THREADS=1
        "$PYTHON_BIN" scripts/star/evaluate_pointgoal_checkpoints.py \
            --checkpoints "$@" \
            --output-dir "$output" \
            --work-dir "$work" \
            --device cuda:0 \
            --eval-seeds "$EVAL_SEEDS" \
            --episodes-per-seed "$episodes_per_seed" \
            --primary-only
    ) >"$RUN_ROOT/evaluation/${tier}-${arm}.log" 2>&1
    local status=$?
    echo "$status" >"$RUN_ROOT/evaluation/${tier}-${arm}.exit_code.txt"
    echo "finished $tier $arm evaluation with exit $status at $(date -u +%FT%TZ)"
    return "$status"
}

wait_group() {
    local label=$1
    shift
    local failures=0
    local pid
    for pid in "$@"; do
        if ! wait "$pid"; then
            failures=$((failures + 1))
        fi
    done
    echo "$label completed with $failures failure(s)"
    return "$failures"
}

run_evaluator intermediate C0 4 "${C0_INTERMEDIATE[@]}" &
p1=$!
run_evaluator intermediate C1 4 "${C1_INTERMEDIATE[@]}" &
p2=$!
run_evaluator intermediate P47 4 "${P47_INTERMEDIATE[@]}" &
p3=$!
intermediate_failures=0
wait_group intermediate "$p1" "$p2" "$p3" || intermediate_failures=$?

if [[ "$intermediate_failures" -ne 0 ]]; then
    echo "$intermediate_failures" >"$RUN_ROOT/evaluation/evaluation_failure_count.txt"
    exit "$intermediate_failures"
fi

run_evaluator formal C0 20 "${C0_FORMAL[@]}" &
p4=$!
run_evaluator formal C1 20 "${C1_FORMAL[@]}" &
p5=$!
run_evaluator formal P47 20 "${P47_FORMAL[@]}" &
p6=$!
formal_failures=0
wait_group formal "$p4" "$p5" "$p6" || formal_failures=$?

echo "$formal_failures" >"$RUN_ROOT/evaluation/evaluation_failure_count.txt"
if [[ "$formal_failures" -eq 0 ]]; then
    touch "$RUN_ROOT/evaluation/evaluation_complete"
fi
exit "$formal_failures"
