#!/usr/bin/env bash
# Launch the locked P4.7 three-seed method run and three sealed C0 continuations.

set -uo pipefail

SOURCE_DIR=${1:-$PWD}
RUN_ROOT=${2:-/root/autodl-tmp/SAFE/p4_7_safety_aware_geometry}
PYTHON_BIN=${3:-/root/autodl-tmp/SAFE/envs/star-p0-p1/bin/python}
C0_ROOT=${4:-/root/autodl-tmp/SAFE/p4_5_532d4b1/runs}
PROTOCOL_SHA256=3310d6f229d579a877cfaa98d3df2a44b25eac9998184623d1723b82d37a03ed
C0_CONFIG_SHA256=d64f56b765724ab6d2e5536971173b6c41e98373fdd04137627a22a6c8791faa
P47_CONFIG_SHA256=570c15805807ae2577add1975afb857e34ef5ea889df0e1124c882e393665471

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

verify_sha256() {
    local path=$1
    local expected=$2
    local actual
    actual=$(sha256sum "$path" | cut -d' ' -f1)
    if [[ "$actual" != "$expected" ]]; then
        echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
        exit 2
    fi
}

verify_sha256 \
    "$SOURCE_DIR/reports/star/p4_7_safety_aware_geometry_preregistration.json" \
    "$PROTOCOL_SHA256"
verify_sha256 "$SOURCE_DIR/configs/star/p4_7_c0_continuation.yaml" "$C0_CONFIG_SHA256"
verify_sha256 "$SOURCE_DIR/configs/star/p4_7_safety_aware_geometry.yaml" "$P47_CONFIG_SHA256"

declare -A C0_CHECKPOINTS=(
    [2027]="$C0_ROOT/C0-2027/training/STAR-{SafetyPointGoal1-v0}/seed-2027-2026-08-08-20-55-46/checkpoints/final_full.pt"
    [2028]="$C0_ROOT/C0-2028/training/STAR-{SafetyPointGoal1-v0}/seed-2028-2026-08-08-20-55-46/checkpoints/final_full.pt"
    [2029]="$C0_ROOT/C0-2029/training/STAR-{SafetyPointGoal1-v0}/seed-2029-2026-08-09-01-42-12/checkpoints/final_full.pt"
)
declare -A C0_SHA256=(
    [2027]=0c8d0261d7b957f73bc72f1ce2542ddaa84a8f5061c43461c72f7a8e390a040b
    [2028]=bdc781c7ad152569314d128ebd65c4e838c6d9cbe6035f2b4d1cfd8c13c053da
    [2029]=3ae57f4eb265cf5939af152eebf82ea4cda1bede3aa99d7f12764b20a47dbf8e
)
for seed in 2027 2028 2029; do
    verify_sha256 "${C0_CHECKPOINTS[$seed]}" "${C0_SHA256[$seed]}"
done

SOURCE_COMMIT=$(git -C "$SOURCE_DIR" rev-parse HEAD)
mkdir -p "$RUN_ROOT/runs"
{
    echo "protocol_version=p4.7-safety-aware-geometry-v1"
    echo "protocol_sha256=$PROTOCOL_SHA256"
    echo "source_commit=$SOURCE_COMMIT"
    echo "source_dir=$SOURCE_DIR"
    echo "schedule=dynamic-4x1:P47-2027,P47-2028,P47-2029,C0-2027,C0-2028,C0-2029"
    echo "scientific_early_stop=false"
} >"$RUN_ROOT/launch_manifest.txt"

run_complete() {
    local receipt=$1
    local config_sha256=$2
    [[ -f "$receipt" ]] || return 1
    "$PYTHON_BIN" - "$receipt" "$SOURCE_COMMIT" "$config_sha256" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding='utf-8'))
valid = (
    payload.get('status') == 'passed'
    and payload.get('steps') == 200_000
    and payload.get('source_commit') == sys.argv[2]
    and payload.get('source_dirty') is False
    and payload.get('config_sha256') == sys.argv[3]
)
raise SystemExit(0 if valid else 1)
PY
}

run_one() {
    local arm=$1
    local seed=$2
    local config
    local config_sha256
    local resume_args=()
    case "$arm" in
        C0)
            config=configs/star/p4_7_c0_continuation.yaml
            config_sha256=$C0_CONFIG_SHA256
            resume_args=(--resume "${C0_CHECKPOINTS[$seed]}")
            ;;
        P47)
            config=configs/star/p4_7_safety_aware_geometry.yaml
            config_sha256=$P47_CONFIG_SHA256
            ;;
        *)
            echo "Unknown P4.7 arm: $arm" >&2
            return 2
            ;;
    esac
    local container="$RUN_ROOT/runs/${arm}-${seed}"
    local receipt="$container/receipt.json"
    mkdir -p "$container/training"
    if run_complete "$receipt" "$config_sha256"; then
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
            --steps 200000 \
            --device cuda:0 \
            --vector-envs 1 \
            --log-dir "$container/training" \
            --result "$receipt" \
            --no-tensorboard \
            "${resume_args[@]}"
    ) >"$container/train.log" 2>&1
    local status=$?
    echo "$status" >"$container/exit_code.txt"
    echo "finished $arm-$seed with exit $status at $(date -u +%FT%TZ)"
    return "$status"
}

declare -A ACTIVE_JOBS=()
failures=0

wait_for_one() {
    local completed_pid
    local status=0
    wait -n -p completed_pid "${!ACTIVE_JOBS[@]}" || status=$?
    echo "${ACTIVE_JOBS[$completed_pid]} exited with status $status"
    unset 'ACTIVE_JOBS[$completed_pid]'
    if [[ "$status" -ne 0 ]]; then
        failures=$((failures + 1))
    fi
}

launch_one() {
    local arm=$1
    local seed=$2
    run_one "$arm" "$seed" &
    local pid=$!
    ACTIVE_JOBS[$pid]="$arm-$seed"
}

for job in P47:2027 P47:2028 P47:2029 C0:2027 C0:2028 C0:2029; do
    while [[ ${#ACTIVE_JOBS[@]} -ge 4 ]]; do
        wait_for_one
    done
    launch_one "${job%%:*}" "${job##*:}"
done
while [[ ${#ACTIVE_JOBS[@]} -gt 0 ]]; do
    wait_for_one
done

echo "$failures" >"$RUN_ROOT/matrix_failure_count.txt"
if [[ "$failures" -eq 0 ]]; then
    touch "$RUN_ROOT/training_matrix_complete"
fi
exit "$failures"
