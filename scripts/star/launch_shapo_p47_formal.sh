#!/usr/bin/env bash
# Launch the sealed P4.7 main matrix as five four-environment seed waves.

set -uo pipefail

SOURCE_DIR=${1:-$PWD}
RUN_ROOT=${2:-/root/autodl-tmp/SAFE/shapo_p47_four_env/formal}
SMOKE_ROOT=${3:-/root/autodl-tmp/SAFE/shapo_p47_four_env/smoke}
PYTHON_BIN=${4:-/root/autodl-tmp/SAFE/envs/star-p0-p1/bin/python}
PROTOCOL_SHA256=007c9a89d6b8f63d2c56516f9611d22e60164b82ccff939aa0af060d1dffa337
MATRIX_SHA256=f22d644f798e1e036d0ef64359d40ff10fb745d9d97eb2887ec6bb02adc66b0f
FROZEN_CONFIG_SHA256=17a99888036e5521be295ef69742577576d365fad906ac0acf387921a5474565

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

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment is unavailable: $PYTHON_BIN" >&2
    exit 2
fi
if ! git -C "$SOURCE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Formal training requires a Git worktree: $SOURCE_DIR" >&2
    exit 2
fi
if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
    echo "Formal training requires clean committed source" >&2
    git -C "$SOURCE_DIR" status --short >&2
    exit 2
fi

verify_sha256 \
    "$SOURCE_DIR/reports/star/shapo_p47_four_env_preregistration.json" \
    "$PROTOCOL_SHA256"
verify_sha256 "$SOURCE_DIR/configs/star/shapo_p47_four_env.yaml" "$MATRIX_SHA256"
verify_sha256 \
    "$SOURCE_DIR/reports/star/p4_7_safety_aware_geometry_artifacts/final/summary/p4_7_frozen_pointgoal_config.yaml" \
    "$FROZEN_CONFIG_SHA256"

SOURCE_COMMIT=$(git -C "$SOURCE_DIR" rev-parse HEAD)
"$PYTHON_BIN" - "$SMOKE_ROOT/smoke_gate.json" "$SOURCE_COMMIT" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding='utf-8'))
valid = payload.get('status') == 'passed' and payload.get('source_commit') == sys.argv[2]
raise SystemExit(0 if valid else 2)
PY
if [[ $? -ne 0 ]]; then
    echo "Formal launch requires a passing four-environment smoke gate from $SOURCE_COMMIT" >&2
    exit 2
fi

mkdir -p "$RUN_ROOT/runs"
{
    echo "protocol_version=shapo-p47-four-env-v1"
    echo "protocol_sha256=$PROTOCOL_SHA256"
    echo "matrix_sha256=$MATRIX_SHA256"
    echo "source_commit=$SOURCE_COMMIT"
    echo "training_seeds=2027,2028,2029,2030,2031"
    echo "schedule=five_waves_x_one_process_per_environment"
    echo "total_environment_steps=150000000"
    echo "scientific_early_stop=false"
    echo "statistical_unit=training_seed"
} >"$RUN_ROOT/launch_manifest.txt"

validate_one() {
    local container=$1
    local env_id=$2
    local seed=$3
    "$PYTHON_BIN" "$SOURCE_DIR/scripts/star/validate_shapo_p47_run.py" \
        --container "$container" \
        --matrix "$SOURCE_DIR/configs/star/shapo_p47_four_env.yaml" \
        --env-id "$env_id" \
        --seed "$seed" \
        --source-commit "$SOURCE_COMMIT" \
        >/dev/null 2>&1
}

run_one() {
    local slug=$1
    local env_id=$2
    local seed=$3
    local steps=$4
    local container="$RUN_ROOT/runs/${slug}-${seed}"
    local config="$container/resolved_formal_config.yaml"
    mkdir -p "$container/training"
    if [[ -f "$container/source_commit.txt" ]] && \
        [[ "$(<"$container/source_commit.txt")" != "$SOURCE_COMMIT" ]]; then
        echo "$container belongs to a different source commit; refusing to mix runs" >&2
        return 2
    fi
    echo "$SOURCE_COMMIT" >"$container/source_commit.txt"
    "$PYTHON_BIN" "$SOURCE_DIR/scripts/star/materialize_shapo_p47_config.py" \
        --matrix "$SOURCE_DIR/configs/star/shapo_p47_four_env.yaml" \
        --env-id "$env_id" \
        --output "$config" \
        >"$container/materialize.json"
    if validate_one "$container" "$env_id" "$seed"; then
        echo "$env_id seed $seed already completed and validated; skipping"
        return 0
    fi

    local resume_args=()
    local latest_checkpoint
    latest_checkpoint=$(
        find "$container/training" -type f -name latest_full.pt -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr \
            | head -n 1 \
            | cut -d' ' -f2-
    )
    if [[ -n "$latest_checkpoint" ]]; then
        resume_args=(--resume "$latest_checkpoint")
        echo "resuming $env_id seed $seed from $latest_checkpoint"
    else
        echo "starting $env_id seed $seed from scratch"
    fi
    (
        cd "$SOURCE_DIR" || exit 2
        export CUDA_VISIBLE_DEVICES=0
        export PYTHONPATH=$SOURCE_DIR
        export OMP_NUM_THREADS=1
        "$PYTHON_BIN" scripts/star/train_star.py \
            --config "$config" \
            --seed "$seed" \
            --steps "$steps" \
            --device cuda:0 \
            --vector-envs 1 \
            --batch-size 256 \
            --replay-size 400000 \
            --steps-per-epoch 5000 \
            --log-dir "$container/training" \
            --result "$container/receipt.json" \
            --no-tensorboard \
            "${resume_args[@]}"
    ) >"$container/train.log" 2>&1
    local status=$?
    echo "$status" >"$container/exit_code.txt"
    if [[ "$status" -ne 0 ]]; then
        echo "$env_id seed $seed failed with exit $status" >&2
        return "$status"
    fi
    if ! validate_one "$container" "$env_id" "$seed"; then
        echo "$env_id seed $seed completed but failed artifact validation" >&2
        return 3
    fi
    echo "$env_id seed $seed completed and validated"
}

failures=0
for seed in 2027 2028 2029 2030 2031; do
    declare -A WAVE_JOBS=()
    run_one point-goal SafetyPointGoal1-v0 "$seed" 10000000 &
    WAVE_JOBS[$!]=point-goal
    run_one point-button SafetyPointButton1-v0 "$seed" 10000000 &
    WAVE_JOBS[$!]=point-button
    run_one ant Ant-v4 "$seed" 5000000 &
    WAVE_JOBS[$!]=ant
    run_one walker2d Walker2d-v4 "$seed" 5000000 &
    WAVE_JOBS[$!]=walker2d
    for pid in "${!WAVE_JOBS[@]}"; do
        status=0
        wait "$pid" || status=$?
        if [[ "$status" -ne 0 ]]; then
            failures=$((failures + 1))
            echo "${WAVE_JOBS[$pid]} seed $seed exited with status $status" >&2
        fi
    done
    unset WAVE_JOBS
    if [[ "$failures" -ne 0 ]]; then
        echo "$failures" >"$RUN_ROOT/matrix_failure_count.txt"
        exit "$failures"
    fi
    echo "$seed" >"$RUN_ROOT/latest_completed_seed_wave.txt"
done

echo "0" >"$RUN_ROOT/matrix_failure_count.txt"
touch "$RUN_ROOT/formal_matrix_complete"
