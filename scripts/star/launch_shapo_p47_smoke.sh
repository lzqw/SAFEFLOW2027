#!/usr/bin/env bash
# Run the four preregistered, non-scientific SHAPO P4.7 engineering checks sequentially.

set -uo pipefail

SOURCE_DIR=${1:-$PWD}
RUN_ROOT=${2:-/root/autodl-tmp/SAFE/shapo_p47_four_env/smoke}
PYTHON_BIN=${3:-/root/autodl-tmp/SAFE/envs/star-p0-p1/bin/python}
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
    echo "Engineering smoke requires a Git worktree: $SOURCE_DIR" >&2
    exit 2
fi
if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
    echo "Engineering smoke requires clean committed source" >&2
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
mkdir -p "$RUN_ROOT"
{
    echo "protocol_version=shapo-p47-four-env-v1"
    echo "protocol_sha256=$PROTOCOL_SHA256"
    echo "matrix_sha256=$MATRIX_SHA256"
    echo "source_commit=$SOURCE_COMMIT"
    echo "scientific_evidence=false"
    echo "excluded_from_formal_aggregation=true"
} >"$RUN_ROOT/launch_manifest.txt"

failures=0
for job in \
    point-goal:SafetyPointGoal1-v0 \
    point-button:SafetyPointButton1-v0 \
    ant:Ant-v4 \
    walker2d:Walker2d-v4; do
    slug=${job%%:*}
    env_id=${job#*:}
    container="$RUN_ROOT/$slug"
    config="$container/resolved_smoke_config.yaml"
    mkdir -p "$container/training"
    "$PYTHON_BIN" "$SOURCE_DIR/scripts/star/materialize_shapo_p47_config.py" \
        --matrix "$SOURCE_DIR/configs/star/shapo_p47_four_env.yaml" \
        --env-id "$env_id" \
        --smoke \
        --output "$config" \
        >"$container/materialize.json"
    echo "starting engineering smoke $env_id at $(date -u +%FT%TZ)"
    (
        cd "$SOURCE_DIR" || exit 2
        export CUDA_VISIBLE_DEVICES=0
        export PYTHONPATH=$SOURCE_DIR
        export OMP_NUM_THREADS=1
        "$PYTHON_BIN" scripts/star/train_star.py \
            --config "$config" \
            --seed 9027 \
            --device cuda:0 \
            --vector-envs 1 \
            --batch-size 8 \
            --replay-size 128 \
            --steps-per-epoch 16 \
            --log-dir "$container/training" \
            --result "$container/receipt.json" \
            --no-tensorboard
    ) >"$container/train.log" 2>&1
    status=$?
    echo "$status" >"$container/exit_code.txt"
    if [[ "$status" -ne 0 ]]; then
        failures=$((failures + 1))
        echo "engineering smoke $env_id failed with exit $status" >&2
    else
        echo "engineering smoke $env_id completed at $(date -u +%FT%TZ)"
    fi
done

echo "$failures" >"$RUN_ROOT/smoke_failure_count.txt"
if [[ "$failures" -ne 0 ]]; then
    exit "$failures"
fi
"$PYTHON_BIN" "$SOURCE_DIR/scripts/star/validate_shapo_p47_smoke.py" \
    --run-root "$RUN_ROOT" \
    >"$RUN_ROOT/smoke_validation.log"
