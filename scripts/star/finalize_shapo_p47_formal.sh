#!/usr/bin/env bash
# Strictly revalidate, aggregate, and atomically seal the completed SHAPO P4.7 matrix.

set -euo pipefail

VALIDATION_SOURCE_DIR=${1:-$PWD}
RUN_ROOT=${2:-/root/autodl-tmp/SAFE/shapo_p47_four_env/formal}
PYTHON_BIN=${3:-/root/autodl-tmp/SAFE/envs/star-p0-p1/bin/python}
OUTPUT_ROOT=${4:-$RUN_ROOT/public_artifacts}
PROTOCOL_SHA256=007c9a89d6b8f63d2c56516f9611d22e60164b82ccff939aa0af060d1dffa337
MATRIX_SHA256=f22d644f798e1e036d0ef64359d40ff10fb745d9d97eb2887ec6bb02adc66b0f
FROZEN_CONFIG_SHA256=17a99888036e5521be295ef69742577576d365fad906ac0acf387921a5474565

fail() {
    printf '%s\n' "$1" >&2
    exit 2
}

verify_sha256() {
    local path=$1
    local expected=$2
    local actual
    actual=$(sha256sum "$path" | cut -d' ' -f1)
    if [[ "$actual" != "$expected" ]]; then
        fail "SHA-256 mismatch for $path: expected $expected, got $actual"
    fi
}

[[ -x "$PYTHON_BIN" ]] || fail "Python environment is unavailable: $PYTHON_BIN"
VALIDATION_SOURCE_DIR=$(realpath -e "$VALIDATION_SOURCE_DIR")
RUN_ROOT=$(realpath -e "$RUN_ROOT")
if ! git -C "$VALIDATION_SOURCE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    fail "Final validation requires a Git worktree: $VALIDATION_SOURCE_DIR"
fi
if [[ -n "$(git -C "$VALIDATION_SOURCE_DIR" status --porcelain)" ]]; then
    git -C "$VALIDATION_SOURCE_DIR" status --short >&2
    fail "Final validation requires clean committed source"
fi

MATRIX_FILE=$VALIDATION_SOURCE_DIR/configs/star/shapo_p47_four_env.yaml
PROTOCOL_FILE=$VALIDATION_SOURCE_DIR/reports/star/shapo_p47_four_env_preregistration.json
FROZEN_CONFIG_FILE=$VALIDATION_SOURCE_DIR/reports/star/p4_7_safety_aware_geometry_artifacts/final/summary/p4_7_frozen_pointgoal_config.yaml
VALIDATOR_FILE=$VALIDATION_SOURCE_DIR/scripts/star/validate_shapo_p47_run.py
FINALIZER_FILE=$VALIDATION_SOURCE_DIR/scripts/star/finalize_shapo_p47_formal.py
verify_sha256 "$PROTOCOL_FILE" "$PROTOCOL_SHA256"
verify_sha256 "$MATRIX_FILE" "$MATRIX_SHA256"
verify_sha256 "$FROZEN_CONFIG_FILE" "$FROZEN_CONFIG_SHA256"

[[ -f "$RUN_ROOT/formal_matrix_complete" ]] || fail "Formal matrix completion seal is absent"
[[ -f "$RUN_ROOT/matrix_failure_count.txt" ]] || fail "Formal matrix failure count is absent"
[[ "$(<"$RUN_ROOT/matrix_failure_count.txt")" == 0 ]] || fail "Formal matrix records failures"

mapfile -t FORMAL_SOURCE_ROWS < <(sed -n 's/^source_commit=//p' "$RUN_ROOT/launch_manifest.txt")
if [[ ${#FORMAL_SOURCE_ROWS[@]} -ne 1 ]] || \
    [[ ! "${FORMAL_SOURCE_ROWS[0]}" =~ ^[0-9a-f]{40}$ ]]; then
    fail "Launch manifest does not contain exactly one full formal source commit"
fi
FORMAL_SOURCE_COMMIT=${FORMAL_SOURCE_ROWS[0]}
VALIDATION_COMMIT=$(git -C "$VALIDATION_SOURCE_DIR" rev-parse HEAD)
git -C "$VALIDATION_SOURCE_DIR" cat-file -e "$FORMAL_SOURCE_COMMIT^{commit}" || \
    fail "Formal source commit is unavailable in the validation repository"
git -C "$VALIDATION_SOURCE_DIR" merge-base --is-ancestor \
    "$FORMAL_SOURCE_COMMIT" "$VALIDATION_COMMIT" || \
    fail "Validation commit is not descended from the frozen formal source"
VALIDATOR_SHA256=$(sha256sum "$VALIDATOR_FILE" | cut -d' ' -f1)

OUTPUT_PARENT=$(dirname "$OUTPUT_ROOT")
OUTPUT_NAME=$(basename "$OUTPUT_ROOT")
mkdir -p "$OUTPUT_PARENT"
OUTPUT_PARENT=$(realpath -e "$OUTPUT_PARENT")
OUTPUT_ROOT=$OUTPUT_PARENT/$OUTPUT_NAME
[[ ! -e "$OUTPUT_ROOT" ]] || fail "Refusing to overwrite final artifacts: $OUTPUT_ROOT"
STAGING_ROOT=$(mktemp -d "$OUTPUT_PARENT/.shapo-p47-finalization.XXXXXX")
FINALIZATION_SUCCEEDED=0
report_staging_on_failure() {
    local status=$?
    if [[ "$FINALIZATION_SUCCEEDED" -ne 1 ]]; then
        printf 'SHAPO P4.7 finalization failed; staging retained at %s\n' \
            "$STAGING_ROOT" >&2
    fi
    return "$status"
}
trap report_staging_on_failure EXIT

mkdir -p "$STAGING_ROOT/strict_validations" "$STAGING_ROOT/protocol"
VALIDATION_MANIFEST=$STAGING_ROOT/shapo_p47_strict_validation_manifest.csv
printf '%s\n' \
    'environment,training_seed,artifact,sha256,bytes,validator_sha256,formal_source_commit,validation_commit' \
    >"$VALIDATION_MANIFEST"

ENV_IDS=(SafetyPointGoal1-v0 SafetyPointButton1-v0 Ant-v4 Walker2d-v4)
SLUGS=(point-goal point-button ant walker2d)
for seed in 2027 2028 2029 2030 2031; do
    for index in "${!ENV_IDS[@]}"; do
        env_id=${ENV_IDS[$index]}
        slug=${SLUGS[$index]}
        container=$RUN_ROOT/runs/$slug-$seed
        [[ -d "$container" ]] || fail "Formal run container is absent: $container"
        [[ -f "$container/source_commit.txt" ]] || \
            fail "Formal run source identity is absent: $container"
        [[ "$(<"$container/source_commit.txt")" == "$FORMAL_SOURCE_COMMIT" ]] || \
            fail "Formal run source identity changed: $container"
        (
            cd "$VALIDATION_SOURCE_DIR"
            export PYTHONPATH=$VALIDATION_SOURCE_DIR
            "$PYTHON_BIN" "$VALIDATOR_FILE" \
                --container "$container" \
                --matrix "$MATRIX_FILE" \
                --env-id "$env_id" \
                --seed "$seed" \
                --source-commit "$FORMAL_SOURCE_COMMIT" \
                >/dev/null
        )
        validation=$container/formal_validation.json
        "$PYTHON_BIN" - \
            "$validation" "$env_id" "$seed" "$VALIDATOR_SHA256" \
            "$FORMAL_SOURCE_COMMIT" <<'PY'
import json
import sys

path, environment, seed, validator_sha256, source_commit = sys.argv[1:]
payload = json.load(open(path, encoding='utf-8'))
expected = {
    'status': 'passed',
    'environment': environment,
    'training_seed': int(seed),
    'validator_sha256': validator_sha256,
    'source_commit': source_commit,
    'training_seed_is_statistical_unit': True,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f'{path}: field {key!r} does not match strict finalization')
PY
        portable_validation=$STAGING_ROOT/strict_validations/$slug-$seed.json
        cp -- "$validation" "$portable_validation"
        validation_sha256=$(sha256sum "$portable_validation" | cut -d' ' -f1)
        validation_bytes=$(stat -c '%s' "$portable_validation")
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$env_id" "$seed" "strict_validations/$slug-$seed.json" \
            "$validation_sha256" "$validation_bytes" "$VALIDATOR_SHA256" \
            "$FORMAL_SOURCE_COMMIT" "$VALIDATION_COMMIT" \
            >>"$VALIDATION_MANIFEST"
    done
done

(
    cd "$VALIDATION_SOURCE_DIR"
    export PYTHONPATH=$VALIDATION_SOURCE_DIR
    "$PYTHON_BIN" "$FINALIZER_FILE" \
        --run-root "$RUN_ROOT" \
        --matrix "$MATRIX_FILE" \
        --output-dir "$STAGING_ROOT" \
        >"$STAGING_ROOT/finalizer_stdout.json"
)

cp -- "$MATRIX_FILE" "$STAGING_ROOT/protocol/shapo_p47_four_env.yaml"
cp -- "$PROTOCOL_FILE" "$STAGING_ROOT/protocol/shapo_p47_four_env_preregistration.json"
cp -- "$FROZEN_CONFIG_FILE" "$STAGING_ROOT/protocol/p4_7_frozen_pointgoal_config.yaml"
cp -- "$VALIDATOR_FILE" "$STAGING_ROOT/protocol/validate_shapo_p47_run.py"
cp -- "$FINALIZER_FILE" "$STAGING_ROOT/protocol/finalize_shapo_p47_formal.py"

"$PYTHON_BIN" - \
    "$STAGING_ROOT/shapo_p47_final_summary.json" \
    "$STAGING_ROOT/finalization_metadata.json" \
    "$FORMAL_SOURCE_COMMIT" "$VALIDATION_COMMIT" "$VALIDATOR_SHA256" <<'PY'
from datetime import datetime, timezone
import json
import sys

summary_path, metadata_path, formal_commit, validation_commit, validator_sha256 = sys.argv[1:]
summary = json.load(open(summary_path, encoding='utf-8'))
expected_counts = {
    'evaluation_artifact_count': 60,
    'provenance_artifact_count': 80,
    'trajectory_artifact_count': 60,
    'checkpoint_artifact_count': 120,
}
if summary.get('source_commit') != formal_commit:
    raise SystemExit('final summary formal source commit mismatch')
if summary.get('validator_sha256') != validator_sha256:
    raise SystemExit('final summary strict validator mismatch')
if summary.get('statistical_unit') != 'training_seed':
    raise SystemExit('final summary changed the statistical unit')
if int(summary.get('n_training_seeds_per_environment', -1)) != 5:
    raise SystemExit('final summary does not contain five training seeds per environment')
for key, expected in expected_counts.items():
    if int(summary.get(key, -1)) != expected:
        raise SystemExit(f'final summary field {key!r} must equal {expected}')
metadata = {
    'status': 'passed',
    'formal_source_commit': formal_commit,
    'validation_commit': validation_commit,
    'validator_sha256': validator_sha256,
    'formal_runs': 20,
    'statistical_unit': 'training_seed',
    'n_training_seeds_per_environment': 5,
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
}
with open(metadata_path, 'w', encoding='utf-8') as stream:
    json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
PY

[[ "$(find "$STAGING_ROOT/strict_validations" -type f -name '*.json' | wc -l)" -eq 20 ]] || \
    fail "Strict validation bundle does not contain exactly 20 run records"
[[ -z "$(find "$STAGING_ROOT" -type l -print -quit)" ]] || \
    fail "Final evidence must not contain symbolic links"
(
    cd "$STAGING_ROOT"
    find . -type f ! -name checksums.sha256 ! -name finalization_complete -print0 \
        | sort -z \
        | xargs -0 sha256sum >checksums.sha256
    sha256sum -c checksums.sha256 >/dev/null
)
touch "$STAGING_ROOT/finalization_complete"
mv -- "$STAGING_ROOT" "$OUTPUT_ROOT"
FINALIZATION_SUCCEEDED=1
printf 'SHAPO P4.7 final artifacts complete: %s\n' "$OUTPUT_ROOT"
