#!/usr/bin/env bash
# Audit and summarize the completed P4.6 matrix against sealed P4.5 evidence.

set -euo pipefail

SOURCE_DIR=${1:-$PWD}
RUN_ROOT=${2:-/root/autodl-tmp/SAFE/p4_6_direction_weight_pareto}
PYTHON_BIN=${3:-/root/autodl-tmp/SAFE/envs/star-p0-p1/bin/python}
BASELINE_ROOT=${4:-$SOURCE_DIR/reports/star/p4_5_unified_cost_geometry_artifacts/final}

mkdir -p "$RUN_ROOT/audits/work" "$RUN_ROOT/summary"
trap 'status=$?; echo "$status" >"$RUN_ROOT/postprocess_exit_code.txt"; if [[ "$status" -eq 0 ]]; then touch "$RUN_ROOT/postprocess_complete"; fi' EXIT

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment is unavailable: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -d "$SOURCE_DIR/.git" ]] || [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
    echo "Post-processing requires the clean formal P4.6 source tree" >&2
    exit 2
fi
if [[ ! -f "$RUN_ROOT/training_matrix_complete" ]]; then
    echo "P4.6 training matrix is not complete" >&2
    exit 2
fi
if [[ ! -f "$BASELINE_ROOT/checksums.sha256" ]]; then
    echo "Sealed P4.5 baseline evidence is unavailable: $BASELINE_ROOT" >&2
    exit 2
fi

resolve_checkpoint() {
    local container=$1
    local matches=()
    mapfile -t matches < <(find "$container" -type f -path '*/checkpoints/final_full.pt' | sort)
    if [[ "${#matches[@]}" -ne 1 ]]; then
        echo "$container has ${#matches[@]} final full checkpoints, expected one" >&2
        return 2
    fi
    echo "${matches[0]}"
}

run_bindings=()
audit_bindings=()
for arm in C025 C050; do
    for seed in 2027 2028 2029; do
        identity="${arm}-${seed}"
        container="$RUN_ROOT/runs/$identity"
        receipt="$container/receipt.json"
        "$PYTHON_BIN" - "$receipt" "$arm" "$seed" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding='utf-8'))
expected_seed = int(sys.argv[3])
if (
    payload.get('status') != 'passed'
    or int(payload.get('steps', -1)) != 100_000
    or int(payload.get('seed', -1)) != expected_seed
    or payload.get('source_dirty') is not False
):
    raise SystemExit(f'invalid formal receipt for {sys.argv[2]}-{expected_seed}')
PY
        checkpoint=$(resolve_checkpoint "$container")
        audit="$RUN_ROOT/audits/$identity.json"
        "$PYTHON_BIN" "$SOURCE_DIR/scripts/star/analyze_p4_6_cost_geometry.py" \
            --checkpoint "$checkpoint" \
            --output "$audit" \
            --work-dir "$RUN_ROOT/audits/work/$identity" \
            --device cuda:0 \
            --state-count 64 \
            --action-samples 256 \
            --audit-seed 4500007 \
            >"$RUN_ROOT/audits/$identity.log" 2>&1
        run_bindings+=("${arm}:${seed}=$container")
        audit_bindings+=("${arm}:${seed}=$audit")
    done
done

(
    cd "$SOURCE_DIR"
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH=$SOURCE_DIR
    "$PYTHON_BIN" scripts/star/summarize_p4_6_direction_weight_pareto.py \
        --protocol reports/star/p4_6_direction_weight_pareto_preregistration.json \
        --baseline-evidence-root "$BASELINE_ROOT" \
        --runs "${run_bindings[@]}" \
        --audits "${audit_bindings[@]}" \
        --output-dir "$RUN_ROOT/summary"
    "$PYTHON_BIN" scripts/star/plot_p4_6_direction_weight_pareto.py \
        --summary-dir "$RUN_ROOT/summary" \
        --output-dir "$RUN_ROOT/summary"
) >"$RUN_ROOT/postprocess.log" 2>&1
