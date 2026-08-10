#!/usr/bin/env bash
# Validate, summarize, plot, and package compact P4.7 evidence outside the source tree.

set -euo pipefail

SOURCE_DIR=${1:-$PWD}
RUN_ROOT=${2:-/root/autodl-tmp/SAFE/p4_7_safety_aware_geometry}
PYTHON_BIN=${3:-/root/autodl-tmp/SAFE/envs/star-p0-p1/bin/python}
P45_ROOT=${4:-/root/autodl-tmp/SAFE/p4_5_532d4b1/runs}
OUTPUT_ROOT=${5:-$RUN_ROOT/public_artifacts}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment is unavailable: $PYTHON_BIN" >&2
    exit 2
fi
if [[ -n "$(git -C "$SOURCE_DIR" status --porcelain)" ]]; then
    echo "P4.7 finalization requires a clean committed source tree" >&2
    git -C "$SOURCE_DIR" status --short >&2
    exit 2
fi
if [[ ! -f "$RUN_ROOT/training_matrix_complete" ]]; then
    echo "P4.7 training matrix is incomplete" >&2
    exit 2
fi
if [[ ! -f "$RUN_ROOT/evaluation/evaluation_complete" ]]; then
    echo "P4.7 evaluation matrix is incomplete" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to overwrite existing P4.7 public artifacts: $OUTPUT_ROOT" >&2
    exit 2
fi

mkdir -p \
    "$OUTPUT_ROOT/summary" \
    "$OUTPUT_ROOT/figures" \
    "$OUTPUT_ROOT/receipts" \
    "$OUTPUT_ROOT/formal_evaluations" \
    "$OUTPUT_ROOT/evaluation_tables" \
    "$OUTPUT_ROOT/manifests"

single_file() {
    local root=$1
    local name=$2
    local matches=()
    mapfile -t matches < <(find "$root" -type f -name "$name" -print | sort)
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "Expected one $name below $root, found ${#matches[@]}" >&2
        exit 2
    fi
    printf '%s\n' "${matches[0]}"
}

(
    cd "$SOURCE_DIR"
    export PYTHONPATH=$SOURCE_DIR
    "$PYTHON_BIN" scripts/star/summarize_p4_7_safety_aware_geometry.py \
        --protocol reports/star/p4_7_safety_aware_geometry_preregistration.json \
        --run-root "$RUN_ROOT" \
        --p45-root "$P45_ROOT" \
        --output "$OUTPUT_ROOT/summary" \
        >"$OUTPUT_ROOT/summary/finalizer_stdout.json"
    "$PYTHON_BIN" scripts/star/plot_p4_7_safety_aware_geometry.py \
        --summary-dir "$OUTPUT_ROOT/summary" \
        --output-dir "$OUTPUT_ROOT/figures" \
        >"$OUTPUT_ROOT/figures/plot_manifest.json"
)

for arm in C0 P47; do
    for seed in 2027 2028 2029; do
        cp "$RUN_ROOT/runs/$arm-$seed/receipt.json" \
            "$OUTPUT_ROOT/receipts/$arm-$seed.json"
        cp "$(single_file "$RUN_ROOT/runs/$arm-$seed/training" run_manifest.json)" \
            "$OUTPUT_ROOT/manifests/$arm-$seed-run.json"
        cp "$(single_file "$RUN_ROOT/runs/$arm-$seed/training" training_summary.json)" \
            "$OUTPUT_ROOT/manifests/$arm-$seed-training.json"
    done
done

for arm in C0 C1 P47; do
    mkdir -p "$OUTPUT_ROOT/formal_evaluations/$arm"
    cp "$RUN_ROOT/evaluation/formal/$arm"/evaluation_seed_*_step_*.json \
        "$OUTPUT_ROOT/formal_evaluations/$arm/"
    cp "$RUN_ROOT/evaluation/intermediate/$arm/pointgoal_checkpoint_evaluation.csv" \
        "$OUTPUT_ROOT/evaluation_tables/${arm}_intermediate.csv"
    cp "$RUN_ROOT/evaluation/formal/$arm/pointgoal_checkpoint_evaluation.csv" \
        "$OUTPUT_ROOT/evaluation_tables/${arm}_formal.csv"
done
cp "$RUN_ROOT/launch_manifest.txt" "$OUTPUT_ROOT/manifests/training_launch.txt"
cp "$RUN_ROOT/evaluation/evaluation_manifest.txt" "$OUTPUT_ROOT/manifests/evaluation.txt"

cat >"$OUTPUT_ROOT/README.md" <<'EOF'
# P4.7 compact final evidence

This directory contains the locked P4.7 summary, all arm-by-training-seed formal evaluation
records, common-protocol intermediate tables, clean-source receipts/manifests, figures, and
checkpoint SHA-256 identities. The primary scientific comparison is equal-budget P47@200k versus
C0@200k; C1@100k is a historical reference. The original preregistered machine audit is preserved
separately. Large model/replay checkpoints and evaluator work directories remain server-only.
High-risk geometry is diagnostic and did not enter the selection gates.
EOF

(
    cd "$OUTPUT_ROOT"
    find . -type f ! -name checksums.sha256 -print0 \
        | sort -z \
        | xargs -0 sha256sum >checksums.sha256
)
touch "$OUTPUT_ROOT/finalization_complete"
echo "P4.7 public artifacts complete: $OUTPUT_ROOT"
