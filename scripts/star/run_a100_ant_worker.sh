#!/usr/bin/env bash
set -euo pipefail

SAFEFLOW_ROOT=/home/lzqw/SAFEFLOW2027_A100
SAFEFLOW_CONTROLLER=${SAFEFLOW_ROOT}/code/controller
SAFEFLOW_TRAINING=${SAFEFLOW_ROOT}/code/training-62ca739
SAFEFLOW_PYTHON=${SAFEFLOW_ROOT}/env/bin/python
SAFEFLOW_RUN_ROOT=${SAFEFLOW_ROOT}/runs/ant
SAFEFLOW_SOURCE_COMMIT=62ca7399357c10645a955dcbead7ed2153a82c69
SAFEFLOW_C0_CONFIG_SHA=db16e3ed72b0a23a47bd9fd7db0f6aea863e8e1f338a35ada3d25618db4848b9
SAFEFLOW_P47_CONFIG_SHA=4d011ec9c0eaedf1f51835804c3e0476114e4a0427d8ef32a5b9a0336fc97ebf

run_job() {
    local gpu_index=$1
    local arm=$2
    local seed=$3
    local config_name=$4
    local config_sha=$5
    local container=${SAFEFLOW_RUN_ROOT}/runs/${arm}-${seed}
    local receipt=${container}/receipt.json
    local log=${container}/train.log
    local resume_path=''

    if [[ -f "${receipt}" ]] && "${SAFEFLOW_PYTHON}" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("status") == "passed" else 1)' \
        "${receipt}"; then
        return 0
    fi

    [[ "$(git -C "${SAFEFLOW_TRAINING}" rev-parse HEAD)" == "${SAFEFLOW_SOURCE_COMMIT}" ]]
    [[ -z "$(git -C "${SAFEFLOW_TRAINING}" status --porcelain)" ]]
    [[ "$(sha256sum "${SAFEFLOW_CONTROLLER}/configs/star/${config_name}" | cut -d' ' -f1)" == \
        "${config_sha}" ]]

    mkdir -p "${container}"
    resume_path=$(find "${container}" -type f -name latest_full.pt -printf '%T@ %p\n' \
        2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    local command=(
        "${SAFEFLOW_PYTHON}"
        "${SAFEFLOW_TRAINING}/scripts/star/train_star.py"
        --config "${SAFEFLOW_CONTROLLER}/configs/star/${config_name}"
        --seed "${seed}"
        --steps 2000000
        --steps-per-epoch 5000
        --device cuda:0
        --vector-envs 1
        --batch-size 256
        --replay-size 400000
        --checkpoint-replay
        --full-checkpoint-freq 250000
        --log-dir "${container}/training"
        --result "${receipt}"
        --no-tensorboard
    )
    if [[ -n "${resume_path}" ]]; then
        command+=(--resume "${resume_path}")
    fi

    (
        cd "${SAFEFLOW_TRAINING}"
        CUDA_VISIBLE_DEVICES="${gpu_index}" \
        MKL_NUM_THREADS=1 \
        OMP_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPATH="${SAFEFLOW_TRAINING}" \
        "${command[@]}"
    ) >>"${log}" 2>&1
}

case "${1:-}" in
    gpu0)
        run_job 0 C0 2029 single_gpu_ant_c0.yaml "${SAFEFLOW_C0_CONFIG_SHA}"
        ;;
    gpu1)
        run_job 1 P47 2029 single_gpu_ant_p47.yaml "${SAFEFLOW_P47_CONFIG_SHA}"
        ;;
    gpu0-c0-2027)
        run_job 0 C0 2027 single_gpu_ant_c0.yaml "${SAFEFLOW_C0_CONFIG_SHA}"
        ;;
    *)
        echo 'usage: run_a100_ant_worker.sh {gpu0|gpu1|gpu0-c0-2027}' >&2
        exit 2
        ;;
esac
