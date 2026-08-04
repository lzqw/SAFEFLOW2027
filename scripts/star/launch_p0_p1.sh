#!/usr/bin/env bash
# Unified launcher for the STAR P0/P1 gates.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SAFE_ROOT="${SAFE_ROOT:-/root/autodl-tmp/SAFE}"
PYTHON="${SAFE_ROOT}/envs/star-p0-p1/bin/python"
STAGE="${1:-}"

export PIP_CACHE_DIR="${SAFE_ROOT}/cache/pip"
export TMPDIR="${SAFE_ROOT}/cache/tmp"
export XDG_CACHE_HOME="${SAFE_ROOT}/cache/xdg"
export HF_HOME="${SAFE_ROOT}/cache/huggingface"
export TORCH_HOME="${SAFE_ROOT}/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="${SAFE_ROOT}/cache/torchinductor"
export WANDB_DIR="${SAFE_ROOT}/runs/active"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${REPO_ROOT}"

case "${STAGE}" in
  p0-capture)
    "${PYTHON}" scripts/star/capture_server_state.py \
      --safe-root "${SAFE_ROOT}" \
      --repo "${REPO_ROOT}" \
      --manifest-output "${SAFE_ROOT}/artifacts/p0/system_manifest.json" \
      --lock-output "${SAFE_ROOT}/artifacts/p0/environment.lock.txt"
    ;;
  p0-env-smoke)
    "${PYTHON}" scripts/star/smoke_four_envs.py \
      --steps 1000 \
      --seed 2027 \
      --output "${SAFE_ROOT}/artifacts/p0/four_env_smoke.json"
    ;;
  p0-saclag-5k)
    "${PYTHON}" scripts/star/run_saclag_smoke.py \
      --env-id SafetyPointGoal1-v0 \
      --steps 5000 \
      --steps-per-epoch 1000 \
      --seed 2027 \
      --device cuda:0 \
      --log-dir "${SAFE_ROOT}/runs/active/p0-saclag-5k" \
      --result "${SAFE_ROOT}/artifacts/p0/saclag_pointgoal_5k.json"
    ;;
  p0-benchmark-point)
    "${PYTHON}" scripts/star/benchmark_throughput.py \
      --env-id SafetyPointGoal1-v0 \
      --safe-root "${SAFE_ROOT}" \
      --configs 1x1 1x4 2x2 3x1 4x1 \
      --warmup-seconds 60 \
      --duration-seconds 600 \
      --output-dir "${SAFE_ROOT}/artifacts/p0/benchmark-point" \
      --csv "${SAFE_ROOT}/artifacts/p0/throughput_benchmark.csv"
    ;;
  p0-benchmark-mujoco)
    "${PYTHON}" scripts/star/benchmark_throughput.py \
      --env-id Ant-v4 \
      --safe-root "${SAFE_ROOT}" \
      --configs 1x1 2x1 3x1 4x1 \
      --warmup-seconds 60 \
      --duration-seconds 600 \
      --output-dir "${SAFE_ROOT}/artifacts/p0/benchmark-ant" \
      --csv "${SAFE_ROOT}/artifacts/p0/throughput_benchmark.csv" \
      --append
    ;;
  p1-tests)
    "${PYTHON}" -m pytest -q tests/star/test_p1_semantics.py
    ;;
  p1-smoke-20k)
    "${PYTHON}" scripts/star/run_p1_smoke_matrix.py \
      --steps 20000 \
      --steps-per-epoch 1000 \
      --device cuda:0 \
      --output-dir "${SAFE_ROOT}/runs/active/p1-smoke-20k" \
      --summary "${SAFE_ROOT}/artifacts/p1/p1_smoke_20k.json" \
      --resume
    ;;
  tensorboard)
    exec "${PYTHON}" -m tensorboard.main \
      --logdir "${SAFE_ROOT}/runs" \
      --port 6006 \
      --bind_all
    ;;
  *)
    echo "Usage: $0 {p0-capture|p0-env-smoke|p0-saclag-5k|p0-benchmark-point|p0-benchmark-mujoco|p1-tests|p1-smoke-20k|tensorboard}" >&2
    exit 2
    ;;
esac
