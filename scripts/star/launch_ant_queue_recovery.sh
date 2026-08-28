#!/usr/bin/env bash
set -euo pipefail

SAFEFLOW_CONTROLLER=/root/autodl-tmp/SAFE/code/SAFEFLOW2027-ant-controller-8d8e4b9
SAFEFLOW_TRAINING=/root/autodl-tmp/SAFE/code/SAFEFLOW2027-ant-training-62ca739
SAFEFLOW_PYTHON=/root/autodl-tmp/SAFE/envs/star-p0-p1/bin/python
SAFEFLOW_RUN_ROOT=/root/autodl-tmp/SAFE/single_gpu_paper_core/ant-2m-v1
SAFEFLOW_POINT_STATE=/root/autodl-tmp/SAFE/single_gpu_paper_core/point-robustness-enhanced-v1/queue_state.json
SAFEFLOW_REUSED_CHECKPOINT='/root/autodl-tmp/SAFE/shapo_p47_four_env/formal/runs/ant-2027/training/STAR-{Ant-v4}/seed-2027-2026-08-14-23-27-40/checkpoints/milestone_2000000.pt'
SAFEFLOW_CONTROLLER_COMMIT=8d8e4b9927847d2d8e8baf1e247278ca515a8e5f
SAFEFLOW_TRAINING_COMMIT=62ca7399357c10645a955dcbead7ed2153a82c69
SAFEFLOW_REUSED_CHECKPOINT_SHA=39a9d7a0717402d5e3fea5694cd0724aebd0982306beb1fb39c8091135fddc35
SAFEFLOW_LOG=${SAFEFLOW_RUN_ROOT}/controller-recovery.log

mkdir -p "${SAFEFLOW_RUN_ROOT}"
exec 9>"${SAFEFLOW_RUN_ROOT}/controller-recovery.lock"
flock -n 9 || exit 0

if pgrep -f '[r]un_single_gpu_ant_queue.py' >/dev/null; then
    exit 0
fi
if pgrep -f '[s]ingle_gpu_ant_c0.yaml --seed 20' >/dev/null || \
   pgrep -f '[s]ingle_gpu_ant_p47.yaml --seed 20' >/dev/null; then
    printf 'refusing controller recovery while an Ant learner is still alive\n' >&2
    exit 1
fi

[[ "$(git -C "${SAFEFLOW_CONTROLLER}" rev-parse HEAD)" == "${SAFEFLOW_CONTROLLER_COMMIT}" ]]
[[ -z "$(git -C "${SAFEFLOW_CONTROLLER}" status --porcelain)" ]]
[[ "$(git -C "${SAFEFLOW_TRAINING}" rev-parse HEAD)" == "${SAFEFLOW_TRAINING_COMMIT}" ]]
[[ -z "$(git -C "${SAFEFLOW_TRAINING}" status --porcelain)" ]]
[[ -x "${SAFEFLOW_PYTHON}" ]]
[[ -f "${SAFEFLOW_POINT_STATE}" ]]
[[ "$(sha256sum "${SAFEFLOW_REUSED_CHECKPOINT}" | cut -d' ' -f1)" == \
    "${SAFEFLOW_REUSED_CHECKPOINT_SHA}" ]]

exec "${SAFEFLOW_PYTHON}" \
    "${SAFEFLOW_CONTROLLER}/scripts/star/run_single_gpu_ant_queue.py" \
    --controller-source "${SAFEFLOW_CONTROLLER}" \
    --training-source "${SAFEFLOW_TRAINING}" \
    --run-root "${SAFEFLOW_RUN_ROOT}" \
    --python-bin "${SAFEFLOW_PYTHON}" \
    --point-core-state "${SAFEFLOW_POINT_STATE}" \
    --reused-p47-2027-checkpoint "${SAFEFLOW_REUSED_CHECKPOINT}" \
    --expected-controller-commit "${SAFEFLOW_CONTROLLER_COMMIT}" \
    --poll-seconds 30 \
    >>"${SAFEFLOW_LOG}" 2>&1
