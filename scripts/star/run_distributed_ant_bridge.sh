#!/usr/bin/env bash
set -euo pipefail

SAFEFLOW_STATE_ROOT=/home/lzqw/SAFEFLOW2027_single_gpu/a100-bridge/ant
SAFEFLOW_STAGE_ROOT=${SAFEFLOW_STATE_ROOT}/staging
SAFEFLOW_SEETA_RUN_ROOT=/root/autodl-tmp/SAFE/single_gpu_paper_core/ant-2m-v1
SAFEFLOW_A100_RUN_ROOT=/home/lzqw/SAFEFLOW2027_A100/runs/ant
SAFEFLOW_A100_LAUNCHER=/home/lzqw/SAFEFLOW2027_A100/launch_a100_ant.sh
SAFEFLOW_SEETA_PYTHON=/root/autodl-tmp/SAFE/envs/star-p0-p1/bin/python
SAFEFLOW_A100_PYTHON=/home/lzqw/SAFEFLOW2027_A100/env/bin/python
SAFEFLOW_QUEUE_PATTERN=/root/autodl-tmp/SAFE/code/SAFEFLOW2027-ant-controller-8d8e4b9/scripts/star/run_single_gpu_ant_queue.py
SAFEFLOW_QUEUE_PGREP='[r]un_single_gpu_ant_queue.py'
SAFEFLOW_P47_RACE_LAUNCHER=/root/autodl-tmp/SAFE/single_gpu_paper_core/ant-2m-v1/launch_p47_2029_distributed.sh
SAFEFLOW_C0_RACE_LAUNCHER=/root/autodl-tmp/SAFE/single_gpu_paper_core/ant-2m-v1/launch_c0_2029_distributed.sh
SAFEFLOW_QUEUE_RECOVERY_LAUNCHER=${SAFEFLOW_SEETA_RUN_ROOT}/launch_ant_queue_recovery.sh
SAFEFLOW_SOURCE_COMMIT=62ca7399357c10645a955dcbead7ed2153a82c69
SAFEFLOW_C0_CONFIG_SHA=db16e3ed72b0a23a47bd9fd7db0f6aea863e8e1f338a35ada3d25618db4848b9
SAFEFLOW_P47_CONFIG_SHA=4d011ec9c0eaedf1f51835804c3e0476114e4a0427d8ef32a5b9a0336fc97ebf
SAFEFLOW_POLL_SECONDS=300
SAFEFLOW_LAST_A100_GUARD=0
SAFEFLOW_QUEUE_RECOVERY_MODE=false
SAFEFLOW_SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)

mkdir -p "${SAFEFLOW_STATE_ROOT}" "${SAFEFLOW_STAGE_ROOT}/runs"

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

host_reachable() {
    local host=$1
    "${SAFEFLOW_SSH[@]}" "${host}" true >/dev/null 2>&1
}

wait_for_host() {
    local host=$1
    while ! host_reachable "${host}"; do
        if ! guard_a100_runs_if_due; then
            log 'A100 liveness guard could not confirm or restore both learners; retrying'
        fi
        log "${host} unavailable; waiting for SSH connectivity"
        sleep 60
    done
    log "${host} SSH connectivity available"
}

receipt_ok() {
    local host=$1
    local python_bin=$2
    local path=$3
    local arm=$4
    local seed=$5
    local config_sha=$6
    "${SAFEFLOW_SSH[@]}" "${host}" \
        "test -f '${path}' && '${python_bin}' -c 'import json,sys; p=json.load(open(sys.argv[1])); ok=(p.get(\"status\")==\"passed\" and p.get(\"algorithm\")==\"STAR\" and p.get(\"environment\")==\"Ant-v4\" and int(p.get(\"seed\",-1))==int(sys.argv[3]) and int(p.get(\"steps\",-1))==2000000 and p.get(\"source_commit\")==sys.argv[4] and p.get(\"source_dirty\") is False and p.get(\"config_sha256\")==sys.argv[2]); raise SystemExit(0 if ok else 1)' '${path}' '${config_sha}' '${seed}' '${SAFEFLOW_SOURCE_COMMIT}'" \
        >/dev/null 2>&1
}

a100_launch_identity_ok() {
    local config_name=$1
    local config_sha=$2
    "${SAFEFLOW_SSH[@]}" tjucta-a100 \
        "test -x '${SAFEFLOW_A100_LAUNCHER}' && test \"\$(git -C '/home/lzqw/SAFEFLOW2027_A100/code/training-62ca739' rev-parse HEAD)\" = '${SAFEFLOW_SOURCE_COMMIT}' && test -z \"\$(git -C '/home/lzqw/SAFEFLOW2027_A100/code/training-62ca739' status --porcelain)\" && test \"\$(sha256sum '/home/lzqw/SAFEFLOW2027_A100/code/controller/configs/star/${config_name}' | cut -d' ' -f1)\" = '${config_sha}'" \
        >/dev/null 2>&1
}

ensure_a100_run() {
    local arm=$1
    local config_name=$2
    local config_sha=$3
    local session=$4
    local gpu_arg=$5
    if receipt_ok seeta-nmb2 "${SAFEFLOW_SEETA_PYTHON}" \
        "${SAFEFLOW_SEETA_RUN_ROOT}/runs/${arm}-2029/receipt.json" \
        "${arm}" 2029 "${config_sha}"; then
        stop_a100_duplicate "${session}" "${arm}-2029"
        return 0
    fi
    if receipt_ok tjucta-a100 "${SAFEFLOW_A100_PYTHON}" \
        "${SAFEFLOW_A100_RUN_ROOT}/runs/${arm}-2029/receipt.json" \
        "${arm}" 2029 "${config_sha}"; then
        return 0
    fi
    if "${SAFEFLOW_SSH[@]}" tjucta-a100 \
        "tmux has-session -t '${session}'" >/dev/null 2>&1; then
        return 0
    fi
    if "${SAFEFLOW_SSH[@]}" tjucta-a100 \
        "pgrep -f '[t]rain_star.py .*${config_name} --seed 2029'" >/dev/null 2>&1; then
        log "${arm}-2029 learner is alive without its expected tmux session; leaving it untouched"
        return 0
    fi
    if ! a100_launch_identity_ok "${config_name}" "${config_sha}"; then
        log "refusing to restart ${arm}-2029 because the frozen A100 source/config identity did not authenticate"
        return 1
    fi
    "${SAFEFLOW_SSH[@]}" tjucta-a100 \
        "tmux new-session -d -s '${session}' '${SAFEFLOW_A100_LAUNCHER}' '${gpu_arg}'"
    log "restarted ${arm}-2029 through its replay-aware A100 launcher"
}

guard_a100_runs_if_due() {
    local now
    local healthy=true
    now=$(date +%s)
    if (( now - SAFEFLOW_LAST_A100_GUARD < 300 )); then
        return 0
    fi
    SAFEFLOW_LAST_A100_GUARD=${now}
    ensure_a100_run C0 single_gpu_ant_c0.yaml "${SAFEFLOW_C0_CONFIG_SHA}" \
        safeflow-ant-gpu0 gpu0 || healthy=false
    ensure_a100_run P47 single_gpu_ant_p47.yaml "${SAFEFLOW_P47_CONFIG_SHA}" \
        safeflow-ant-gpu1 gpu1 || healthy=false
    [[ "${healthy}" == true ]]
}

resolve_one_pid() {
    local host=$1
    local pattern=$2
    local pids
    pids=$("${SAFEFLOW_SSH[@]}" "${host}" "pgrep -f '${pattern}'" 2>/dev/null || true)
    if [[ ! "${pids}" =~ ^[0-9]+$ ]]; then
        log "expected one PID for ${pattern}, found: ${pids:-none}"
        return 1
    fi
    printf '%s\n' "${pids}"
}

process_alive_non_zombie() {
    local host=$1
    local pid=$2
    local state
    state=$("${SAFEFLOW_SSH[@]}" "${host}" "ps -o stat= -p '${pid}'" 2>/dev/null || true)
    [[ -n "${state}" && "${state}" != Z* ]]
}

ensure_seeta_runner() {
    local arm=$1
    local seed=$2
    local config_sha=$3
    local config_name=$4
    local launcher=$5
    local launcher_name=${launcher##*/}
    if receipt_ok seeta-nmb2 "${SAFEFLOW_SEETA_PYTHON}" \
        "${SAFEFLOW_SEETA_RUN_ROOT}/runs/${arm}-${seed}/receipt.json" \
        "${arm}" "${seed}" "${config_sha}"; then
        return 0
    fi
    if "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
        "pgrep -f '[l]${launcher_name#l}' >/dev/null || pgrep -f '[t]rain_star.py .*${config_name} --seed ${seed}' >/dev/null"; then
        return 0
    fi
    "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
        "nohup setsid '${launcher}' >/dev/null 2>&1 </dev/null &"
    log "restarted ${arm}-${seed} through its replay-aware Seeta launcher"
}

wave_one_arm_healthy() {
    local arm=$1
    local pid=$2
    local config_sha=$3
    if receipt_ok seeta-nmb2 "${SAFEFLOW_SEETA_PYTHON}" \
        "${SAFEFLOW_SEETA_RUN_ROOT}/runs/${arm}-2028/receipt.json" \
        "${arm}" 2028 "${config_sha}"; then
        return 0
    fi
    [[ "${pid}" =~ ^[0-9]+$ ]] && process_alive_non_zombie seeta-nmb2 "${pid}"
}

wave_one_learners_healthy() {
    wave_one_arm_healthy C0 "${SAFEFLOW_C0_2028_PID:-}" "${SAFEFLOW_C0_CONFIG_SHA}" && \
        wave_one_arm_healthy P47 "${SAFEFLOW_P47_2028_PID:-}" "${SAFEFLOW_P47_CONFIG_SHA}"
}

resume_queue() {
    if [[ "${SAFEFLOW_QUEUE_RECOVERY_MODE}" == true ]]; then
        "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
            "set -e; test -x '${SAFEFLOW_QUEUE_RECOVERY_LAUNCHER}'; if pgrep -f '${SAFEFLOW_QUEUE_PGREP}' >/dev/null; then exit 0; fi; nohup setsid '${SAFEFLOW_QUEUE_RECOVERY_LAUNCHER}' >'${SAFEFLOW_SEETA_RUN_ROOT}/controller-recovery-supervisor.out' 2>&1 </dev/null &"
        log 'started the strict Seeta queue through its restart-safe recovery launcher'
        return 0
    fi
    "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
        "ps -p '${SAFEFLOW_QUEUE_PID}' -o args= | grep -F '${SAFEFLOW_QUEUE_PATTERN}' >/dev/null && kill -CONT '${SAFEFLOW_QUEUE_PID}'"
    log "resumed Seeta Ant queue controller PID ${SAFEFLOW_QUEUE_PID}"
}

stop_a100_duplicate() {
    local session=$1
    local label=$2
    if "${SAFEFLOW_SSH[@]}" tjucta-a100 "tmux has-session -t '${session}'" 2>/dev/null; then
        "${SAFEFLOW_SSH[@]}" tjucta-a100 "tmux send-keys -t '${session}' C-c"
        log "stopped completed ${label} duplicate on A100"
    fi
}

sync_a100_run() {
    local arm=$1
    local seed=$2
    local marker=${SAFEFLOW_STATE_ROOT}/synced-${arm}-${seed}
    local local_container=${SAFEFLOW_STAGE_ROOT}/runs/${arm}-${seed}
    if [[ -f "${marker}" ]]; then
        return 0
    fi
    mkdir -p "${local_container}"
    log "staging ${arm}-${seed} from A100 to host"
    rsync -a --partial \
        "tjucta-a100:${SAFEFLOW_A100_RUN_ROOT}/runs/${arm}-${seed}/" \
        "${local_container}/"
    log "copying ${arm}-${seed} from host to Seeta"
    rsync -a --partial \
        "${local_container}/" \
        "seeta-nmb2:${SAFEFLOW_SEETA_RUN_ROOT}/runs/${arm}-${seed}/"
    local config_sha=${SAFEFLOW_C0_CONFIG_SHA}
    if [[ "${arm}" == P47 ]]; then
        config_sha=${SAFEFLOW_P47_CONFIG_SHA}
    fi
    receipt_ok \
        seeta-nmb2 \
        "${SAFEFLOW_SEETA_PYTHON}" \
        "${SAFEFLOW_SEETA_RUN_ROOT}/runs/${arm}-${seed}/receipt.json" \
        "${arm}" \
        "${seed}" \
        "${config_sha}"
    touch "${marker}"
    log "validated ${arm}-${seed} on Seeta"
}

stage_incomplete_a100_run() {
    local arm=$1
    local seed=$2
    local minimum_step=$3
    local a100_container=${SAFEFLOW_A100_RUN_ROOT}/runs/${arm}-${seed}
    local seeta_container=${SAFEFLOW_SEETA_RUN_ROOT}/runs/${arm}-${seed}
    local local_container=${SAFEFLOW_STAGE_ROOT}/recovery/${arm}-${seed}
    local source_checkpoint
    local checkpoint_relative
    local local_checkpoint
    local checkpoint_sha
    local source_sha
    local staged_sha
    local checkpoint_step

    source_checkpoint=$("${SAFEFLOW_SSH[@]}" tjucta-a100 \
        "find '${a100_container}/training' -type f -path '*/checkpoints/latest_full.pt' -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-" \
        2>/dev/null || true)
    if [[ -z "${source_checkpoint}" ]]; then
        return 1
    fi
    checkpoint_step=$("${SAFEFLOW_SSH[@]}" tjucta-a100 \
        "'${SAFEFLOW_A100_PYTHON}' -c 'import sys,torch; p=torch.load(sys.argv[1],map_location=\"cpu\",weights_only=False); c=p.get(\"counters\",{}); e=p.get(\"extra\",{}); r=p.get(\"replay\"); step=int(c.get(\"env_steps\",-1)); ok=(p.get(\"format\")==\"star-checkpoint\" and e.get(\"environment\")==\"Ant-v4\" and int(e.get(\"seed\",-1))==int(sys.argv[2]) and int(sys.argv[3])<=step<2000000 and isinstance(r,dict) and int(r.get(\"capacity\",-1))==400000 and int(r.get(\"size\",-1))==min(step,400000)); print(step); raise SystemExit(0 if ok else 1)' '${source_checkpoint}' '${seed}' '${minimum_step}'" \
        2>/dev/null || true)
    if [[ ! "${checkpoint_step}" =~ ^[0-9]+$ ]] || (( checkpoint_step < minimum_step )); then
        return 1
    fi

    checkpoint_relative=${source_checkpoint#"${a100_container}/"}
    if [[ "${checkpoint_relative}" == "${source_checkpoint}" ]]; then
        log "refusing checkpoint outside ${arm}-${seed} container: ${source_checkpoint}"
        return 1
    fi
    mkdir -p "${local_container}"
    log "staging ${arm}-${seed} replay checkpoint at step ${checkpoint_step} from A100"
    rsync -a --partial --exclude=/receipt.json \
        "tjucta-a100:${a100_container}/" \
        "${local_container}/"
    local_checkpoint=${local_container}/${checkpoint_relative}
    if [[ ! -s "${local_checkpoint}" ]]; then
        log "staged ${arm}-${seed} checkpoint is missing"
        return 1
    fi
    checkpoint_sha=$(sha256sum "${local_checkpoint}" | cut -d' ' -f1)
    source_sha=$("${SAFEFLOW_SSH[@]}" tjucta-a100 \
        "sha256sum '${source_checkpoint}' | cut -d' ' -f1" 2>/dev/null || true)
    if [[ "${checkpoint_sha}" != "${source_sha}" ]]; then
        log "A100 ${arm}-${seed} checkpoint advanced during transfer; retrying later"
        return 1
    fi

    "${SAFEFLOW_SSH[@]}" seeta-nmb2 "mkdir -p '${seeta_container}'"
    rsync -a --partial "${local_container}/" "seeta-nmb2:${seeta_container}/"
    staged_sha=$("${SAFEFLOW_SSH[@]}" seeta-nmb2 \
        "sha256sum '${seeta_container}/${checkpoint_relative}' | cut -d' ' -f1" \
        2>/dev/null || true)
    if [[ "${checkpoint_sha}" != "${staged_sha}" ]]; then
        log "Seeta ${arm}-${seed} checkpoint digest mismatch"
        return 1
    fi
    if ! "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
        "'${SAFEFLOW_SEETA_PYTHON}' -c 'import sys,torch; p=torch.load(sys.argv[1],map_location=\"cpu\",weights_only=False); c=p.get(\"counters\",{}); e=p.get(\"extra\",{}); r=p.get(\"replay\"); step=int(c.get(\"env_steps\",-1)); ok=(p.get(\"format\")==\"star-checkpoint\" and e.get(\"environment\")==\"Ant-v4\" and int(e.get(\"seed\",-1))==int(sys.argv[2]) and step==int(sys.argv[3]) and isinstance(r,dict) and int(r.get(\"capacity\",-1))==400000 and int(r.get(\"size\",-1))==min(step,400000)); raise SystemExit(0 if ok else 1)' '${seeta_container}/${checkpoint_relative}' '${seed}' '${checkpoint_step}'"; then
        log "Seeta ${arm}-${seed} replay checkpoint validation failed"
        return 1
    fi
    touch "${SAFEFLOW_STATE_ROOT}/staged-recovery-${arm}-${seed}-${checkpoint_step}-${checkpoint_sha}"
    log "validated ${arm}-${seed} replay recovery on Seeta at step ${checkpoint_step} (${checkpoint_sha})"
}

while true; do
    wait_for_host seeta-nmb2
    queue_state=$("${SAFEFLOW_SSH[@]}" seeta-nmb2 \
        "'${SAFEFLOW_SEETA_PYTHON}' -c 'import json; print(json.load(open(\"${SAFEFLOW_SEETA_RUN_ROOT}/queue_state.json\"))[\"state\"])'" \
        2>/dev/null || true)
    if [[ "${queue_state}" == complete ]]; then
        log 'Ant queue is already complete'
        exit 0
    fi
    SAFEFLOW_C0_2028_PID=
    SAFEFLOW_P47_2028_PID=
    if receipt_ok seeta-nmb2 "${SAFEFLOW_SEETA_PYTHON}" \
        "${SAFEFLOW_SEETA_RUN_ROOT}/runs/C0-2028/receipt.json" C0 2028 "${SAFEFLOW_C0_CONFIG_SHA}" || \
       SAFEFLOW_C0_2028_PID=$(resolve_one_pid seeta-nmb2 '[s]ingle_gpu_ant_c0.yaml --seed 2028'); then
        SAFEFLOW_C0_2028_READY=true
    else
        SAFEFLOW_C0_2028_READY=false
    fi
    if receipt_ok seeta-nmb2 "${SAFEFLOW_SEETA_PYTHON}" \
        "${SAFEFLOW_SEETA_RUN_ROOT}/runs/P47-2028/receipt.json" P47 2028 "${SAFEFLOW_P47_CONFIG_SHA}" || \
       SAFEFLOW_P47_2028_PID=$(resolve_one_pid seeta-nmb2 '[s]ingle_gpu_ant_p47.yaml --seed 2028'); then
        SAFEFLOW_P47_2028_READY=true
    else
        SAFEFLOW_P47_2028_READY=false
    fi
    if [[ "${SAFEFLOW_C0_2028_READY}" == true ]] && \
       [[ "${SAFEFLOW_P47_2028_READY}" == true ]]; then
        if SAFEFLOW_QUEUE_PID=$(resolve_one_pid seeta-nmb2 "${SAFEFLOW_QUEUE_PGREP}"); then
            break
        fi
        if "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
            "test -x '${SAFEFLOW_QUEUE_RECOVERY_LAUNCHER}'" >/dev/null 2>&1; then
            SAFEFLOW_QUEUE_RECOVERY_MODE=true
            log 'wave-one receipts are valid but the old queue controller is absent; entering restart-safe post-wave mode'
            break
        fi
    fi
    log 'Seeta is reachable but the wave-one controller and healthy learner/receipt states are not all present; waiting for recovery'
    sleep 60
done

if [[ "${SAFEFLOW_QUEUE_RECOVERY_MODE}" == false ]]; then
    "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
        "ps -p '${SAFEFLOW_QUEUE_PID}' -o args= | grep -F '${SAFEFLOW_QUEUE_PATTERN}' >/dev/null && kill -STOP '${SAFEFLOW_QUEUE_PID}'"
    sleep 1
    SAFEFLOW_QUEUE_STAT=$("${SAFEFLOW_SSH[@]}" seeta-nmb2 "ps -o stat= -p '${SAFEFLOW_QUEUE_PID}'")
    if [[ "${SAFEFLOW_QUEUE_STAT}" != T* ]]; then
        log "failed to freeze queue controller: state=${SAFEFLOW_QUEUE_STAT}"
        exit 1
    fi
    log "froze Seeta queue controller PID ${SAFEFLOW_QUEUE_PID}; current learners continue"
else
    log 'no live Seeta queue controller to freeze; current recovered learners continue'
fi

while true; do
    if ! host_reachable seeta-nmb2; then
        wait_for_host seeta-nmb2
        continue
    fi
    if receipt_ok seeta-nmb2 "${SAFEFLOW_SEETA_PYTHON}" \
        "${SAFEFLOW_SEETA_RUN_ROOT}/runs/C0-2028/receipt.json" C0 2028 "${SAFEFLOW_C0_CONFIG_SHA}" && \
       receipt_ok seeta-nmb2 "${SAFEFLOW_SEETA_PYTHON}" \
        "${SAFEFLOW_SEETA_RUN_ROOT}/runs/P47-2028/receipt.json" P47 2028 "${SAFEFLOW_P47_CONFIG_SHA}"; then
        break
    fi
    if ! wave_one_learners_healthy; then
        log 'Seeta learner status query failed once; retrying before restoring the queue'
        sleep 15
        if ! wave_one_learners_healthy; then
            sleep 30
            if ! wave_one_learners_healthy; then
                log 'one current Seeta learner exited without a valid receipt after retries; restoring original queue'
                resume_queue
                exit 0
            fi
        fi
    fi
    sleep "${SAFEFLOW_POLL_SECONDS}"
done
log 'Seeta wave one completed; launching C0-2027 on the freed RTX 3090'

ensure_seeta_runner C0 2027 "${SAFEFLOW_C0_CONFIG_SHA}" \
    single_gpu_ant_c0.yaml "${SAFEFLOW_SEETA_RUN_ROOT}/launch_c0_2027_distributed.sh"

if ! receipt_ok seeta-nmb2 "${SAFEFLOW_SEETA_PYTHON}" \
    "${SAFEFLOW_SEETA_RUN_ROOT}/runs/P47-2029/receipt.json" P47 2029 "${SAFEFLOW_P47_CONFIG_SHA}" && \
   ! receipt_ok tjucta-a100 "${SAFEFLOW_A100_PYTHON}" \
    "${SAFEFLOW_A100_RUN_ROOT}/runs/P47-2029/receipt.json" P47 2029 "${SAFEFLOW_P47_CONFIG_SHA}"; then
    if ! "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
        "pgrep -f '[l]aunch_p47_2029_distributed.sh' >/dev/null || pgrep -f '[t]rain_star.py .*single_gpu_ant_p47.yaml --seed 2029' >/dev/null || pgrep -f '[l]aunch_c0_2029_distributed.sh' >/dev/null || pgrep -f '[t]rain_star.py .*single_gpu_ant_c0.yaml --seed 2029' >/dev/null"; then
        while ! stage_incomplete_a100_run P47 2029 250000; do
            if receipt_ok tjucta-a100 "${SAFEFLOW_A100_PYTHON}" \
                "${SAFEFLOW_A100_RUN_ROOT}/runs/P47-2029/receipt.json" \
                P47 2029 "${SAFEFLOW_P47_CONFIG_SHA}"; then
                break
            fi
            if ! "${SAFEFLOW_SSH[@]}" tjucta-a100 \
                "tmux has-session -t safeflow-ant-gpu1" 2>/dev/null; then
                log 'A100 P47-2029 has no usable recovery; launching the RTX race from scratch'
                break
            fi
            sleep 60
        done
        if ! receipt_ok tjucta-a100 "${SAFEFLOW_A100_PYTHON}" \
            "${SAFEFLOW_A100_RUN_ROOT}/runs/P47-2029/receipt.json" \
            P47 2029 "${SAFEFLOW_P47_CONFIG_SHA}"; then
            ensure_seeta_runner P47 2029 "${SAFEFLOW_P47_CONFIG_SHA}" \
                single_gpu_ant_p47.yaml "${SAFEFLOW_P47_RACE_LAUNCHER}"
            log 'launched P47-2029 replay-resumed work-conserving race on the second RTX learner slot'
        fi
    fi
fi

while true; do
    if ! guard_a100_runs_if_due; then
        log 'A100 liveness guard could not confirm or restore both learners; retrying'
    fi
    SAFEFLOW_C0_2029=false
    SAFEFLOW_SEETA_C0=false
    SAFEFLOW_SEETA_P47=false
    if receipt_ok seeta-nmb2 "${SAFEFLOW_SEETA_PYTHON}" \
        "${SAFEFLOW_SEETA_RUN_ROOT}/runs/C0-2029/receipt.json" C0 2029 "${SAFEFLOW_C0_CONFIG_SHA}"; then
        SAFEFLOW_C0_2029=true
    elif receipt_ok tjucta-a100 "${SAFEFLOW_A100_PYTHON}" \
        "${SAFEFLOW_A100_RUN_ROOT}/runs/C0-2029/receipt.json" C0 2029 "${SAFEFLOW_C0_CONFIG_SHA}"; then
        SAFEFLOW_C0_2029=true
        sync_a100_run C0 2029
    fi
    if receipt_ok seeta-nmb2 "${SAFEFLOW_SEETA_PYTHON}" \
        "${SAFEFLOW_SEETA_RUN_ROOT}/runs/C0-2027/receipt.json" C0 2027 "${SAFEFLOW_C0_CONFIG_SHA}"; then
        SAFEFLOW_SEETA_C0=true
    fi
    if receipt_ok seeta-nmb2 "${SAFEFLOW_SEETA_PYTHON}" \
        "${SAFEFLOW_SEETA_RUN_ROOT}/runs/P47-2029/receipt.json" P47 2029 "${SAFEFLOW_P47_CONFIG_SHA}"; then
        SAFEFLOW_SEETA_P47=true
    elif receipt_ok tjucta-a100 "${SAFEFLOW_A100_PYTHON}" \
        "${SAFEFLOW_A100_RUN_ROOT}/runs/P47-2029/receipt.json" P47 2029 "${SAFEFLOW_P47_CONFIG_SHA}" && \
         ! "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
            "pgrep -f '[t]rain_star.py .*single_gpu_ant_p47.yaml --seed 2029' >/dev/null"; then
        sync_a100_run P47 2029
        SAFEFLOW_SEETA_P47=true
    fi
    if [[ "${SAFEFLOW_SEETA_C0}" == false ]]; then
        ensure_seeta_runner C0 2027 "${SAFEFLOW_C0_CONFIG_SHA}" \
            single_gpu_ant_c0.yaml "${SAFEFLOW_SEETA_RUN_ROOT}/launch_c0_2027_distributed.sh"
    fi
    if [[ "${SAFEFLOW_SEETA_P47}" == false ]] && \
       ! receipt_ok tjucta-a100 "${SAFEFLOW_A100_PYTHON}" \
        "${SAFEFLOW_A100_RUN_ROOT}/runs/P47-2029/receipt.json" P47 2029 "${SAFEFLOW_P47_CONFIG_SHA}" && \
       ! "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
        "pgrep -f '[l]aunch_c0_2029_distributed.sh' >/dev/null || pgrep -f '[t]rain_star.py .*single_gpu_ant_c0.yaml --seed 2029' >/dev/null"; then
        ensure_seeta_runner P47 2029 "${SAFEFLOW_P47_CONFIG_SHA}" \
            single_gpu_ant_p47.yaml "${SAFEFLOW_P47_RACE_LAUNCHER}"
    fi
    if [[ "${SAFEFLOW_SEETA_P47}" == true && "${SAFEFLOW_C0_2029}" == false ]] && \
       ! "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
        "pgrep -f '[l]aunch_c0_2029_distributed.sh' >/dev/null || pgrep -f '[t]rain_star.py .*single_gpu_ant_c0.yaml --seed 2029' >/dev/null"; then
        if stage_incomplete_a100_run C0 2029 250000; then
            "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
                "nohup setsid '${SAFEFLOW_C0_RACE_LAUNCHER}' >/dev/null 2>&1 </dev/null &"
            log 'launched C0-2029 replay-resumed race on the RTX slot released by P47-2029'
        else
            log 'waiting for a valid A100 C0-2029 replay checkpoint before RTX migration'
        fi
    fi
    if [[ "${SAFEFLOW_C0_2029}" == true && "${SAFEFLOW_SEETA_P47}" == true && \
          "${SAFEFLOW_SEETA_C0}" == true ]]; then
        break
    fi
    if [[ "${SAFEFLOW_C0_2029}" == false ]] && \
       ! "${SAFEFLOW_SSH[@]}" tjucta-a100 "tmux has-session -t safeflow-ant-gpu0" 2>/dev/null && \
       ! "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
        "pgrep -f '[l]aunch_c0_2029_distributed.sh' >/dev/null || pgrep -f '[t]rain_star.py .*single_gpu_ant_c0.yaml --seed 2029' >/dev/null"; then
        log 'A100 C0-2029 stopped without a valid receipt; waiting for both RTX race runs before fallback'
        while ! receipt_ok seeta-nmb2 "${SAFEFLOW_SEETA_PYTHON}" \
            "${SAFEFLOW_SEETA_RUN_ROOT}/runs/C0-2027/receipt.json" C0 2027 "${SAFEFLOW_C0_CONFIG_SHA}" || \
              ! receipt_ok seeta-nmb2 "${SAFEFLOW_SEETA_PYTHON}" \
            "${SAFEFLOW_SEETA_RUN_ROOT}/runs/P47-2029/receipt.json" P47 2029 "${SAFEFLOW_P47_CONFIG_SHA}"; do
            sleep "${SAFEFLOW_POLL_SECONDS}"
        done
        resume_queue
        exit 0
    fi
    sleep "${SAFEFLOW_POLL_SECONDS}"
done

log 'all required Ant runs passed across the work-conserving schedule; resuming the frozen queue for strict finalization'
resume_queue

while true; do
    queue_state=$("${SAFEFLOW_SSH[@]}" seeta-nmb2 \
        "'${SAFEFLOW_SEETA_PYTHON}' -c 'import json; print(json.load(open(\"${SAFEFLOW_SEETA_RUN_ROOT}/queue_state.json\"))[\"state\"])'" \
        2>/dev/null || true)
    case "${queue_state}" in
        complete)
            log 'Ant distributed queue finalized successfully'
            stop_a100_duplicate safeflow-ant-gpu0 C0-2029
            stop_a100_duplicate safeflow-ant-gpu1 P47-2029
            exit 0
            ;;
        failed)
            log 'Ant strict finalization reported failure'
            exit 1
            ;;
        training)
            if [[ "${SAFEFLOW_QUEUE_RECOVERY_MODE}" == true ]] && \
               ! "${SAFEFLOW_SSH[@]}" seeta-nmb2 \
                "pgrep -f '${SAFEFLOW_QUEUE_PGREP}'" >/dev/null 2>&1; then
                resume_queue
            fi
            ;;
    esac
    sleep 30
done
