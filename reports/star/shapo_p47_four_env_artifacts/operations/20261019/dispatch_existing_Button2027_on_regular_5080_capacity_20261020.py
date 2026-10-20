#!/usr/bin/env python3
"""Use existing30min observations to balance the slowest existing Point cell.

No additional seed or scientific evaluation. A genuine fresh full checkpoint,
same9-client stable target cohort, speed headroom and all immutable handoff
guards are mandatory. Partial transactions stop for inspection, never retry.
"""
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATE = BASE / 'Button2027_5080_regular_capacity_dispatch_20261020_state.json'
CONTROLLER = BASE / 'handoff_Button2027_A100_to_5080_MPS_20261020.py'
ACCOUNTING = BASE / 'crosshost_button2027_MPS_accounting_20261020.py'
HANDOFF = BASE / 'point-button-2027_balanced_MPS_5080_20261020_state.json'
PLAN = BASE / 'point_button2027_5080_MPS_balanced_20261020.json'
BOOT = '5f46bb0c-b4ec-4802-8f6e-01ac611c0f2f'
GPU = 'GPU-ac315b09-0bb7-2072-efd1-cdfc5fae96a5'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    value['updated_at_unix'] = time.time()
    part = path.with_suffix('.tmp')
    part.write_text(json.dumps(value, indent=2) + '\n')
    part.replace(path)


def cohort(raw):
    host = raw.get('hosts', {}).get('5080', {})
    gpu = next((g for g in host.get('gpus', []) if g['uuid'] == GPU), {})
    workers = [w for w in host.get('training', []) if w.get('mps_gpu_uuid_verified') == GPU]
    return host, gpu, workers


def main():
    assert not HANDOFF.exists(), 'Existing transaction: inspect, never duplicate'
    if STATE.exists():
        state = read(STATE)
        assert state['phase'] == 'waiting_on_regular_observation'
    else:
        state = {'phase': 'waiting_on_regular_observation', 'poll_seconds': 1800,
                 'pending_existing_job': 'point-button-2027', 'new_experiment_cells': 0,
                 'scientific_parameters_changed': False, 'foreign_projects_modified': False,
                 'controller_sha256': hashlib.sha256(CONTROLLER.read_bytes()).hexdigest(),
                 'accounting_sha256': hashlib.sha256(ACCOUNTING.read_bytes()).hexdigest(),
                 'previous_regular_raw': read(BASE / 'capacity_progress_observation_20261020_1921.json')}
    assert hashlib.sha256(CONTROLLER.read_bytes()).hexdigest() == state['controller_sha256']
    assert hashlib.sha256(ACCOUNTING.read_bytes()).hexdigest() == state['accounting_sha256']
    # This distinct, persistent scheduler owns only a never-started transaction.
    state.update(local_pid=os.getpid(), local_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 not_before_observation_unix=time.time())
    save(STATE, state)
    while True:
        assert not HANDOFF.exists(), 'Transaction appeared: inspect its live handle'
        raw = read(BASE / 'gpu_capacity_state.json')
        host, gpu, workers = cohort(raw)
        t = host.get('observed_at_unix', 0)
        if (t < state['not_before_observation_unix'] or t == state.get('last_frame')
                or not 0 <= time.time() - t <= 2100):
            time.sleep(30)
            continue
        old = state['previous_regular_raw']
        previous_host, previous_gpu, previous_workers = cohort(old)
        pids = sorted(w['pid'] for w in workers)
        source = next((w for w in raw.get('hosts', {}).get('tjucta-a100', {}).get('training', [])
                       if w['job'] == 'point-button-2027'), None)
        ready = (len(workers) == 9 and pids == sorted(w['pid'] for w in previous_workers)
                 and 1800 <= t - previous_host.get('observed_at_unix', 0) <= 3700
                 and gpu.get('status') == 'available_for_rebalance'
                 and not gpu.get('foreign_pids') and not previous_gpu.get('foreign_pids')
                 and not gpu.get('MPS_accounting_unproven')
                 and gpu.get('free_mib', 0) >= 8192
                 and all(w.get('boot_id') == BOOT and w.get('last_epoch_fps', 0) > 0 for w in workers)
                 and source is not None and source.get('last_epoch_fps', 0) > 0)
        if ready:
            ready = (source['last_epoch_fps'] * 1.5 <= min(w['last_epoch_fps'] for w in workers)
                     and source['steps'] % 50000 <= 5000)
        state.update(last_frame=t, previous_regular_raw=raw,
                     qualification_status='ready_for_immutable_live_preflight' if ready else
                     'waiting_for_stable_same_cohort_capacity_speed_gain_and_fresh50k_full_boundary')
        save(STATE, state)
        if not ready:
            time.sleep(30)
            continue
        plan = read(BASE / 'point_goal2027_5080_MPS_balanced_20261020.json')
        plan.update(job='point-button-2027', source_gpu_uuid='GPU-164dfe85-4b4b-0f44-fcd8-99ca6d1e8b78',
                    expected_source_identity=source, controller_sha256=state['controller_sha256'],
                    accounting_module_sha256=state['accounting_sha256'])
        proof = workers[0]['mps_connection']
        plan['rebalance_qualification'] = {
            'kind': 'existing_cohort_makespan_balance', 'uuid': GPU, 'owned_gpu_pids': pids,
            'MPS_server_pid': proof['server_pid'], 'foreign_pids': [], 'observed_at_unix': t,
            'previous_observed_at_unix': previous_host['observed_at_unix'],
            'source_last_epoch_fps': source['last_epoch_fps'],
            'target_minimum_last_epoch_fps': min(w['last_epoch_fps'] for w in workers),
            'source_native_step_anchor': source['steps'],
            'current_native_GPU': gpu, 'previous_native_GPU': previous_gpu,
            'same_nine_native_MPS_clients_across_regular_frames': True}
        save(PLAN, plan)
        assert 'verify_Button2027_crosshost_MPS(BASE)' in (BASE / 'publish_final.py').read_text()
        spec = importlib.util.spec_from_file_location('button2027_live_crosshost_handoff', CONTROLLER)
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        state['phase'] = 'immutable_handoff_live_preflight'
        save(STATE, state)
        try:
            module.run()
        except Exception as error:
            if HANDOFF.exists():
                state.update(phase='transaction_exists_inspect_no_automatic_retry', error=str(error))
                save(STATE, state)
                raise
            # No transaction/source pause/target launch happened. The next
            # regular frame can newly qualify, not replay a started controller.
            state.update(phase='waiting_on_regular_observation', preflight_deferral=str(error)[:500])
            save(STATE, state)
            continue
        state.update(phase='existing_Button2027_handoff_completed', handoff_state=HANDOFF.name)
        save(STATE, state)
        return


if __name__ == '__main__':
    main()
