#!/usr/bin/env python3
"""Balance two existing Ant cells, one at a time, using existing30min frames."""
import fcntl
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATE = BASE / 'Ant_CPU_5080_to_4080_regular_dispatch_20261020_state.json'
CONTROLLER = BASE / 'handoff_Ant_CPU_5080_to_4080_20261020.py'
ACCOUNTING = BASE / 'cpu_5080_to_4080_recovery_accounting_20261020.py'
SIDE = 'cpu_5080_to_4080_20261020'


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    data['updated_at_unix'] = time.time()
    part = path.with_suffix('.tmp')
    part.write_text(json.dumps(data, indent=2) + '\n')
    part.replace(path)


def main():
    lock = (BASE / 'Ant_CPU_5080_to_4080_regular_dispatch_20261020.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STATE.exists():
        state = read(STATE)
        assert state['phase'] in ('waiting_on_regular_capacity_and_fresh_full', 'all_requested_handoffs_completed')
        if state['phase'] == 'all_requested_handoffs_completed':
            return
    else:
        proof = read(BASE / '4080_CPU_placement_preflight_20261020_2051.json')
        assert proof['phase'] == 'read_only_CPU_placement_preflight_completed'
        assert proof['target_CPU_proof']['CPU_headroom_eligible_under_existing_conservative_policy'] is True
        state = {'phase': 'waiting_on_regular_capacity_and_fresh_full',
                 'pending': ['ant-2028', 'ant-2029'], 'completed': [], 'last_attempted_frame': {},
                 'controller_sha256': hashlib.sha256(CONTROLLER.read_bytes()).hexdigest(),
                 'accounting_module_sha256': hashlib.sha256(ACCOUNTING.read_bytes()).hexdigest(),
                 'initial_CPU_qualification': '4080_CPU_placement_preflight_20261020_2051.json',
                 'source_originals_preserved': True, 'scientific_parameters_changed': False,
                 'new_experiment_cells': 0, 'other_projects_paused': False,
                 'full_training_budget_each': 5000000, 'ordinary_poll_seconds': 1800}
    assert hashlib.sha256(CONTROLLER.read_bytes()).hexdigest() == state['controller_sha256']
    assert hashlib.sha256(ACCOUNTING.read_bytes()).hexdigest() == state['accounting_module_sha256']
    state.update(local_pid=os.getpid(), local_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    save(STATE, state)
    while state['pending']:
        raw = read(BASE / 'gpu_capacity_state.json')
        frame = raw.get('observed_at_unix', 0)
        if not 0 <= time.time() - frame <= 2100:
            time.sleep(30)
            continue
        dst = raw.get('hosts', {}).get('4080', {})
        src = raw.get('hosts', {}).get('5080', {})
        target_ants = [w for w in dst.get('training', []) if w['job'].startswith('ant-')
                       and w['device'] == 'cpu' and w.get('last_epoch_fps', 0) > 0]
        for job in state['pending'][:]:
            handoff = BASE / (job + '_' + SIDE + '_state.json')
            assert not handoff.exists(), 'Inspect original per-cell transaction; never repeat it'
            if state['last_attempted_frame'].get(job) == frame:
                continue
            source = next((w for w in src.get('training', []) if w['job'] == job and w['device'] == 'cpu'), None)
            if (not source or source['steps'] % 50000 > 5000 or not target_ants
                    or min(w['last_epoch_fps'] for w in target_ants) < 1.2 * source['last_epoch_fps']):
                continue
            state['last_attempted_frame'][job] = frame
            native_name = 'Ant_CPU_placement_regular_frame_' + str(int(frame)) + '.json'
            native_path = BASE / native_name
            if not native_path.exists():
                native_path.write_text(json.dumps(raw, indent=2) + '\n')
            plan = {'job': job, 'seed': int(job.rsplit('-', 1)[1]), 'environment': 'Ant-v4',
                    'source_host': '5080', 'target_host': '4080',
                    'source_commit': '62ca7399357c10645a955dcbead7ed2153a82c69',
                    'total_training_budget': 5000000, 'source_device': 'cpu', 'target_device': 'cpu',
                    'source_cuda_visible_devices': '-1', 'target_cuda_visible_devices': '-1',
                    'expected_source_identity': source,
                    'expected_target_boot_id': '3b362c03-4ed3-4e03-aea4-cf7339356026',
                    'regular_native_observed_at_unix': frame, 'regular_native_file': native_name,
                    'initial_CPU_qualification': state['initial_CPU_qualification'],
                    'scientific_parameters_changed': False, 'additional_evaluation': False,
                    'additional_experiment_cells': 0, 'no_other_projects_stopped': True,
                    'GPU_capacity_not_claimed_or_used': True,
                    'accounting_hook_integrated_before_any_transition': True,
                    'controller_sha256': state['controller_sha256'],
                    'accounting_module_sha256': state['accounting_module_sha256'],
                    'rpc_helper_filename': 'rebalance_goal2030_a100_to_5080_20261020_0621.py',
                    'rpc_helper_sha256': '690fb3391274fe30f08aabb1b91f8df246276547bac9b3bd21a3f3a2c4158323',
                    'requires_first_new_CPU_epoch_before_source_retirement': True}
            assert 'new_crosshost_cpu_balances = verify_Ant_CPU_5080_to_4080(BASE)' in (BASE / 'publish_final.py').read_text()
            save(BASE / (job + '_' + SIDE + '_plan.json'), plan)
            spec = importlib.util.spec_from_file_location('fresh_CPU_controller_' + job.replace('-', '_'), CONTROLLER)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.configure(job)
            state.update(phase='checking_one_existing_cell_before_bounded_handoff', attempted_job=job)
            save(STATE, state)
            try:
                completed = module.main()
            except Exception as error:
                if handoff.exists():
                    state.update(phase='per_cell_transaction_needs_inspection_no_automatic_retry', error=str(error))
                    save(STATE, state)
                    raise
                state.update(phase='waiting_on_regular_capacity_and_fresh_full', preflight_deferral=str(error)[:600])
                save(STATE, state)
                continue
            if completed:
                state['pending'].remove(job)
                state['completed'].append({'job': job, 'state_file': handoff.name, 'completed_at_unix': time.time()})
            else:
                assert not handoff.exists()
            state['phase'] = 'waiting_on_regular_capacity_and_fresh_full'
            save(STATE, state)
        if state['pending']:
            time.sleep(30)
    state['phase'] = 'all_requested_handoffs_completed'
    save(STATE, state)


if __name__ == '__main__':
    main()
