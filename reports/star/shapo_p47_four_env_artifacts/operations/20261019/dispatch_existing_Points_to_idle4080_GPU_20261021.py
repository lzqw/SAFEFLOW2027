#!/usr/bin/env python3
"""Fill newly stable spare4080 GPU with at most3 existing Point cells.

Uses existing1800-second frames; no new broad remote polling or training cells.
"""
import fcntl
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATE = BASE / 'Point_5080_to_4080_MPS_regular_dispatch_20261021_state.json'
CONTROLLER = BASE / 'handoff_Point_5080_to_4080_MPS_20261021.py'
ACCOUNTING = BASE / 'point_5080_to_4080_MPS_accounting_20261021.py'
SIDE = 'point_5080_to_4080_MPS_20261021'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    data['updated_at_unix'] = time.time()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def checkpoint_due(worker, frame):
    if worker['steps'] % 50000 <= 5000:
        return frame
    # The logged epoch may already be several minutes old. Use its own age,
    # not the resource-frame timestamp, then allow45s for atomic full-save.
    remaining = 50000 - worker['steps'] % 50000
    return frame - worker.get('progress_age_seconds', 0) + remaining / worker['last_epoch_fps'] + 45


def main():
    lock = (BASE / 'Point_5080_to_4080_MPS_regular_dispatch_20261021.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STATE.exists():
        s = read(STATE)
        assert s['phase'] in ('waiting_for_regular_GPU_capacity_and_fresh_full', 'all_requested_placements_completed'), 'Inspect original transaction; never replay'
        if s['phase'] == 'all_requested_placements_completed':
            return
    else:
        s = {'phase': 'waiting_for_regular_GPU_capacity_and_fresh_full', 'completed': [],
             'maximum_existing_cells_to_move': 3, 'attempted_frames': {},
             'controller_sha256': digest(CONTROLLER), 'accounting_module_sha256': digest(ACCOUNTING),
             'source_commit': '62ca7399357c10645a955dcbead7ed2153a82c69',
             'scientific_parameters_changed': False, 'additional_training_cells': 0,
             'additional_evaluation_rollouts': 0, 'other_projects_paused': False,
             'full_training_budget_each': 10000000, 'regular_poll_seconds': 1800,
             'candidate_order': 'largest_remaining_budget_among_fresh_full_existing_Point_workers',
             'GPU_stable_seconds': 1800, 'minimum_GPU_free_MiB': 8192, 'maximum_GPU_utilization_percent': 85}
    assert digest(CONTROLLER) == s['controller_sha256'] and digest(ACCOUNTING) == s['accounting_module_sha256']
    s.setdefault('checkpoint_attempts', {})
    s.update(checkpoint_query_minimum_interval_seconds=120, maximum_source_checkpoint_attempts_per_job_per_frame=2,
             source_checkpoint_timing_only_not_extra_resource_polling=True,
             candidate_order='largest_remaining_budget_among_existing_Point_workers_ready_for_predicted_full_checkpoint')
    s.update(local_pid=os.getpid(), local_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    save(STATE, s)
    while len(s['completed']) < s['maximum_existing_cells_to_move']:
        raw = read(BASE / 'gpu_capacity_state.json')
        frame = raw['observed_at_unix']
        if not 0 <= time.time() - frame <= 2100:
            time.sleep(20); continue
        target = raw['hosts'].get('4080', {})
        source = raw['hosts'].get('5080', {})
        gpu = next((g for g in target.get('gpus', []) if g['uuid'] == 'GPU-66ed7e12-a0c3-9273-463d-56a5e5f0a4f7'), None)
        if (source.get('status') != 'observed' or target.get('status') != 'observed'
                or not gpu or gpu['status'] != 'available_for_rebalance'
                or gpu['foreign_pids'] or gpu.get('MPS_accounting_unproven')
                or frame - gpu['candidate_since_unix'] < 1800):
            time.sleep(20); continue
        completed = {x['job'] for x in s['completed']}
        if time.time() < s.get('next_checkpoint_query_not_before', 0):
            time.sleep(20); continue
        candidates = [w for w in source.get('training', [])
                      if w['job'].startswith('point-') and w['job'] not in completed
                      and w['device'] == 'cuda:0' and w.get('cuda_execution_backend') == 'mps_verified'
                      and w.get('steps') is not None and w.get('last_epoch_fps', 0) > 0]
        ready = []
        future = []
        for worker in candidates:
            attempt = s['checkpoint_attempts'].get(worker['job'], {})
            count = attempt.get('count', 0) if attempt.get('frame') == frame else 0
            if count >= 2:
                continue
            due = attempt['next_due'] if count else checkpoint_due(worker, source['observed_at_unix'])
            if due <= time.time():ready.append(worker)
            else:future.append({'job': worker['job'], 'due_at_unix': due})
        candidates = ready
        next_query = min(future, key=lambda x: x['due_at_unix']) if future else None
        if s.get('next_predicted_checkpoint_query') != next_query:
            s['next_predicted_checkpoint_query'] = next_query
            save(STATE, s)
        candidates.sort(key=lambda w: (w['steps'], w['job']))
        for worker in candidates:
            if len(s['completed']) >= s['maximum_existing_cells_to_move']:
                break
            job = worker['job']
            path = BASE / (job + '_' + SIDE + '_state.json')
            assert not path.exists(), 'Inspect existing per-cell transaction, never repeat'
            s['attempted_frames'][job] = frame
            previous = s['checkpoint_attempts'].get(job, {})
            count = previous.get('count', 0) if previous.get('frame') == frame else 0
            s['checkpoint_attempts'][job] = {'frame': frame, 'count': count + 1, 'next_due': time.time() + 120}
            s['next_checkpoint_query_not_before'] = time.time() + 120
            native_name = 'Point_4080_GPU_placement_regular_frame_' + str(int(frame)) + '.json'
            native_path = BASE / native_name
            if not native_path.exists():
                with native_path.open('x') as stream: json.dump(raw, stream, indent=2)
            plan = {'job': job, 'source_commit': s['source_commit'], 'regular_frame_file': native_name,
                    'regular_frame_at_unix': frame, 'regular_gpu_status': gpu['status'],
                    'candidate_since_unix': gpu['candidate_since_unix'], 'expected_source_identity': worker,
                    'controller_sha256': s['controller_sha256'], 'accounting_module_sha256': s['accounting_module_sha256'],
                    'accounting_hook_integrated_before_any_transition': True}
            save(BASE / (job + '_' + SIDE + '_plan.json'), plan)
            spec = importlib.util.spec_from_file_location('Point_idle4080_' + job.replace('-', '_'), CONTROLLER)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            s.update(phase='one_existing_Point_GPU_transaction_in_progress', attempted_job=job)
            save(STATE, s)
            try:
                completed_move = module.run(job)
            except Exception as error:
                if path.exists():
                    s.update(phase='per_cell_transaction_needs_inspection_no_automatic_retry', error=str(error))
                    save(STATE, s)
                    raise
                s.update(preflight_deferral=str(error)[:700])
                completed_move = False
            if completed_move:
                s['completed'].append({'job': job, 'state_file': path.name, 'completed_at_unix': time.time()})
            s['phase'] = 'waiting_for_regular_GPU_capacity_and_fresh_full'
            save(STATE, s)
            # One source-only checkpoint attempt at a time. No sampling wave of
            # all Point jobs between ordinary1800-second resource frames.
            break
        time.sleep(20)
    s['phase'] = 'all_requested_placements_completed'
    save(STATE, s)


if __name__ == '__main__':
    main()
