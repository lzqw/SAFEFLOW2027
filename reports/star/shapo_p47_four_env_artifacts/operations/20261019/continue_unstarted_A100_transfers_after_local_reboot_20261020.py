#!/usr/bin/env python3
"""Continue only two never-started per-cell transfers after local host reboot.

Never replays an old controller main, restarts a learner, signals stale local
PIDs, or repeats a partial transaction. The original immutable transition and
accounting functions retain their original guards and evidence formats.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATE = BASE / 'A100_unstarted_transfers_post_local_reboot_20261020_state.json'
ENTRIES = (
    {'job': 'ant-2030', 'module': 'handoff_ant2030_A100_GPU0_MPS_20261020.py',
     'sha': '18eed33ca1cd365251e81a37b81c3b43e5ed9b0b3760accee0da751b7744c1ca',
     'preparation': 'cpu_to_mps_ant2030_a100_gpu0_20261020_preparation.json',
     'per_cell': 'ant-2030_cpu_to_mps_a100_gpu0_20261020_state.json'},
    {'job': 'walker2d-2030', 'module': 'handoff_walkers_A100_GPU1_MPS_20261020.py',
     'sha': '532fe6429ca46c59636da17912adde6e641700278438a116fa034d797d7885b7',
     'preparation': 'cpu_to_mps_walkers_a100_gpu1_20261020_preparation.json',
     'per_cell': 'walker2d-2030_cpu_to_mps_a100_gpu1_20261020_state.json'},
)


def save(value):
    value['updated_at_unix'] = time.time()
    part = STATE.with_suffix('.tmp')
    part.write_text(json.dumps(value, indent=2) + '\n')
    part.replace(STATE)
    print(json.dumps({'phase': value['phase'], 'pending': value['pending'],
                      'updated_at_unix': value['updated_at_unix']}), flush=True)


def load(entry):
    path = BASE / entry['module']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha']
    spec = importlib.util.spec_from_file_location('post_reboot_' + entry['job'].replace('-', '_'), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    # Persistent service may resume a WAIT after another local reboot, never a
    # partly executed handoff. No old PID receives any signal in this script.
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if STATE.exists():
        state = json.loads(STATE.read_text())
        assert state['phase'] in ('waiting_on_regular_capacity', 'all_requested_transfers_finished')
        if state['phase'] == 'all_requested_transfers_finished':
            return
    else:
        state = {'phase': 'waiting_on_regular_capacity', 'pending': [e['job'] for e in ENTRIES],
                 'completed': [], 'last_attempted_frame': {}, 'capacity_deferrals': [],
                 'old_coordinator_mains_never_replayed': True, 'stale_local_PIDs_never_signaled': True,
                 'new_seeds': 0, 'scientific_parameters_changed': False,
                 'foreign_projects_modified': False, 'regular_observation_seconds': 1800}
    state.update(local_boot_id=boot, local_pid=os.getpid(), not_before_observation_unix=time.time())
    save(state)
    while state['pending']:
        raw = json.loads((BASE / 'gpu_capacity_state.json').read_text())
        host = raw.get('hosts', {}).get('tjucta-a100', {})
        observed = host.get('observed_at_unix', 0)
        for entry in ENTRIES:
            job = entry['job']
            if job not in state['pending']:
                continue
            if (BASE / entry['per_cell']).exists():
                raise RuntimeError('Per-cell transaction now exists: inspect it; never repeat ' + job)
            if not (observed >= state['not_before_observation_unix'] and 0 <= time.time() - observed <= 2100):
                continue
            if state['last_attempted_frame'].get(job) == observed:
                continue
            m = load(entry)
            gpu = next((g for g in host.get('gpus', []) if g['uuid'] == m.GPU), None)
            source = next((w for w in host.get('training', []) if w['job'] == job and w['device'] == 'cpu'), None)
            if not gpu or not source or gpu.get('MPS_accounting_unproven'):
                continue
            ceiling = 60 if gpu.get('foreign_pids') else 85
            if gpu['free_mib'] < 16384 or gpu['utilization_percent'] > ceiling:
                continue
            state['last_attempted_frame'][job] = observed
            expected = next(w for w in json.loads((BASE / entry['preparation']).read_text())['workers_at_ordinary_observation'] if w['job'] == job)
            assert all(expected[k] == source[k] for k in ('pid', 'start_ticks', 'boot_id', 'progress_csv'))
            h = m.helpers(job)
            try:
                m.identity(h, expected)
                live = m.capacity(h)
            except (AssertionError, OSError, ValueError, subprocess.SubprocessError) as error:
                state['capacity_deferrals'].append({'job': job, 'frame': observed, 'error': str(error)[:500]})
                save(state)
                continue
            # Existing Walker31 daemon remains in place. Ant0 may not exist yet.
            controller = (json.loads(m.STATE.read_text()) if m.STATE.exists()
                          else {'private_daemon_started': False, 'completed_jobs': []})
            state.update(phase='attempting_original_never_started_per_cell_transition',
                         attempted_job=job, live_capacity=live)
            save(state)
            try:
                completed = m.transition(h, expected, controller)
            except Exception as error:
                state.update(phase='needs_inspection_no_automatic_retry', error=str(error))
                save(state)
                raise
            if completed:
                state['pending'].remove(job)
                state['completed'].append({'job': job, 'per_cell_state': entry['per_cell'],
                                           'completed_at_unix': time.time()})
            else:
                assert not (BASE / entry['per_cell']).exists()
            state['phase'] = 'waiting_on_regular_capacity'
            save(state)
        if state['pending']:
            # Local file polling only: all broad SSH observations remain30min.
            time.sleep(60)
    state['phase'] = 'all_requested_transfers_finished'
    save(state)


if __name__ == '__main__':
    main()
