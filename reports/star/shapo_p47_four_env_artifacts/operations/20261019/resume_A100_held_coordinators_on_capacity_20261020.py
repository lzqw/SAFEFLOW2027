#!/usr/bin/env python3
"""Release only our two idle local coordinators on fresh shared GPU headroom.

This never signals a learner, changes science, starts a new experiment, or edits
an executed controller. Original controllers recheck capacity before any pause.
"""
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATE = BASE/'A100_held_coordinator_capacity_resumer_20261020_state.json'
HOST = 'tjucta-a100'
ENTRIES = [
    {'job':'ant-2030', 'pid':256662, 'start_ticks':10847651,
     'script':'handoff_ant2030_A100_GPU0_MPS_20261020.py',
     'sha256':'18eed33ca1cd365251e81a37b81c3b43e5ed9b0b3760accee0da751b7744c1ca',
     'gpu_uuid':'GPU-7b6a829b-7b56-af3e-cf56-1af815c62dbe',
     'source_learner_pid':14550,
     'per_cell_state':'ant-2030_cpu_to_mps_a100_gpu0_20261020_state.json'},
    {'job':'walker2d-2030', 'pid':259143, 'start_ticks':10945243,
     'script':'handoff_walkers_A100_GPU1_MPS_20261020.py',
     'sha256':'532fe6429ca46c59636da17912adde6e641700278438a116fa034d797d7885b7',
     'gpu_uuid':'GPU-164dfe85-4b4b-0f44-fcd8-99ca6d1e8b78',
     'source_learner_pid':14547,
     'per_cell_state':'walker2d-2030_cpu_to_mps_a100_gpu1_20261020_state.json'},
]


def save(value):
    value['updated_at_unix'] = time.time()
    part = STATE.with_suffix('.tmp')
    part.write_text(json.dumps(value, indent=2)+'\n')
    part.replace(STATE)
    print(json.dumps({'phase':value['phase'], 'pending_jobs':[e['job'] for e in value['pending']],
                      'updated_at_unix':value['updated_at_unix']}), flush=True)


def checked_module(entry):
    path = BASE/entry['script']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    spec = importlib.util.spec_from_file_location('held_'+entry['job'].replace('-','_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.GPU==entry['gpu_uuid']
    return module


def checked_local_coordinator(entry):
    assert entry['pid'] not in (14547,14550,2384563)
    p = Path('/proc')/str(entry['pid'])
    stat = (p/'stat').read_text().rsplit(') ',1)[1].split()
    assert int(stat[19])==entry['start_ticks']
    args = [v.decode() for v in (p/'cmdline').read_bytes().split(bytes([0])) if v]
    assert str(BASE/entry['script']) in args
    assert stat[0]=='T', 'Coordinator not held: inspect rather than duplicate or stop it'
    assert not (p/'task'/str(entry['pid'])/'children').read_text().strip()
    assert not (BASE/entry['per_cell_state']).exists()


def main():
    assert not STATE.exists(), 'Never duplicate a resumer; inspect its original PID'
    # Reject the old17:21 quiet GPU0 frame: live17:30 preflight showed foreign100%.
    minimum_observation = time.time()
    for entry in ENTRIES:
        checked_local_coordinator(entry)
        checked_module(entry)
    state = {'phase':'waiting_for_fresh_regular_shared_GPU_headroom',
             'pending':ENTRIES[:], 'released':[],
             'not_before_observation_unix':minimum_observation,
             'only_local_coordinators_may_receive_SIGCONT':True,
             'training_learners_and_foreign_projects_never_signaled':True,
             'last_attempted_native_frame_by_job':{}, 'capacity_failures':[]}
    save(state)
    while state['pending']:
        raw = json.loads((BASE/'gpu_capacity_state.json').read_text())
        host = raw.get('hosts',{}).get(HOST,{})
        observed = host.get('observed_at_unix',0)
        if observed>=minimum_observation and 0<=time.time()-observed<=2100:
            for entry in state['pending'][:]:
                if state['last_attempted_native_frame_by_job'].get(entry['job'])==observed:
                    continue
                gpu = next((g for g in host.get('gpus',[]) if g['uuid']==entry['gpu_uuid']),None)
                source = next((w for w in host.get('training',[]) if w['job']==entry['job']
                               and w['pid']==entry['source_learner_pid'] and w['device']=='cpu'),None)
                if not gpu or not source or gpu.get('MPS_accounting_unproven'):
                    continue
                ceiling = 60 if gpu.get('foreign_pids') else 85
                if gpu['free_mib']<16384 or gpu['utilization_percent']>ceiling:
                    continue
                checked_local_coordinator(entry)
                state['last_attempted_native_frame_by_job'][entry['job']] = observed
                module = checked_module(entry)
                try:
                    live = module.capacity(module.helpers(entry['job']))
                except (AssertionError, OSError, ValueError, subprocess.SubprocessError) as error:
                    state['capacity_failures'].append({'job':entry['job'], 'native_frame':observed,
                                                      'error':str(error)[:500]})
                    save(state)
                    continue
                checked_local_coordinator(entry)
                os.kill(entry['pid'], signal.SIGCONT)
                state['released'].append({'job':entry['job'], 'coordinator_pid':entry['pid'],
                    'native_frame':observed, 'native_GPU':gpu, 'live_capacity':live,
                    'released_at_unix':time.time(), 'no_learner_or_foreign_signal':True})
                state['pending'].remove(entry)
                save(state)
        if state['pending']:
            time.sleep(300)
    state['phase'] = 'all_held_local_coordinators_released_training_completion_not_claimed'
    save(state)


if __name__=='__main__':
    main()
