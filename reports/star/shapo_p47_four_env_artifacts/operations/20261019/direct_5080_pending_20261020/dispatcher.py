#!/usr/bin/env python3
"""One-shot placement of an existing cell after the next ordinary observation.

Read the existing observer locally; do not add remote polling or evaluations.
Never retry a handoff, mutate another project's process, or shorten a budget.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATE = BASE / 'direct_5080_button2031_dispatch_20261020_state.json'
PLAN = BASE / 'point_button2031_5080_prepared_candidate_20261020.json'
CONTROLLER = BASE / 'handoff_button2031_4080_to_5080_20261020.py'
CONTROLLER_SHA = '36e97a390287392bf4f08916c069525bf8bc6bb3126b22b75b059c3bc0b479c8'
QUALIFICATION = BASE / 'direct_5080_button2031_capacity_qualification_20261020.json'
BASELINE = 1792434084.4766638  # Existing 02:21 ordinary observation, CST.
SOURCE_ID = {'pid': 9473, 'start_ticks': 94069,
             'boot_id': '3b362c03-4ed3-4e03-aea4-cf7339356026',
             'device': 'cuda:0'}


def save(phase, **fields):
    value = {'phase': phase, 'updated_at_unix': time.time(),
             'job': 'point-button-2031', 'source_host': '4080',
             'target_host': '5080', **fields}
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATE)
    print(json.dumps(value), flush=True)


def main():
    assert not STATE.exists(), 'Inspect the existing dispatcher; never duplicate'
    assert not PLAN.exists(), 'Inspect the existing plan; never overwrite'
    assert not (BASE / 'point-button-2031_direct_5080_20261020_state.json').exists()
    assert hashlib.sha256(CONTROLLER.read_bytes()).hexdigest() == CONTROLLER_SHA
    save('waiting_for_next_ordinary_capacity_observation',
         baseline_observed_at_unix=BASELINE, remote_polling_added=False)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        observed = json.loads((BASE / 'gpu_capacity_state.json').read_text())
        now = time.time()
        stamp = observed['observed_at_unix']
        if stamp <= BASELINE:
            time.sleep(30)
            continue
        assert 0 <= now - stamp < 120, 'New observation is stale; no mutation'
        host = observed['hosts'].get('5080', {})
        gpu = next((g for g in host.get('gpus', []) if g['index'] == 0), {})
        if (gpu.get('status') != 'available_for_rebalance'
                or gpu.get('foreign_pids')
                or gpu.get('candidate_since_unix') is None
                or stamp - gpu['candidate_since_unix'] < 1800):
            save('deferred_capacity_not_qualified_source_untouched',
                 observed_at_unix=stamp, qualification=gpu)
            return
        workers = [w for w in observed['hosts'].get('4080', {}).get('training', [])
                   if w['job'] == 'point-button-2031']
        assert len(workers) == 1
        source = workers[0]
        assert all(source.get(key) == value for key, value in SOURCE_ID.items())
        boots = {w['boot_id'] for w in host.get('training', [])}
        assert len(boots) == 1, 'Target boot not established by current observation'
        assert not PLAN.exists() and not QUALIFICATION.exists()
        QUALIFICATION.write_text(json.dumps(observed, indent=2) + '\n')
        plan = {
            'phase': 'prepared_not_started', 'job': 'point-button-2031',
            'source_host': '4080', 'target_host': '5080',
            'expected_source_identity': source,
            'expected_target_boot_id': boots.pop(),
            'total_training_budget': 10000000, 'target_physical_gpu_index': 0,
            'capacity_qualification': {**gpu, 'observed_at_unix': stamp},
            'capacity_observation_file': QUALIFICATION.name,
            'requires_fresh_capacity_check_before_source_pause': True,
            'requires_first_new_target_epoch_before_source_retirement': True,
            'scientific_parameters_changed': False, 'additional_evaluation': False,
            'source_full_checkpoint_counter_not_yet_verified': True,
            'source_commit': '62ca7399357c10645a955dcbead7ed2153a82c69',
            'controller_sha256': CONTROLLER_SHA,
        }
        PLAN.write_text(json.dumps(plan, indent=2) + '\n')
        save('qualified_invoking_single_handoff_controller',
             observed_at_unix=stamp, plan=str(PLAN),
             controller_sha256=CONTROLLER_SHA)
        result = subprocess.run([sys.executable, '-u', str(CONTROLLER)])
        save('controller_returned', controller_exit_code=result.returncode,
             controller_state_file='point-button-2031_direct_5080_20261020_state.json')
        if result.returncode:
            raise SystemExit(result.returncode)
        return
    save('observation_wait_timeout_source_untouched', remote_polling_added=False)


if __name__ == '__main__':
    main()
