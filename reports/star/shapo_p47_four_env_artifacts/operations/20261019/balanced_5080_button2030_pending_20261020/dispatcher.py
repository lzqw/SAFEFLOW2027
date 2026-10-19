#!/usr/bin/env python3
"""One existing-cell balance after the next ordinary sample, never a new seed."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATE = BASE / 'balanced_5080_button2030_dispatch_20261020_state.json'
PLAN = BASE / 'point_button2030_5080_balanced_20261020_0421.json'
CONTROLLER = BASE / 'rebalance_button2030_4080_to_5080_20261020_0421.py'
CONTROLLER_SHA = '1d2df07e784601722fda533a48843c73727c4ae255bdf4e16857e018758bc23a'
BASELINE = 1792439486.9336076
EXPECTED_TARGET_PIDS = [28355, 31275, 51246, 135035]
SOURCE_ID = {'pid': 9472, 'start_ticks': 94069,
             'boot_id': '3b362c03-4ed3-4e03-aea4-cf7339356026', 'device': 'cuda:0'}


def save(phase, **fields):
    value = {'phase': phase, 'job': 'point-button-2030',
             'updated_at_unix': time.time(), **fields}
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATE)
    print(json.dumps(value), flush=True)


def owned_gpu_pids(host):
    gpu = next(g for g in host.get('gpus', []) if g['index'] == 0)
    if gpu.get('foreign_pids') or gpu['free_mib'] < 8192:
        return []
    return sorted(p['pid'] for p in gpu['processes'] if p['role'] == 'safeflow')


def main():
    assert not STATE.exists() and not PLAN.exists(), 'Inspect prior attempt; never duplicate'
    assert not (BASE / 'point-button-2030_balanced_5080_20261020_0421_state.json').exists()
    assert hashlib.sha256(CONTROLLER.read_bytes()).hexdigest() == CONTROLLER_SHA
    previous_file = 'capacity_progress_observation_20261020_0351.json'
    previous = json.loads((BASE / previous_file).read_text())
    assert previous['observed_at_unix'] == BASELINE
    assert owned_gpu_pids(previous['hosts']['5080']) == EXPECTED_TARGET_PIDS
    save('waiting_for_next_ordinary_observation', baseline_observed_at_unix=BASELINE,
         remote_polling_added=False, qualification='own_cohort_balance_not_spare_compute')
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        observed = json.loads((BASE / 'gpu_capacity_state.json').read_text())
        stamp = observed['observed_at_unix']
        if stamp <= BASELINE:
            time.sleep(30)
            continue
        assert 0 <= time.time() - stamp < 120, 'New observation stale; no mutation'
        host = observed['hosts'].get('5080', {})
        sources = [w for w in observed['hosts'].get('4080', {}).get('training', [])
                   if w['job'] == 'point-button-2030']
        targets = [w for w in host.get('training', []) if w['device'] == 'cuda:0']
        source = sources[0] if len(sources) == 1 else {}
        boots = {w['boot_id'] for w in targets}
        previous_boots = {w['boot_id'] for w in previous['hosts']['5080']['training']
                          if w['device'] == 'cuda:0'}
        qualified = (stamp - BASELINE >= 1800
                     and owned_gpu_pids(host) == EXPECTED_TARGET_PIDS
                     and len(targets) == 4 and len(boots) == 1 and boots == previous_boots
                     and all(source.get(k) == v for k, v in SOURCE_ID.items())
                     and (source.get('last_epoch_fps') or 0) > 0
                     and all(w.get('last_epoch_fps') is not None for w in targets))
        if qualified:
            qualified = min(w['last_epoch_fps'] for w in targets) >= 4 * source['last_epoch_fps']
        if not qualified:
            save('deferred_qualification_changed_source_untouched', observed_at_unix=stamp)
            return
        current_file = 'capacity_progress_observation_20261020_button2030_balance_0421.json'
        assert not (BASE / current_file).exists()
        (BASE / current_file).write_text(json.dumps(observed, indent=2) + '\n')
        gpu = next(g for g in host['gpus'] if g['index'] == 0)
        qualification = {
            'kind': 'existing_cohort_makespan_balance', 'observed_at_unix': stamp,
            'previous_observed_at_unix': BASELINE,
            'previous_observation_file': previous_file, 'current_observation_file': current_file,
            'uuid': gpu['uuid'], 'owned_gpu_pids': EXPECTED_TARGET_PIDS,
            'foreign_pids': [], 'source_last_epoch_fps': source['last_epoch_fps'],
            'target_minimum_last_epoch_fps': min(w['last_epoch_fps'] for w in targets),
            'ordinary_monitor_status_unchanged': gpu['status'],
            'ordinary_monitor_utilization_percent': gpu['utilization_percent'],
            'free_mib': gpu['free_mib'], 'spare_compute_claimed': False,
            'expected_concurrency_after_transfer': 5,
        }
        plan = {
            'phase': 'prepared_not_started', 'job': 'point-button-2030',
            'source_host': '4080', 'target_host': '5080',
            'expected_source_identity': source, 'expected_target_boot_id': boots.pop(),
            'total_training_budget': 10000000, 'target_physical_gpu_index': 0,
            'rebalance_qualification': qualification,
            'requires_fresh_capacity_check_before_source_pause': True,
            'requires_first_new_target_epoch_before_source_retirement': True,
            'scientific_parameters_changed': False, 'additional_evaluation': False,
            'additional_experiment_cells': 0, 'source_full_checkpoint_counter_not_yet_verified': True,
            'source_commit': '62ca7399357c10645a955dcbead7ed2153a82c69',
            'controller_sha256': CONTROLLER_SHA,
            'placement_policy': 'Balance own existing cohort; not a spare-compute claim.',
            'other_owned_target_cells_may_slow_down': True,
            'not_a_guaranteed_throughput_or_completion_time_improvement': True,
            'target_must_have_no_foreign_gpu_processes': True,
        }
        PLAN.write_text(json.dumps(plan, indent=2) + '\n')
        save('invoking_single_controller', plan=str(PLAN), observed_at_unix=stamp)
        result = subprocess.run([sys.executable, '-u', str(CONTROLLER)])
        save('controller_returned', controller_exit_code=result.returncode)
        raise SystemExit(result.returncode)
    save('ordinary_observation_wait_timeout_source_untouched')


if __name__ == '__main__':
    main()
