#!/usr/bin/env python3
"""Publish a new completed handoff, never splice its later proof into an old frame."""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from cpu_to_mps_walkers_a100_recovery_accounting_20261020 import verify_A100_Walker_CPU_to_MPS_transitions

BASE = Path(__file__).resolve().parent
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/operations/20261019')
PREFIX = 'walker2d-2030_cpu_to_mps_a100_gpu1_20261020'
CHECK_NAME = 'A100_Walker_2030_2031_completed_accounting_check_20261021.json'


def read(path):
    return json.loads(path.read_text())


def immutable(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def main():
    state = read(BASE / (PREFIX + '_state.json'))
    assert state['phase'] == 'completed_first_MPS_epoch_source_retired'
    assert state['source_retired'] is True and state['source_paused'] is False
    assert not (BASE / CHECK_NAME).exists() and not (OUT / CHECK_NAME).exists(), 'Inspect original publication; do not repeat'
    for name in ('ant-2030_cpu_to_mps_a100_gpu0_20261020_state.json',
                 'walker2d-2029_walker_cpu_5080_to_4080_20261021_state.json'):
        assert not (BASE / name).exists(), 'New concurrent transaction needs a separate evidence-aware report'
    stamp = datetime.fromtimestamp(state['updated_at_unix'], ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
    name = 'training_status_' + stamp + '_Walker2030_A100_MPS_completed.json'
    snapshot = PREFIX + '_completed_state_' + stamp + '.json'
    native_name = PREFIX + '_target_first_epoch_native_' + stamp + '.json'
    assert not any((OUT / n).exists() for n in (name, snapshot, native_name))
    # One engineering accounting pass for this genuinely new second handoff.
    # This does not run the frozen full-cohort scientific validator or rollouts.
    records = verify_A100_Walker_CPU_to_MPS_transitions(BASE)
    assert len(records) == 2 and {r['job'] for r in records} == {'walker2d-2030', 'walker2d-2031'}
    record = next(r for r in records if r['job'] == 'walker2d-2030')
    native = record['first_target_native_observation']
    first = state['first_target_epoch']
    frame_name = 'ordinary_observation_20261021_0321.json'
    frame = read(OUT / frame_name)
    sources = [w for v in frame['hosts'].values() for w in v.get('training', [])
               if w['job'] == 'walker2d-2030' and w['pid'] == state['source_identity']['pid']]
    assert len(sources) == 1
    source = sources[0]
    assert source['start_ticks'] == state['source_identity']['start_ticks']
    assert source['boot_id'] == state['source_identity']['boot_id']
    assert source['steps'] == state['last_logged_source_steps']
    baseline = read(OUT / 'training_status_20261021_0251.json')
    old = baseline['known_recovery_accounting']
    steps, cost = record['known_additional_physical_steps'], record['known_additional_physical_training_cost']
    check = {
        'kind': 'completed_existing_A100_Walker_runtime_accounting_not_final_scientific_audit',
        'checked_at_unix': time.time(), 'completed_records_checked': len(records),
        'records': [{'job': r['job'], 'extra_steps_lower_bound': r['known_additional_physical_steps'],
                     'extra_cost_lower_bound': r['known_additional_physical_training_cost']} for r in records],
        'accounting_module_sha256': hashlib.sha256((BASE / 'cpu_to_mps_walkers_a100_recovery_accounting_20261020.py').read_bytes()).hexdigest(),
        'additional_evaluation_rollouts': 0, 'full_final_scientific_audit': False,
        'strict_final_matrix_requirements_waived': False}
    root = read(OUT / 'training_status.json')
    report = {
        'kind': 'completed_existing_Walker2030_CPU_to_GPU_MPS_not_full_cohort_snapshot',
        'completed_at_unix': state['updated_at_unix'], 'published_at_unix': time.time(),
        'previous_status': root['superseded_by'], 'job': state['job'],
        'last_all_host_native_frame_before_this_handoff': frame_name,
        'all_host_native_counters_not_spliced_with_later_single_host_probe': True,
        'completed_state': snapshot, 'first_target_native_observation': native_name,
        'engineering_accounting_check': CHECK_NAME,
        'genuine_resume_steps': state['resume_steps'], 'genuine_full_input_sha256': state['checkpoint_sha256'],
        'first_target_epoch': first, 'target_identity': state['target_identity'],
        'MPS_connection_proof': state['MPS_connection_proof'],
        'runtime_change': {'from': 'CPU', 'to': 'A100_GPU1_private_MPS',
                           'bitwise_trajectory_equivalence_claimed': False},
        'source_previous_epoch_FPS': source['last_epoch_fps'],
        'source_FPS_reference': frame_name,
        'first_epoch_only_speed_ratio_not_sustained_or_causal_guarantee': first['fps'] / source['last_epoch_fps'],
        'source_retirement': state['wrapper_retirement'],
        'input_preservation': state['input_preservation'],
        'original_segments_preserved_at_original_paths': state['original_training_segments_preserved_at_original_paths'],
        'new_completed_recovery_addition': {'environment_steps': steps, 'safety_cost': cost},
        'completed_recovery_totals_lower_bound': {
            'baseline': 'training_status_20261021_0251.json',
            'environment_steps': old['extra_environment_steps_lower_bound'] + steps,
            'safety_cost': old['extra_training_safety_cost_lower_bound'] + cost,
            'unknown_historical_and_unlogged_work_not_assumed_zero': True},
        'full_training_budget_steps': 5000000, 'cohort_budget_steps': 150000000,
        'new_experiment_cells': 0, 'additional_scientific_evaluation_rollouts': 0,
        'scientific_parameters_changed': False, 'foreign_projects_modified': False,
        'full_final_scientific_audit_executed': False, 'final_scientific_report': False,
        'legacy_provenance': baseline['legacy_provenance'],
        'final_scientific_requirements_unchanged': baseline['final_scientific_requirements_unchanged'],
        'data_notice': 'Walker2030 completed an existing-cell CPU-to-A100GPU1 handoff from the genuine4.35M full replay/RNG/optimizer input. A real new5k epoch and verified private-MPS GPU/client identity preceded original source retirement; original full input and training segments remain at their original paths, with host and separate4T preservation. This report is not a simultaneous all-host census or sustained speed guarantee. All150M nominal budgets and strict final provenance gates remain; early2027 original full-parent gaps are not repaired.'}
    immutable(BASE / CHECK_NAME, check)
    for filename, payload in ((CHECK_NAME, check), (snapshot, state), (native_name, native), (name, report)):
        immutable(OUT / filename, payload)
    root.update(superseded_by=name, latest_completed_engineering_transition=snapshot,
                snapshot_notice='Historical17:27 payload below. Latest completed engineering report:' + name + '. ' + report['data_notice'])
    root.pop('latest_inflight_engineering_transition', None)
    temporary = OUT / 'training_status.Walker2030_completed.tmp'
    with temporary.open('x') as stream:
        stream.write(json.dumps(root, indent=2) + '\n')
    temporary.replace(OUT / 'training_status.json')
    print(json.dumps({'status': name, 'first_target_FPS': first['fps'],
                      'source_previous_epoch_FPS': source['last_epoch_fps'],
                      'new_accounting_records_checked': len(records),
                      'recovery_lower_bound': report['completed_recovery_totals_lower_bound']}))


if __name__ == '__main__':
    main()
