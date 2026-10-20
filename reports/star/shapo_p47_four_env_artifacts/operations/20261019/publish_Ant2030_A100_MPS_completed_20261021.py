#!/usr/bin/env python3
"""Publish one completed Ant GPU handoff without rerunning old Walker checks."""
import hashlib
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from cpu_to_mps_a100_recovery_accounting_20261020 import verify_A100_CPU_to_MPS_transitions
from publish_Walker2030_A100_MPS_inflight_20261021_0322 import BASE, OUT, immutable, read

PREFIX = 'ant-2030_cpu_to_mps_a100_gpu0_20261020'
CHECK_NAME = 'A100_Ant2030_completed_accounting_check_20261021.json'


def main():
    s = read(BASE / (PREFIX + '_state.json'))
    assert s['phase'] == 'completed_first_MPS_epoch_source_retired'
    assert s['source_retired'] is True and s['source_paused'] is False
    assert not (BASE / CHECK_NAME).exists() and not (OUT / CHECK_NAME).exists(), 'Inspect original report; never replay'
    assert not (BASE / 'walker2d-2029_walker_cpu_5080_to_4080_20261021_state.json').exists(), 'Concurrent new move needs separate accounting'
    stamp = datetime.fromtimestamp(s['updated_at_unix'], ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
    name = 'training_status_' + stamp + '_Ant2030_A100_MPS_completed.json'
    snapshot = PREFIX + '_completed_state_' + stamp + '.json'
    native_name = PREFIX + '_target_first_epoch_native_' + stamp + '.json'
    assert not any((OUT / n).exists() for n in (name, snapshot, native_name))
    # Exactly one accounting call for this new Ant event; no Walker re-audit.
    records = verify_A100_CPU_to_MPS_transitions(BASE)
    assert len(records) == 1 and records[0]['job'] == 'ant-2030'
    r = records[0]
    native = r['first_target_native_observation']
    frame_name = 'ordinary_observation_20261021_0351.json'
    frame = read(OUT / frame_name)
    sources = [w for v in frame['hosts'].values() for w in v.get('training', [])
               if w['job'] == 'ant-2030' and w['pid'] == s['source_identity']['pid']]
    assert len(sources) == 1
    source = sources[0]
    assert source['start_ticks'] == s['source_identity']['start_ticks']
    assert source['boot_id'] == s['source_identity']['boot_id']
    assert source['steps'] == s['last_logged_source_steps']
    baseline = read(OUT / 'training_status_20261021_0351.json')
    old = baseline['known_recovery_accounting']
    steps, cost = r['known_additional_physical_steps'], r['known_additional_physical_training_cost']
    first = s['first_target_epoch']
    check = {
        'kind': 'new_Ant_runtime_accounting_not_final_scientific_validation',
        'checked_at_unix': time.time(), 'completed_records_checked': 1,
        'job': s['job'], 'extra_steps_lower_bound': steps, 'extra_cost_lower_bound': cost,
        'unknown_unlogged_source_tail_not_zero': True,
        'accounting_module_sha256': hashlib.sha256((BASE / 'cpu_to_mps_a100_recovery_accounting_20261020.py').read_bytes()).hexdigest(),
        'existing_Walker_check_not_repeated': 'A100_Walker_2030_2031_completed_accounting_check_20261021.json',
        'additional_scientific_rollouts': 0, 'full_final_scientific_audit': False,
        'strict_final_matrix_requirements_waived': False}
    root = read(OUT / 'training_status.json')
    report = {
        'kind': 'completed_existing_Ant2030_CPU_to_GPU0_MPS_not_full_cohort_census',
        'completed_at_unix': s['updated_at_unix'], 'published_at_unix': time.time(),
        'previous_status': root['superseded_by'], 'job': s['job'],
        'last_all_host_native_frame_before_this_handoff': frame_name,
        'old_all_host_counters_not_spliced_with_later_single_host_probe': True,
        'completed_state': snapshot, 'first_target_native_observation': native_name,
        'engineering_accounting_check': CHECK_NAME,
        'genuine_resume_steps': s['resume_steps'], 'genuine_full_input_sha256': s['checkpoint_sha256'],
        'first_target_epoch': first, 'target_identity': s['target_identity'],
        'MPS_connection_proof': s['MPS_connection_proof'],
        'runtime_change': {'from': 'CPU', 'to': 'A100_GPU0_private_MPS',
                           'bitwise_trajectory_equivalence_claimed': False},
        'source_previous_epoch_FPS': source['last_epoch_fps'], 'source_FPS_reference': frame_name,
        'first_epoch_only_speed_ratio_not_sustained_or_causal_guarantee': first['fps'] / source['last_epoch_fps'],
        'source_retirement': s['wrapper_retirement'], 'input_preservation': s['input_preservation'],
        'original_segments_preserved_at_original_paths': s['original_training_segments_preserved_at_original_paths'],
        'new_completed_recovery_addition': {'environment_steps': steps, 'safety_cost': cost},
        'completed_recovery_totals_lower_bound': {
            'baseline': 'training_status_20261021_0351.json',
            'environment_steps': old['extra_environment_steps_lower_bound'] + steps,
            'safety_cost': old['extra_training_safety_cost_lower_bound'] + cost,
            'unknown_historical_and_unlogged_work_not_assumed_zero': True},
        'full_training_budget_steps': 5000000, 'cohort_budget_steps': 150000000,
        'new_experiment_cells': 0, 'additional_scientific_evaluation_rollouts': 0,
        'scientific_parameters_changed': False, 'foreign_projects_modified': False,
        'full_final_scientific_audit_executed': False, 'final_scientific_report': False,
        'legacy_provenance': baseline['legacy_provenance'],
        'final_scientific_requirements_unchanged': baseline['final_scientific_requirements_unchanged'],
        'data_notice': 'Ant2030 continued from its genuine4.35M full replay/RNG/optimizer input. A real new5k GPU epoch and actual private-MPS GPU0/client proof preceded source retirement; original full and training segments remain at original paths, with sameSHA host and separate4T preservation. Known logged rollback is0, but unknown source tail is not assumed zero. FPS describes the first target epoch only. This is not a new simultaneous all-host census or a final scientific audit. All150M budgets and provenance gates remain; early2027 missing original parents are unresolved.'}
    immutable(BASE / CHECK_NAME, check)
    for filename, payload in ((CHECK_NAME, check), (snapshot, s), (native_name, native), (name, report)):
        immutable(OUT / filename, payload)
    root.update(superseded_by=name, latest_completed_engineering_transition=snapshot,
                snapshot_notice='Historical17:27 payload below. Latest completed engineering report:' + name + '. ' + report['data_notice'])
    root.pop('latest_inflight_engineering_transition', None)
    temporary = OUT / 'training_status.Ant2030_completed.tmp'
    with temporary.open('x') as stream:
        stream.write(json.dumps(root, indent=2) + '\n')
    temporary.replace(OUT / 'training_status.json')
    print(json.dumps({'status': name, 'first_target_FPS': first['fps'],
                      'source_previous_epoch_FPS': source['last_epoch_fps'],
                      'new_accounting_records_checked': 1,
                      'recovery_lower_bound': report['completed_recovery_totals_lower_bound']}))


if __name__ == '__main__':
    main()
