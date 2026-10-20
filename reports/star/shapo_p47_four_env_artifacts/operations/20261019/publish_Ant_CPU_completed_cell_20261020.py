#!/usr/bin/env python3
"""Report a completed existing-cell CPU move without claiming cohort completion."""
import argparse
import hashlib
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/operations/20261019')
SIDE = 'cpu_5080_to_4080_20261020'


def read(path):
    return json.loads(path.read_text())


def immutable(path, value):
    assert not path.exists(), 'Never overwrite an engineering observation'
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', choices=('ant-2028', 'ant-2029'))
    parser.add_argument('stamp')
    args = parser.parse_args()
    assert re.fullmatch(r'2026102[01]_\d{4}', args.stamp)
    name = 'training_status_' + args.stamp + '_' + args.job + '_handoff_completed.json'
    assert not (OUT / name).exists()
    prefix = args.job + '_' + SIDE
    state = read(BASE / (prefix + '_state.json'))
    assert state['phase'] == 'handoff_completed_first_CPU_epoch_verified'
    assert state['source_retired'] is True and state['source_paused'] is False
    assert state['source_canonical_path_vacated'] is True
    assert state['retired_source_full_checkpoint_SHA_verified'] is True
    assert state['controller_sha256'] == hashlib.sha256((BASE / 'handoff_Ant_CPU_5080_to_4080_20261020.py').read_bytes()).hexdigest()
    native_name = state['target_first_epoch_native_observation']
    native = read(BASE / native_name)
    worker = next(w for w in native['training'] if w['job'] == args.job)
    assert worker['pid'] == state['target_identity']['pid']
    assert worker['start_ticks'] == state['target_identity']['start_ticks']
    assert worker['device'] == 'cpu' and worker['boot_id'] == state['target_identity']['boot_id']
    assert worker['steps'] >= state['first_verified_target_steps'] >= state['resume_steps'] + 5000
    transitions = {}
    for job in ('ant-2028', 'ant-2029'):
        path = BASE / (job + '_' + SIDE + '_state.json')
        if path.exists():
            transitions[job] = read(path)
    completed = [s for s in transitions.values() if s['phase'] == 'handoff_completed_first_CPU_epoch_verified']
    additions = [{'job': s['job'], 'steps': s['known_repeated_environment_steps_lower_bound'],
                  'safety_cost': s['known_additional_training_cost_lower_bound']} for s in completed]
    for s in completed:
        assert s['source_retired'] is True and s['source_paused'] is False
        assert s['last_logged_source_steps'] - s['resume_steps'] == s['known_repeated_environment_steps_lower_bound']
        assert s['last_logged_source_cumulative_training_cost'] - s['resume_cumulative_training_cost'] == s['known_additional_training_cost_lower_bound']
    other_additions = []
    for job, suffix in (('ant-2030', 'cpu_to_mps_a100_gpu0_20261020'),
                        ('walker2d-2030', 'cpu_to_mps_a100_gpu1_20261020')):
        path = BASE / (job + '_' + suffix + '_state.json')
        if not path.exists():
            continue
        s = read(path)
        if s['phase'] != 'completed_first_MPS_epoch_source_retired':
            continue  # An unfinished move never contributes completed recovery work.
        assert s['source_retired'] is True and s['source_paused'] is False
        steps = s['last_logged_source_steps'] - s['resume_steps']
        cost = s.get('last_logged_source_cumulative_training_cost', s.get('last_logged_source_cost')) - s['resume_cumulative_training_cost']
        assert steps == s.get('known_repeated_environment_steps_lower_bound', s.get('known_repeated_steps_lower_bound'))
        assert cost == s.get('known_additional_training_cost_lower_bound', s.get('known_extra_training_cost_lower_bound'))
        other_additions.append({'job': job, 'state_file': path.name, 'steps': steps, 'safety_cost': cost})
    root = read(OUT / 'training_status.json')
    previous = read(OUT / root['superseded_by'])
    snapshot_name = prefix + '_completed_state_' + args.stamp + '.json'
    immutable(OUT / snapshot_name, state)
    native_snapshot = prefix + '_first_epoch_native_' + args.stamp + '.json'
    immutable(OUT / native_snapshot, native)
    report = {
        'schema_version': 1, 'kind': 'completed_existing_cell_CPU_handoff_not_full_cohort_snapshot',
        'engineering_completed_at_unix': state['updated_at_unix'], 'previous_status': root['superseded_by'],
        'latest_all_host_native_frame': previous.get('native_observation_file', previous.get('latest_all_host_native_frame')),
        'completed_state': snapshot_name, 'first_target_native_observation': native_snapshot,
        'job': args.job, 'target_identity': state['target_identity'],
        'genuine_resume_steps': state['resume_steps'], 'first_target_steps': state['first_verified_target_steps'],
        'first_target_FPS': state['first_verified_target_fps'],
        'source_previous_epoch_FPS': state['expected_source_identity']['last_epoch_fps'],
        'first_epoch_only_speed_ratio_not_sustained_or_causal_guarantee': state['first_verified_target_fps'] / state['expected_source_identity']['last_epoch_fps'],
        'runtime_change': state['runtime_change'], 'GPU_capacity_not_claimed_or_used': True,
        'source_original_directory_recoverably_archived': state['retired_source_archive'],
        'input_preservation': state['input_preservation'],
        'completed_Ant_CPU_recovery_additions': additions,
        'subsequent_completed_A100_runtime_recovery_additions': other_additions,
        'completed_recovery_totals_lower_bound': {
            'environment_steps': 1142276 + sum(a['steps'] for a in additions + other_additions),
            'safety_cost': 10370 + sum(a['safety_cost'] for a in additions + other_additions),
            'unknown_historical_and_unlogged_work_not_assumed_zero': True},
        'other_Ant_CPU_state_phases_at_report': {j: s['phase'] for j, s in transitions.items() if j != args.job},
        'full_CPU_group_accounting_not_claimed_by_this_single_cell_report': True,
        'all_host_counters_not_spliced_with_later_target_probe': True,
        'additional_training_cells': 0, 'additional_scientific_evaluation_rollouts': 0,
        'scientific_parameters_changed': False, 'training_budgets_reduced': False,
        'other_projects_paused': False, 'full_final_scientific_audit_executed': False,
        'final_scientific_report': False,
        'data_notice': 'Existing Ant cell continued from a genuine full checkpoint with replay/RNG/optimizers. A new real5k CPU epoch and native process identity preceded source retirement and recoverable archival. This is engineering evidence only, not a simultaneous all-host frame, sustained speed guarantee, cohort accounting pass, or final scientific certification. All150M budgets and final provenance gates remain; legacy2027 missing original full parents are not repaired.'}
    immutable(OUT / name, report)
    root.update(superseded_by=name, latest_completed_engineering_transition=snapshot_name,
                snapshot_notice='Historical17:27 payload below. Latest completed existing-cell CPU placement:' + name + '. ' + report['data_notice'])
    root.pop('latest_inflight_engineering_transition', None)
    (OUT / 'training_status.json').write_text(json.dumps(root, indent=2) + '\n')
    print(json.dumps({'status': name, 'job': args.job, 'first_target_FPS': report['first_target_FPS']}))


if __name__ == '__main__':
    main()
