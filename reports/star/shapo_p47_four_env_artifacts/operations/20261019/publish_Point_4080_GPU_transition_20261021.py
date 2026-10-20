#!/usr/bin/env python3
"""Publish an existing Point placement observation; never execute ML or SSH."""
import argparse
import hashlib
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/operations/20261019')
SIDE = 'point_5080_to_4080_MPS_20261021'


def read(path):
    return json.loads(path.read_text())


def immutable(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    parser.add_argument('stamp')
    parser.add_argument('--native-frame', default='ordinary_observation_20261021_0551.json')
    args = parser.parse_args()
    assert re.fullmatch(r'2026102[1-9]_\d{4}', args.stamp)
    assert args.job in {slug + '-' + str(seed) for slug in ('point-goal', 'point-button') for seed in range(2027, 2032)}
    assert re.fullmatch(r'ordinary_observation_\d{8}_\d{4}\.json', args.native_frame)
    read(OUT / args.native_frame)
    state = read(BASE / (args.job + '_' + SIDE + '_state.json'))
    assert state['source_commit'] == '62ca7399357c10645a955dcbead7ed2153a82c69'
    assert state['total_training_budget'] == 10000000 and state['additional_experiment_cells'] == 0
    assert state['scientific_parameters_changed'] is False and state['additional_evaluation'] is False
    assert state['controller_sha256'] == hashlib.sha256((BASE / 'handoff_Point_5080_to_4080_MPS_20261021.py').read_bytes()).hexdigest()
    completed = state['phase'] == 'completed_first_MPS_epoch_source_retired'
    label = 'completed' if completed else 'inflight'
    name = 'training_status_' + args.stamp + '_' + args.job + '_4080_GPU_' + label + '.json'
    assert not (OUT / name).exists(), 'Never overwrite a transition observation'
    snapshot = args.job + '_' + SIDE + '_' + label + '_state_' + args.stamp + '.json'
    immutable(OUT / snapshot, state)
    native_name = None
    if completed:
        assert state['source_retired'] is True and state['source_paused'] is False
        assert state['source_canonical_path_vacated'] is True
        assert state['retired_source_full_checkpoint_SHA_verified'] is True
        assert state['first_verified_target_steps'] >= state['resume_steps'] + 5000
        native = read(BASE / state['target_first_epoch_native_observation'])
        worker = next(w for w in native['training'] if w['job'] == args.job)
        assert worker['pid'] == state['target_identity']['pid']
        assert worker['start_ticks'] == state['target_identity']['start_ticks']
        assert worker['mps_gpu_uuid_verified'] == 'GPU-66ed7e12-a0c3-9273-463d-56a5e5f0a4f7'
        native_name = args.job + '_' + SIDE + '_first_epoch_native_' + args.stamp + '.json'
        immutable(OUT / native_name, native)
    all_states = [read(p) for p in sorted(BASE.glob('point-*_' + SIDE + '_state.json'))]
    done = [s for s in all_states if s['phase'] == 'completed_first_MPS_epoch_source_retired']
    pending = [{'job': s['job'], 'phase': s['phase']} for s in all_states if s not in done]
    additions = [{'job': s['job'], 'steps': s['known_repeated_environment_steps_lower_bound'],
                  'safety_cost': s['known_additional_training_cost_lower_bound']} for s in done]
    queue_name = 'snapshot_' + args.stamp + '_' + args.job + '_' + label + '_Point_GPU_dispatch.json'
    queue = read(BASE / 'Point_5080_to_4080_MPS_regular_dispatch_20261021_state.json')
    immutable(OUT / queue_name, queue)
    root = read(OUT / 'training_status.json')
    baseline = read(OUT / 'training_status_20261021_0551.json')['known_recovery_accounting']
    report = {
        'kind': 'existing_Point_GPU_placement_observation_not_full_cohort_census',
        'job': args.job, 'phase': state['phase'], 'engineering_observed_at_unix': state['updated_at_unix'],
        'completed_existing_cell_placement': completed, 'previous_status': root['superseded_by'],
        'latest_all_host_native_frame': args.native_frame, 'state_snapshot': snapshot,
        'dispatch_snapshot': queue_name, 'first_target_native_observation': native_name,
        'source_host': '5080', 'target_host': '4080', 'runtime_change': state['runtime_change'],
        'first_target_epoch_not_claimed_when_inflight': not completed,
        'source_paused': state.get('source_paused', False), 'source_retired': state.get('source_retired', False),
        'genuine_resume_steps': state.get('resume_steps'), 'input_preservation': state.get('input_preservation'),
        'first_target_FPS': state.get('first_verified_target_fps') if completed else None,
        'source_previous_native_epoch_FPS': state['plan']['expected_source_identity']['last_epoch_fps'],
        'FPS_is_single_epoch_not_sustained_or_causal_guarantee': True,
        'previous_target_container_preserved_at': state.get('previous_target_container_preserved_at'),
        'completed_new_Point_recovery_additions': additions,
        'completed_recovery_totals_lower_bound': {
            'environment_steps': baseline['extra_environment_steps_lower_bound'] + sum(a['steps'] for a in additions),
            'safety_cost': baseline['extra_training_safety_cost_lower_bound'] + sum(a['safety_cost'] for a in additions),
            'unlogged_and_historical_unknown_work_not_zero': True},
        'unfinished_Point_transactions_not_in_completed_floors': pending,
        'new_whole_group_accounting_check_not_claimed_by_this_report': True,
        'scientific_parameters_changed': False, 'training_budget_each': 10000000,
        'additional_training_cells_or_evaluation_rollouts': 0, 'other_projects_modified': False,
        'all_host_counters_not_spliced_with_later_single_host_observations': True,
        'full_final_scientific_audit_executed': False, 'original_2027_parent_gaps_not_repaired': True,
        'data_notice': 'Engineering continuation of an existing full-budget Point cell, not a new seed or scientific result. Inflight source/target work must not be counted as a completed handoff; target first5k epoch and MPS identity precede source retirement. Native all-host frame remains a separate historical census. Final20-cell scientific/provenance gates are unchanged.'}
    immutable(OUT / name, report)
    root.update(superseded_by=name, snapshot_notice='Historical17:27 payload below; latest:' + name + '. ' + report['data_notice'])
    if completed:
        root['latest_completed_engineering_transition'] = snapshot
        if not pending:root.pop('latest_inflight_engineering_transition', None)
    else:
        root['latest_inflight_engineering_transition'] = snapshot
    (OUT / 'training_status.json').write_text(json.dumps(root, indent=2) + '\n')
    print(json.dumps({'status': name, 'phase': state['phase'], 'completed': completed}))


if __name__ == '__main__':
    main()
