#!/usr/bin/env python3
"""Publish the existing 22:51 frame and subsequent Ant CPU move, not completion."""
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/operations/20261019')


def read(path):
    return json.loads(path.read_text())


def immutable(path, value):
    assert not path.exists(), 'Never overwrite historical transition snapshots'
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    stamp = '20261020_2251'
    name = 'training_status_' + stamp + '_inflight.json'
    assert not (OUT / name).exists()
    raw = read(BASE / ('capacity_progress_observation_' + stamp + '.json'))
    previous = read(OUT / 'training_status_20261020_2221.json')
    old = {w['job']: w for w in previous['active_training']}
    workers = [dict(w, host=h) for h, d in raw['hosts'].items() for w in d.get('training', [])]
    assert all(d['status'] == 'observed' for d in raw['hosts'].values())
    assert len(workers) == len({w['job'] for w in workers}) == 18
    assert set(old) == {w['job'] for w in workers}
    assert all(w['steps'] > old[w['job']]['steps'] for w in workers)
    prefix = 'ant-2028_cpu_5080_to_4080_20261020'
    state = read(BASE / (prefix + '_state.json'))
    assert state['phase'] in ('preparing_CPU_to_CPU_handoff', 'copying_paused_CPU_source',
                              'target_CPU_starting_source_CPU_paused')
    assert state['source_retired'] is False
    assert state['resume_steps'] == 2100000
    state_name = prefix + '_inflight_state_' + stamp + '.json'
    immutable(OUT / state_name, state)
    dispatch = 'snapshot_' + stamp + '_Ant_CPU_5080_to_4080_regular_dispatch_20261020_state.json'
    immutable(OUT / dispatch, read(BASE / 'Ant_CPU_5080_to_4080_regular_dispatch_20261020_state.json'))
    native_name = 'ordinary_observation_' + stamp + '.json'
    immutable(OUT / native_name, raw)
    root = read(OUT / 'training_status.json')
    report = {
        'schema_version': 1, 'kind': 'native_progress_frame_with_later_inflight_existing_cell_migration',
        'previous_status': 'training_status_20261020_2221.json',
        'observed_at_unix': raw['observed_at_unix'], 'native_observation_file': native_name,
        'native_workers_before_handoff': workers,
        'native_unique_active_cells_before_handoff': 18,
        'all_18_native_logged_counters_advanced_since_2221': True,
        'native_logged_counter_total_including_two_5M_budget_complete_cells': 10000000 + sum(w['steps'] for w in workers),
        'native_counter_total_is_not_current_inflight_accepted_lineage_or_physical_compute_total': True,
        'training_matrix': previous['training_matrix'], 'training_budgets_complete': 2,
        'total_training_cells': 20, 'engineering_observed_at_unix': time.time(),
        'inflight_existing_cell': state['job'], 'inflight_state_snapshot': state_name,
        'inflight_phase_at_engineering_observation': state['phase'],
        'source_controlled_pause_recorded': state['source_paused'],
        'source_original_kept_in_memory_until_target_first_real_5k_epoch': True,
        'genuine_resume_steps': state['resume_steps'], 'runtime_change': state['runtime_change'],
        'GPU_capacity_not_claimed_or_used_by_CPU_move': True,
        'target_first_epoch_not_yet_proven_by_this_snapshot': True,
        'conditional_source_tail_if_handoff_completes': {
            'logged_steps': state['known_repeated_environment_steps_lower_bound'],
            'logged_safety_cost': state['known_additional_training_cost_lower_bound'],
            'not_added_to_completed_recovery_totals_until_target_epoch_and_retirement_proof': True,
            'unknown_unlogged_work_not_assumed_zero': True},
        'completed_recovery_accounting_before_this_inflight_move': previous['known_recovery_accounting'],
        'capacity_dispatch_state_snapshot': dispatch,
        'storage_at_native_frame': {h: d['storage'] for h, d in raw['hosts'].items()},
        'A100_project_low_space_warning_free_GiB': 19.16,
        'separate_4080_4T_free_GiB': 2898.41,
        'latest_published_native_science_summary_reference': root.get('newest_intermediate_scientific_result'),
        'additional_evaluation_rollouts': 0, 'new_training_cells': 0,
        'scientific_parameters_changed': False, 'training_budgets_reduced': False,
        'other_projects_paused': False, 'full_scientific_audit_executed': False,
        'final_scientific_report': False,
        'final_scientific_requirements_unchanged': previous['final_scientific_requirements_unchanged'],
        'legacy_provenance': previous['legacy_provenance'],
        'next_ordinary_observation_local': '2026-10-20T23:21:49+08:00',
        'data_notice': 'The native22:51 frame showed18 unique existing cells advancing before the later controlled Ant2028 CPU move. Ant2028 resumes a genuine2.1M full checkpoint; its source remains held in memory until a real new5k target epoch. This partial snapshot does not prove target progress or source retirement. Ant2029 has not moved. CPU-build cu128-to-cu118 transition is disclosed; no bitwise equivalence is claimed. No foreign GPU job was paused, no extra rollout or seed was added, and all150M steps and strict final provenance gates remain. Known completed physical recovery floors remain1142276steps/10370cost plus unknown historical/unlogged work; the pending Ant transition is not credited as completed. Earlier2027 full-parent gaps remain unresolved.'
    }
    immutable(OUT / name, report)
    root.update(superseded_by=name, latest_inflight_engineering_transition=state_name,
                snapshot_notice='Historical17:27 payload below. Latest native frame and subsequent in-flight move:' + name + '. ' + report['data_notice'])
    (OUT / 'training_status.json').write_text(json.dumps(root, indent=2) + '\n')
    print(json.dumps({'status': name, 'phase': state['phase'],
                      'native_logged_total_before_handoff': report['native_logged_counter_total_including_two_5M_budget_complete_cells']}))


if __name__ == '__main__':
    main()
