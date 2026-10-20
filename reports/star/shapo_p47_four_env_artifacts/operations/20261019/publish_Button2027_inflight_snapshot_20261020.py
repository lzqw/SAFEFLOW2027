#!/usr/bin/env python3
"""Record a native frame plus a later in-flight move, never as steady completion."""
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/operations/20261019')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    assert not path.exists(), 'Immutable in-flight snapshot already exists; do not overwrite'
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    prefix = 'point-button-2027_balanced_MPS_5080_20261020'
    s = read(BASE / (prefix + '_state.json'))
    assert s['phase'] in ('preparing', 'copying_paused_source', 'target_starting_source_paused')
    raw = read(BASE / 'capacity_progress_observation_20261020_2151.json')
    previous = read(OUT / 'training_status_20261020_2121.json')
    old = {w['job']: w for w in previous['active_training']}
    workers = [dict(w, host=h) for h, d in raw['hosts'].items() for w in d.get('training', [])]
    assert len(workers) == len({w['job'] for w in workers}) == 18
    assert all(w['steps'] > old[w['job']]['steps'] for w in workers)
    state_name = prefix + '_inflight_state_20261020_2151.json'
    write(OUT / state_name, s)
    root = read(OUT / 'training_status.json')
    report = {'schema_version': 1, 'kind': 'native_progress_frame_with_later_inflight_existing_cell_migration',
              'previous_status': 'training_status_20261020_2121.json',
              'observed_at_unix': raw['observed_at_unix'],
              'native_observation_file': 'ordinary_observation_20261020_2151.json',
              'native_workers_before_handoff': workers,
              'native_unique_active_cells_before_handoff': 18,
              'all_18_native_logged_counters_advanced_since_2121': True,
              'native_logged_counter_total_including_two_5M_budget_complete_cells': 10000000 + sum(w['steps'] for w in workers),
              'native_counter_total_is_not_current_inflight_accepted_lineage_or_physical_compute_total': True,
              'training_matrix': previous['training_matrix'],
              'training_budgets_complete': 2, 'total_training_cells': 20,
              'inflight_existing_cell': s['job'], 'inflight_state_snapshot': state_name,
              'inflight_phase_at_engineering_observation': s['phase'],
              'engineering_observed_at_unix': s['updated_at_unix'],
              'source_controlled_pause_recorded': s.get('source_paused', False),
              'source_original_kept_in_memory_until_target_first_real_5k_epoch': True,
              'genuine_resume_steps': s.get('resume_steps'),
              'input_preservation': s.get('input_preservation'),
              'target_first_epoch_not_yet_proven_by_this_snapshot': True,
              'conditional_source_tail_if_handoff_completes': {
                  'logged_steps': s.get('known_repeated_environment_steps_lower_bound'),
                  'logged_safety_cost': s.get('known_additional_training_cost_lower_bound'),
                  'not_added_to_completed_recovery_totals_until_target_epoch_and_retirement_proof': True,
                  'unknown_unlogged_work_not_assumed_zero': True},
              'completed_recovery_accounting_before_this_inflight_move': previous['known_recovery_accounting'],
              'latest_published_native_science_summary_reference': root.get('newest_intermediate_scientific_result'),
              'additional_evaluation_rollouts': 0, 'new_training_cells': 0,
              'scientific_parameters_changed': False, 'training_budgets_reduced': False,
              'other_projects_paused': False, 'full_scientific_audit_executed': False,
              'final_scientific_report': False,
              'final_scientific_requirements_unchanged': previous['final_scientific_requirements_unchanged'],
              'legacy_provenance': previous['legacy_provenance'],
              'next_ordinary_observation_local': '2026-10-20T22:21:49+08:00',
              'data_notice': 'Native21:51 frame showed18 advancing existing cells. Subsequently the existingButton2027 source was held for a genuine full5.65M continuation to private5080MPS; this is not a fresh seed. This in-flight snapshot does not prove target training or source retirement, and must not pass the steady-state or final-scientific audit. A5000-step/26-cost logged source tail is conditional recovery work, plus unknown unlogged work, not yet declared as completed repeated work. All original training/evaluation budgets and strict final gates remain; earlier2027 parent gaps are not repaired.'}
    name = 'training_status_20261020_2151_inflight.json'
    write(OUT / name, report)
    shutil.copyfile(BASE / 'capacity_progress_observation_20261020_2151.json', OUT / report['native_observation_file'])
    root.update(superseded_by=name, latest_inflight_engineering_transition=state_name,
                snapshot_notice='Historical17:27 payload below. Latest native frame plus in-flight move:' + name + '. ' + report['data_notice'])
    if root.get('newer_scientific_summary_after_snapshot') == report['latest_published_native_science_summary_reference']:
        root.pop('newer_scientific_summary_after_snapshot', None)
    (OUT / 'training_status.json').write_text(json.dumps(root, indent=2) + '\n')
    print(json.dumps({'status': name, 'inflight_phase': s['phase'], 'source_resume_steps': s.get('resume_steps'),
                      'native_logged_total_before_handoff': report['native_logged_counter_total_including_two_5M_budget_complete_cells']}))


if __name__ == '__main__':
    main()
