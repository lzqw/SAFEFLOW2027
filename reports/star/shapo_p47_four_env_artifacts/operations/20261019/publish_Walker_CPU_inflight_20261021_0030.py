#!/usr/bin/env python3
"""Publish later engineering state without splicing a pre-handoff native frame."""
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/operations/20261019')


def read(path):
    return json.loads(path.read_text())


def immutable(path, value):
    assert not path.exists(), 'Never overwrite an engineering observation'
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    prefix = 'walker2d-2028_walker_cpu_5080_to_4080_20261021'
    state = read(BASE / (prefix + '_state.json'))
    assert state['phase'] in ('copying_paused_CPU_source', 'target_CPU_starting_source_CPU_paused')
    assert state['source_paused'] is True and state['source_retired'] is False
    state_name = prefix + '_inflight_state_20261021_0030.json'
    immutable(OUT / state_name, state)
    queue_name = 'snapshot_20261021_0030_Walker_CPU_5080_to_4080_regular_dispatch_20261021_state.json'
    immutable(OUT / queue_name, read(BASE / 'Walker_CPU_5080_to_4080_regular_dispatch_20261021_state.json'))
    root = read(OUT / 'training_status.json')
    old = read(OUT / 'training_status_20261021_0021.json')
    report = {'kind': 'existing_Walker_CPU_inflight_engineering_not_new_native_cohort_frame',
        'previous_status': root['superseded_by'], 'engineering_observed_at_unix': time.time(),
        'latest_all_host_native_frame': 'ordinary_observation_20261021_0021.json',
        'earlier_native_cohort_counters_not_relabelled_as_current_migration_counters': True,
        'job': state['job'], 'inflight_phase': state['phase'],
        'inflight_state_snapshot': state_name, 'dispatcher_snapshot': queue_name,
        'source_identity': state['source_identity'], 'source_held_in_memory_until_real_new_target_epoch': True,
        'target_first_epoch_and_source_retirement_not_yet_proven': True,
        'genuine_resume_steps': state['resume_steps'], 'genuine_input_sha256': state['checkpoint_sha256'],
        'runtime_change': state['runtime_change'], 'GPU_capacity_not_claimed_or_used': True,
        'target_capacity_before_pause': state['target_capacity_before_pause'],
        'conditional_repeated_work_if_move_completes': {
            'logged_steps': state['known_repeated_environment_steps_lower_bound'],
            'safety_cost': state['known_additional_training_cost_lower_bound'],
            'not_added_to_completed_recovery_totals_yet': True,
            'unknown_unlogged_source_tail_not_zero': True},
        'last_completed_recovery_accounting': old['known_recovery_accounting'],
        'second_Walker_move_requires_actual_first_target_same_environment_FPS_gain': True,
        'new_training_cells': 0, 'additional_scientific_evaluation_rollouts': 0,
        'scientific_parameters_changed': False, 'budgets_reduced': False, 'other_projects_paused': False,
        'full_scientific_audit_executed': False, 'final_scientific_report': False,
        'legacy_provenance': old['legacy_provenance'],
        'final_scientific_requirements_unchanged': old['final_scientific_requirements_unchanged'],
        'next_ordinary_resource_check_local': '2026-10-21T00:51:51+08:00',
        'data_notice': 'Existing Walker2028 is being continued from a genuine2.2M full replay/RNG/optimizer checkpoint. All original5M budgets and scientific settings remain. Source held in memory; target first5k epoch and source retirement are not yet established. No candidate policy selection or extra evaluation occurs. First placement uses live CPU capacity, not an unmeasured Walker speed guarantee. Second placement requires actual target Walker FPS improvement. This is runtime-only engineering admission, not a scientific metric gate. Early2027 original full-parent gaps remain unresolved.'}
    name = 'training_status_20261021_0030_Walker_inflight.json'
    immutable(OUT / name, report)
    root.update(superseded_by=name, latest_inflight_engineering_transition=state_name,
                snapshot_notice='Historical17:27 payload below. Latest later engineering transition:' + name + '. ' + report['data_notice'])
    (OUT / 'training_status.json').write_text(json.dumps(root, indent=2) + '\n')
    print(json.dumps({'status': name, 'phase': state['phase'], 'resume_steps': state['resume_steps']}))


if __name__ == '__main__':
    main()
