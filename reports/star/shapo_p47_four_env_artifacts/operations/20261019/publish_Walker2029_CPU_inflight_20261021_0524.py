#!/usr/bin/env python3
"""Record an existing CPU handoff separately from the earlier all-host census."""
import json
import time

from publish_Walker2030_A100_MPS_inflight_20261021_0322 import BASE, OUT, immutable, read


def main():
    prefix = 'walker2d-2029_walker_cpu_5080_to_4080_20261021'
    s = read(BASE / (prefix + '_state.json'))
    assert s['phase'] in ('copying_paused_CPU_source', 'target_CPU_starting_source_CPU_paused')
    assert s['source_paused'] is True and s['source_retired'] is False
    queue = read(BASE / 'Walker_CPU_5080_to_4080_regular_dispatch_20261021_state.json')
    assert queue['second_move_requires_observed_same_environment_FPS_gain_ratio'] == 1.2
    old = read(OUT / 'training_status_20261021_0521.json')
    assert old['observed_at_unix'] < s['updated_at_unix']
    state_name = prefix + '_inflight_state_20261021_0524.json'
    queue_name = 'snapshot_20261021_0524_Walker_CPU_5080_to_4080_regular_dispatch_20261021_state.json'
    immutable(OUT / state_name, s)
    immutable(OUT / queue_name, queue)
    root = read(OUT / 'training_status.json')
    report = {
        'kind': 'later_Walker2029_CPU_to_CPU_transaction_not_new_all_host_census',
        'engineering_observed_at_unix': time.time(), 'previous_status': root['superseded_by'],
        'last_steady_training_status': 'training_status_20261021_0521.json',
        'last_all_host_native_frame_before_transaction': 'ordinary_observation_20261021_0521.json',
        'pre_pause_worker_counts_not_relabelled_as_post_pause_census': True,
        'job': s['job'], 'inflight_phase': s['phase'],
        'inflight_state_snapshot': state_name, 'dispatcher_snapshot': queue_name,
        'source_held_paused_until_actual_first_target_epoch': True,
        'target_first_epoch_and_source_retirement_not_yet_proven': True,
        'source_identity': s['source_identity'], 'runtime_change': s['runtime_change'],
        'genuine_resume_steps': s['resume_steps'], 'genuine_input_sha256': s['checkpoint_sha256'],
        'input_preservation': s.get('input_preservation'),
        'target_capacity_before_pause': s['target_capacity_before_pause'],
        'original_runtime_FPS_gain_requirement_unchanged': 1.2,
        'observed_runtime_admission_ratio': s['same_environment_target_FPS_before_this_move'] / s['expected_source_identity']['last_epoch_fps'],
        'runtime_admission_not_scientific_policy_selection': True,
        'GPU_capacity_not_claimed_or_used': True,
        'conditional_repeated_work_if_completed': {
            'logged_steps': s['known_repeated_environment_steps_lower_bound'],
            'safety_cost': s['known_additional_training_cost_lower_bound'],
            'not_added_to_completed_recovery_totals_yet': True,
            'unknown_unlogged_source_tail_not_zero': True},
        'last_completed_recovery_accounting': old['known_recovery_accounting'],
        'full_training_budget_steps': 5000000,
        'new_experiment_cells': 0, 'additional_scientific_evaluation_rollouts': 0,
        'scientific_parameters_changed': False, 'foreign_projects_modified': False,
        'full_final_scientific_audit_executed': False, 'final_scientific_report': False,
        'legacy_provenance': old['legacy_provenance'],
        'final_scientific_requirements_unchanged': old['final_scientific_requirements_unchanged'],
        'next_regular_resource_check_local': old['next_ordinary_observation_local'],
        'data_notice': 'The05:21 all-host census preceded this later Walker2029 CPU transfer. Do not read those counters as a post-pause worker census. The existing unchanged dispatcher admitted the move using its original20percent same-environment FPS rule plus fresh full-checkpoint and live CPU headroom checks. Target staysCPU; no4080GPU allocation or foreign task pause occurs. Genuine2.3M full replay/RNG/optimizer bytes are preserved; source remains held until a real new5k target epoch. All5M budgets/source62ca739/final gates remain; early2027 parent gaps are unresolved.'}
    name = 'training_status_20261021_0524_Walker2029_CPU_inflight.json'
    immutable(OUT / name, report)
    root.update(superseded_by=name, latest_inflight_engineering_transition=state_name,
                snapshot_notice='Historical17:27 payload below. Latest engineering report:' + name + '. ' + report['data_notice'])
    temporary = OUT / 'training_status.Walker2029_inflight.tmp'
    with temporary.open('x') as stream:
        stream.write(json.dumps(root, indent=2) + '\n')
    temporary.replace(OUT / 'training_status.json')
    print(json.dumps({'status': name, 'phase': s['phase'], 'first_target_epoch_proven': False}))


if __name__ == '__main__':
    main()
