#!/usr/bin/env python3
"""Publish the later Ant transaction separately from the preceding native census."""
import json
import time

from publish_Walker2030_A100_MPS_inflight_20261021_0322 import BASE, OUT, immutable, read


def main():
    prefix = 'ant-2030_cpu_to_mps_a100_gpu0_20261020'
    s = read(BASE / (prefix + '_state.json'))
    assert s['phase'] == 'MPS_target_starting_source_paused'
    assert s['source_paused'] is True and s['source_retired'] is False
    old = read(OUT / 'training_status_20261021_0351.json')
    assert old['observed_at_unix'] < s['updated_at_unix']
    state_name = prefix + '_inflight_state_20261021_0353.json'
    queue_name = 'snapshot_20261021_0353_A100_unstarted_transfers_post_local_reboot_20261020_state.json'
    immutable(OUT / state_name, s)
    immutable(OUT / queue_name, read(BASE / 'A100_unstarted_transfers_post_local_reboot_20261020_state.json'))
    root = read(OUT / 'training_status.json')
    report = {
        'kind': 'later_Ant2030_CPU_to_GPU0_partial_handoff_not_new_all_host_frame',
        'engineering_observed_at_unix': time.time(), 'previous_status': root['superseded_by'],
        'last_steady_training_status': 'training_status_20261021_0351.json',
        'last_all_host_native_frame_before_transaction': 'ordinary_observation_20261021_0351.json',
        'pre_pause_census_not_relabelled_as_current_post_pause_workers': True,
        'job': s['job'], 'inflight_phase': s['phase'],
        'inflight_state_snapshot': state_name, 'dispatcher_snapshot': queue_name,
        'source_identity': s['source_identity'],
        'source_held_paused_until_verified_first_target_epoch': True,
        'target_launch_requested': s['target_launch_requested'],
        'first_target_epoch_and_source_retirement_not_yet_proven': True,
        'genuine_resume_steps': s['resume_steps'], 'genuine_input_sha256': s['checkpoint_sha256'],
        'genuine_input_has_full_replay_rng_optimizers_and_finite_model': s['full_replay_rng_optimizers_finite_model'],
        'input_preservation': s['input_preservation'], 'capacity_before_pause': s['capacity_before_pause'],
        'physical_target': 'A100 GPU0 private MPS',
        'bitwise_trajectory_equivalence_claimed': False,
        'conditional_repeated_work_if_completed': {
            'logged_steps': s['known_repeated_steps_lower_bound'],
            'safety_cost': s['known_extra_training_cost_lower_bound'],
            'not_added_to_completed_recovery_totals_yet': True,
            'unknown_unlogged_source_tail_not_zero': True},
        'last_completed_recovery_accounting': old['known_recovery_accounting'],
        'full_training_budget_steps': 5000000,
        'additional_experiment_cells': 0, 'additional_scientific_evaluation_rollouts': 0,
        'scientific_parameters_changed': False, 'foreign_projects_modified': False,
        'no_speed_gain_guaranteed_before_actual_target_epochs': True,
        'full_final_scientific_audit_executed': False, 'final_scientific_report': False,
        'legacy_provenance': old['legacy_provenance'],
        'final_scientific_requirements_unchanged': old['final_scientific_requirements_unchanged'],
        'next_regular_resource_check_local': old['next_ordinary_observation_local'],
        'data_notice': 'The03:51 all-host census preceded the later Ant2030 CPU pause and GPU0 launch request. It remains an immutable pre-transaction frame, not a current worker count. Ant resumes from its genuine4.35M full replay/RNG/optimizer checkpoint, preserved on source, host and separate4T. Source remains held until a real new5k epoch and actual MPS GPU/client proof, then retirement. Known logged rollback is0, but unlogged source tail is unknown, not assumed zero. All5M budgets, source62ca739 and final gates remain unchanged; early2027 parent gaps remain unresolved.'}
    name = 'training_status_20261021_0353_Ant2030_A100_inflight.json'
    immutable(OUT / name, report)
    root.update(superseded_by=name, latest_inflight_engineering_transition=state_name,
                snapshot_notice='Historical17:27 payload below. Latest engineering report:' + name + '. ' + report['data_notice'])
    temporary = OUT / 'training_status.Ant2030_inflight.tmp'
    with temporary.open('x') as stream:
        stream.write(json.dumps(root, indent=2) + '\n')
    temporary.replace(OUT / 'training_status.json')
    print(json.dumps({'status': name, 'phase': s['phase'], 'first_target_epoch_proven': False}))


if __name__ == '__main__':
    main()
