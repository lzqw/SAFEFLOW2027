#!/usr/bin/env python3
"""Publish completed runtime evidence without inventing a fresh all-host frame."""
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/operations/20261019')


def read(path):
    return json.loads(path.read_text())


def main():
    prefix = 'point-button-2027_balanced_MPS_5080_20261020'
    s = read(BASE / (prefix + '_state.json'))
    check_name = 'Button2027_completed_handoff_accounting_check_20261020_2206.json'
    check = read(BASE / check_name)
    assert s['phase'] == 'handoff_completed_first_target_epoch_verified'
    assert s['source_retired'] is True and s['source_paused'] is False
    assert check['verified_new_handoff_records'] == 1
    assert s['checkpoint_sha256'] == check['input_checkpoint_sha256']
    assert s['target_identity']['pid'] == check['target_pid']
    assert s['first_verified_target_steps'] == check['first_verified_target_steps']
    assert s['known_repeated_environment_steps_lower_bound'] == 5000
    assert s['known_additional_training_cost_lower_bound'] == 26
    for name in (prefix + '_state.json', prefix + '_receipt.json',
                 s['target_first_epoch_native_observation'], check_name):
        shutil.copyfile(BASE / name, OUT / name)
    root = read(OUT / 'training_status.json')
    old = read(OUT / 'training_status_20261020_2121.json')
    report = {'schema_version': 1, 'kind': 'completed_existing_cell_runtime_handoff_not_full_cohort_snapshot',
        'engineering_completed_at_unix': s['updated_at_unix'],
        'previous_status': 'training_status_20261020_2151_inflight.json',
        'latest_all_host_native_frame': 'ordinary_observation_20261020_2151.json',
        'latest_target_native_probe': s['target_first_epoch_native_observation'],
        'completed_state': prefix + '_state.json', 'completed_accounting_check': check_name,
        'historical_input_transfer_receipt_not_completion_proof': prefix + '_receipt.json',
        'new_target_identity': s['target_identity'], 'verified_target_GPU_UUID': s['target_gpu_uuid_verified'],
        'first_target_steps': s['first_verified_target_steps'], 'first_target_FPS': s['first_verified_target_fps'],
        'source_last_epoch_FPS': check['source_last_epoch_FPS'],
        'first_epoch_only_speed_ratio_not_sustained_or_causal_guarantee': check['first_epoch_only_ratio_not_sustained_guarantee'],
        'new_5080_Point_GPU_client_count_at_target_probe': 10,
        'target_Point_FPS_snapshot_sum_not_whole_training_guarantee': check['target_native_Point_FPS_snapshot_sum'],
        'source_retired_after_target_first_epoch': True,
        'source_original_directory_recoverably_archived': s['retired_source_archive'],
        'source_canonical_path_vacated_for_existing_final_collector': True,
        'input_preservation': s['input_preservation'],
        'new_known_additional_physical_steps_lower_bound': 5000,
        'new_known_additional_physical_safety_cost_lower_bound': 26,
        'completed_recovery_totals_lower_bound': {'environment_steps': 1142276, 'safety_cost': 10370,
            'unknown_historical_and_unlogged_work_not_assumed_zero': True},
        'last_all_host_native_logical_active_cells': 18,
        'post_handoff_placement_from_target_probe_and_verified_source_retirement': {
            'tjucta-a100': {'GPU': 1, 'CPU': 2}, '4080': {'GPU': 0, 'CPU': 1},
            '5080': {'GPU': 10, 'CPU': 4}},
        'not_a_new_simultaneous_all_host_observation_or_spliced_progress_total': True,
        'Ant_CPU_handoffs': {'queue': 'Ant_CPU_5080_to_4080_regular_dispatch_20261020_state.json',
            'pending': ['ant-2028', 'ant-2029'], 'sources_continue_until_fresh_full_input_and_live_CPU_capacity': True},
        'latest_native_science_summary_reference': root.get('newest_intermediate_scientific_result'),
        'training_matrix': old['training_matrix'], 'last_known_training_budgets_complete': 2,
        'total_training_cells': 20, 'additional_training_cells': 0,
        'additional_scientific_evaluation_rollouts': 0, 'scientific_parameters_changed': False,
        'training_budgets_reduced': False, 'other_projects_paused': False,
        'full_final_scientific_audit_executed': False, 'final_scientific_report': False,
        'final_scientific_requirements_unchanged': old['final_scientific_requirements_unchanged'],
        'legacy_provenance': old['legacy_provenance'],
        'next_ordinary_resource_check_local': '2026-10-20T22:21:49+08:00',
        'data_notice': 'ExistingButton2027 continued from genuine5.65M input on private5080MPS; a real5000-step epoch and native client/server/GPU proof preceded source retirement and recoverable source archival. First epoch13.333FPS vs source4.828FPS is not a sustained whole-training guarantee. Target cohort now10 PointGPU clients. Known extra5000steps/26cost are now included:global floors1142276steps/10370cost plus unknown work. This is not a new seed or an early2027 provenance repair, and not a final20-cell scientific report. Latest all-host frame remains21:51; next regular all-host observation22:21 will supply fresh cohort counters.'}
    name = 'training_status_20261020_2204_handoff_completed.json'
    target = OUT / name
    assert not target.exists(), 'Never overwrite completed engineering snapshot'
    target.write_text(json.dumps(report, indent=2) + '\n')
    root.pop('latest_inflight_engineering_transition', None)
    root.update(superseded_by=name, latest_completed_engineering_transition=prefix + '_state.json',
                newer_engineering_runtime=name,
                snapshot_notice='Historical17:27 payload below. Latest completed engineering evidence:' + name + '. ' + report['data_notice'])
    (OUT / 'training_status.json').write_text(json.dumps(root, indent=2) + '\n')
    print(json.dumps({'completed_status': name, 'target_pid': check['target_pid'],
                      'known_recovery_floors': report['completed_recovery_totals_lower_bound']}))


if __name__ == '__main__':
    main()
