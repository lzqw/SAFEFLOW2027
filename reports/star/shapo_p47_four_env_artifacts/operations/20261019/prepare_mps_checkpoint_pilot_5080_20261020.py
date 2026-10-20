#!/usr/bin/env python3
"""Prepare, but do not start, a two-existing-cell MPS checkpoint transition.

This emits the exact engineering contract to be consumed by a separately reviewed
handoff controller. Preparation never signals trainers or starts a daemon.
"""
import hashlib
import json
import subprocess
from pathlib import Path

OPS = Path(__file__).resolve().parent
ROOT = '/home/autodrive/LZQW/SAFEFLOW2027'


def main():
    observation = json.loads((OPS/'capacity_progress_observation_20261020_0951.json').read_text())
    comparison = json.loads((OPS/'frozen_update_MPS_comparison_5080_20261020_0944.json').read_text())
    host = observation['hosts']['5080']
    gpu = host['gpus'][0]
    assert gpu['uuid'] == 'GPU-ac315b09-0bb7-2072-efd1-cdfc5fae96a5'
    assert gpu['foreign_pids'] == [] and gpu['safeflow_gpu_processes'] == 8
    assert comparison['mps']['MPS_connection_proof']['both_disposable_client_PIDs_proven']
    assert comparison['MPS_disposable_clients_exited'] and comparison['private_daemon_quit_stdout'] == ''
    assert comparison['MPS_over_default_update_throughput_ratio_not_full_training_guarantee'] > 1.2
    workers = [w for w in host['training'] if w['job'] in ('point-goal-2029','point-button-2029')]
    assert len(workers) == 2 and all(w['device']=='cuda:0' for w in workers)
    plan = {
        'kind':'engineering_checkpoint_transition_contract_not_started',
        'host':'5080', 'project':ROOT,
        'source_commit':'62ca7399357c10645a955dcbead7ed2153a82c69',
        'gpu_uuid':gpu['uuid'], 'expected_boot_id':workers[0]['boot_id'],
        'workers_at_ordinary_observation':workers,
        'formal_transition_started':False, 'MPS_daemon_started':False,
        'source_trainers_paused_or_restarted':False,
        'training_configuration_and_budgets_changed':False,
        'budgets_per_cell':10000000,
        'input_requirements':{
            'native_checkpoint_name':'latest_full.pt',
            'must_remain_at_original_same_run_training_path':True,
            'full_replay_rng_optimizers_and_finite_model_required':True,
            'wait_for_fresh_checkpoint_if_logged_source_ahead_by_more_than_steps':5000,
            'immutable_input_copy_and_sha256_required_before_launch':True,
            'host_and_4T_backups_must_preserve_genuine_full_input':True,
        },
        'transition_requirements':{
            'matching_pid_start_ticks_boot_config_before_any_signal':True,
            'no_global_compute_mode_changes_or_unrelated_process_actions':True,
            'pause_only_one_original_default_CUDA_worker_at_a_time':True,
            'keep_original_paused_in_memory_until_new_native_epoch':True,
            'same_container_keep_all_old_training_segments_at_original_paths':True,
            'new_target_log_and_exit_marker_names_must_be_unique':True,
            'require_new_native_5000_step_epoch_and_live_MPS_client_server_UUID_proof':True,
            'retire_original_default_CUDA_worker_only_after_target_proof':True,
            'archive_original_wrapper_exit_marker_before_target_canonical_marker_promotion':True,
            'failed_target_absence_required_before_resuming_original_in_memory':True,
            'do_not_SIGKILL_MPS_clients_or_quit_daemon_with_live_formal_clients':True,
        },
        'accounting_requirements':{
            'record_input_step_cost_hash_and_last_logged_source_step_cost':True,
            'do_not_claim_unlogged_source_tail_zero':True,
            'failed_target_work_if_original_resumes_is_separate_attempt_not_source_rollback':True,
            'final_accounting_publisher_must_include_completed_and_aborted_transition_receipts':True,
            'all_existing_scientific_gates_and_legacy_provenance_gaps_unchanged':True,
        },
        'prepared_bootstrap':{
            'filename':'mps_training_bootstrap_20261020.py',
            'sha256':hashlib.sha256((OPS/'mps_training_bootstrap_20261020.py').read_bytes()).hexdigest(),
            'same_frozen_train_script_via_runpy':True,
            'synchronize_CUDA_on_natural_exception_or_graceful_SIGTERM_exit':True,
        },
        'resource_observer':{
            'filename':'watch_gpu_capacity.py',
            'sha256':hashlib.sha256((OPS/'watch_gpu_capacity.py').read_bytes()).hexdigest(),
            'verified_MPS_client_counting_implemented':True,
            'syntax_checked_no_extra_remote_sampling':True,
            'normal_schedule_preserved_after_monitor_only_restart':True,
        },
        'remaining_before_actual_launch':[
            'Implement and review identity-checked same-container handoff controller and rollback path.',
            'Implement receipt accounting hook before introducing a new accepted runtime segment.',
            'Use next ordinary observation for both complete P-core pilot epochs.',
            'Revalidate live identities and GPU ownership at launch, not only from this saved observation.'
        ],
        'performance_evidence_limits':{
            'MPS_test_seconds_per_phase':12,
            'MPS_updates_only_ratio':comparison['MPS_over_default_update_throughput_ratio_not_full_training_guarantee'],
            'default_measurement_reused_within_minutes_not_simultaneous':True,
            'full_training_acceleration_not_proven':True,
        },
        'ordinary_observer_next_check_local':'2026-10-20T10:21:35+08:00',
    }
    target = OPS/'mps_checkpoint_pilot_5080_20261020_preparation.json'
    assert not target.exists(), 'Do not overwrite an existing preparation or live transition'
    target.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'plan':str(target),'formal_transition_started':False,
                      'workers_selected':[w['job'] for w in workers]}))


if __name__ == '__main__':
    main()
