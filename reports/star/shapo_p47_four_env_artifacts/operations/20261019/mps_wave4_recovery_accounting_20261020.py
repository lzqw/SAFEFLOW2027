"""Engineering accounting for existing-cell default-CUDA -> MPS transitions.

This is not a scientific audit waiver. The original strict matrix/checkpoint gates
still run, independently, before final publication.
"""
import hashlib
import json
import math
from pathlib import Path

SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
BOOT = '5f46bb0c-b4ec-4802-8f6e-01ac611c0f2f'
GPU = 'GPU-ac315b09-0bb7-2072-efd1-cdfc5fae96a5'


def verify_wave4_MPS_transitions(base):
    base = Path(base)
    records = []
    for job in ('point-goal-2031', 'point-button-2031'):
        path = base/(job+'_mps_wave4_5080_20261020_state.json')
        recovered = base/(job+'_mps_wave4_recovery_5080_20261020_state.json')
        recovery_receipt = None
        if recovered.exists():
            recovery_receipt=json.loads((base/'mps_wave4_prelaunch_storage_abort_resolution_20261020.json').read_text())
            helper=base/recovery_receipt['recovery_helper']
            assert helper.name=='recover_wave4_after_host_storage_guard_20261020.py'
            assert hashlib.sha256(helper.read_bytes()).hexdigest()==recovery_receipt['recovery_helper_sha256']
            assert recovery_receipt['original_transaction_implementation_sha256']=='458e802571d17d47acb70586ce9ed4e82334e08381a756497ce7402f5b29de84'
            assert recovery_receipt['transaction_namespace_override']=='mps_wave4_recovery_5080_20261020'
            failed_path=base/recovery_receipt['failed_original_state']
            assert failed_path.name=='point-button-2031_mps_wave4_5080_20261020_state.json'
            assert hashlib.sha256(failed_path.read_bytes()).hexdigest()==recovery_receipt['failed_original_state_sha256']
            failed=json.loads(failed_path.read_text())
            assert failed['target_launch_requested'] is False and failed['target_absence_confirmed'] is True
            assert failed['original_source_resumed_in_memory'] is True and failed['source_paused'] is False
            assert recovery_receipt['prelaunch_failed_attempt_additional_physical_steps']==0
            assert recovery_receipt['prelaunch_failed_attempt_additional_training_cost']==0
            assert recovery_receipt['no_original_state_log_or_input_deleted'] is True
            assert len(recovery_receipt['source_continuity_proofs'])==2
            for row in recovery_receipt['source_continuity_proofs']:
                assert row['source_identity']['state'] not in ('T','t')
                assert row['source_identity']['boot_id']==BOOT
            path=recovered
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        if state.get('phase') != 'completed_first_MPS_epoch_source_retired':
            # Never hide unresolved partial/failed work in the final training cost.
            raise ValueError('MPS transition/failed-attempt accounting unresolved: '+job)
        assert state['job']==job and state['source_commit']==SOURCE
        assert state['total_budget_steps']==10000000
        assert state['scientific_parameters_changed'] is False and state['other_projects_modified'] is False
        assert state['source_runtime_backend']=='default_CUDA' and state['target_runtime_backend']=='private_MPS'
        assert state['source_default_CUDA_backend_verified'] is True
        assert state['source_paused'] is False and state['source_retired'] is True
        assert state['original_training_segments_preserved_at_original_paths'] is True
        assert state['full_replay_rng_optimizers_finite_model'] is True
        preservation=state['input_preservation']
        assert preservation['host_input_sha256']==state['checkpoint_sha256']
        assert preservation['archive_input_sha256']==state['checkpoint_sha256']
        assert preservation['archive_separate_4T_filesystem_verified'] is True
        assert hashlib.sha256(Path(preservation['host_input_path']).read_bytes()).hexdigest()==state['checkpoint_sha256']
        for filename,key in [('handoff_point2031_MPS_wave4_5080_20261020.py','controller_sha256'),
                             ('mps_training_bootstrap_20261020.py','bootstrap_sha256')]:
            assert hashlib.sha256((base/filename).read_bytes()).hexdigest()==state[key]
        steps=state['last_logged_source_steps']-state['resume_steps']
        cost=state['last_logged_source_cost']-state['resume_cumulative_training_cost']
        assert 0<=steps<=5000 and steps==state['known_repeated_steps_lower_bound']
        assert math.isfinite(cost) and cost>=0 and cost==state['known_extra_training_cost_lower_bound']
        assert state['unlogged_source_tail_unknown_not_zero'] is True
        source,target=state['source_identity'],state['target_identity']
        assert source['pid']!=target['pid'] and source['boot_id']==target['boot_id']==BOOT
        assert source['device']==target['device']=='cuda:0'
        assert source['cuda_visible_devices']==target['cuda_visible_devices']=='0'
        proof=state['MPS_connection_proof']
        assert proof['connection_verified'] is True and proof['gpu_uuid']==GPU
        assert proof['client_pid']==target['pid'] and target['pid'] in proof['client_pids']
        first=state['first_target_epoch']
        assert first['steps']>=state['resume_steps']+5000 and first['cost']>=state['resume_cumulative_training_cost']
        retirement=state['wrapper_retirement']
        assert retirement['original_wrapper_exit_code']!=0
        assert retirement['old_full_input_still_at_original_path_and_hash'] is True
        assert retirement['target_marker_promoted'] is True
        native=json.loads((base/state['target_first_epoch_native_observation']).read_text())
        workers=[w for w in native['training'] if w['job']==job]
        assert len(workers)==1
        worker=workers[0]
        assert worker['pid']==target['pid'] and worker['start_ticks']==target['start_ticks']
        assert worker['boot_id']==BOOT and worker['progress_csv']==first['csv']
        assert worker['steps']>=first['steps'] and worker['mps_gpu_uuid_verified']==GPU
        assert worker['cuda_execution_backend']=='mps_verified'
        assert worker['mps_connection']['server_pid']==proof['server_pid']
        records.append({'job':job,'runtime_placement':'same5080_GPU_private_MPS_execution',
                        'prelaunch_storage_abort_resolution':recovery_receipt,
                        'completion_state':state,'first_target_native_observation':native,
                        'known_additional_physical_steps':steps,
                        'known_additional_physical_training_cost':cost,
                        'unknown_unlogged_tail_not_zero':True})
    return records
