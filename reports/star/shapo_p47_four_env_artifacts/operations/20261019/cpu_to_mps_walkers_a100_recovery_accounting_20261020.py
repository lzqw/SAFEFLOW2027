"""Engineering accounting for existing-cell CPU -> GPU MPS transitions.

This is not a scientific audit waiver. The original strict matrix/checkpoint gates
still run, independently, before final publication.
"""
import hashlib
import json
import math
from pathlib import Path

SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
BOOT = '6928f333-1036-473c-92d5-4c482d5fef59'
GPU = 'GPU-164dfe85-4b4b-0f44-fcd8-99ca6d1e8b78'


def verify_A100_Walker_CPU_to_MPS_transitions(base):
    base = Path(base)
    records = []
    for job in ('walker2d-2030','walker2d-2031'):
        path = base/(job+'_cpu_to_mps_a100_gpu1_20261020_state.json')
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        if state.get('phase') != 'completed_first_MPS_epoch_source_retired':
            # Never hide unresolved partial/failed work in the final training cost.
            raise ValueError('MPS transition/failed-attempt accounting unresolved: '+job)
        assert state['job']==job and state['source_commit']==SOURCE
        assert state['total_budget_steps']==5000000
        assert state['scientific_parameters_changed'] is False and state['other_projects_modified'] is False
        assert state['source_runtime_backend']=='CPU' and state['target_runtime_backend']=='private_MPS'
        assert state['source_CPU_backend_verified'] is True
        assert state['source_paused'] is False and state['source_retired'] is True
        assert state['original_training_segments_preserved_at_original_paths'] is True
        assert state['full_replay_rng_optimizers_finite_model'] is True
        preservation=state['input_preservation']
        assert preservation['host_input_sha256']==state['checkpoint_sha256']
        assert preservation['archive_input_sha256']==state['checkpoint_sha256']
        assert preservation['archive_separate_4T_filesystem_verified'] is True
        assert hashlib.sha256(Path(preservation['host_input_path']).read_bytes()).hexdigest()==state['checkpoint_sha256']
        for filename,key in [('handoff_walkers_A100_GPU1_MPS_20261020.py','controller_sha256'),
                             ('mps_training_bootstrap_20261020.py','bootstrap_sha256')]:
            assert hashlib.sha256((base/filename).read_bytes()).hexdigest()==state[key]
        steps=state['last_logged_source_steps']-state['resume_steps']
        cost=state['last_logged_source_cost']-state['resume_cumulative_training_cost']
        assert 0<=steps<=5000 and steps==state['known_repeated_steps_lower_bound']
        assert math.isfinite(cost) and cost>=0 and cost==state['known_extra_training_cost_lower_bound']
        assert state['unlogged_source_tail_unknown_not_zero'] is True
        source,target=state['source_identity'],state['target_identity']
        assert source['pid']!=target['pid'] and source['boot_id']==target['boot_id']==BOOT
        assert source['device']=='cpu' and target['device']=='cuda:0'
        assert source['cuda_visible_devices']=='-1'
        assert target['cuda_visible_devices']==GPU
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
        records.append({'job':job,'runtime_placement':'same_A100_host_GPU1_private_MPS_execution',
                        'physical_gpu_label':'A100 GPU1',
                        'completion_state':state,'first_target_native_observation':native,
                        'known_additional_physical_steps':steps,
                        'known_additional_physical_training_cost':cost,
                        'unknown_unlogged_tail_not_zero':True})
    return records
