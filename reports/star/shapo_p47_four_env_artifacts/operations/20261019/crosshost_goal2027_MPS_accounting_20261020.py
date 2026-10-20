"""Account only a genuine completed continuation of the existing Goal2027 cell.

This does not repair the known missing early2027 parents or waive final gates.
"""
import hashlib
import json
import math
from pathlib import Path

SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
JOB = 'point-goal-2027'
SIDE = 'balanced_MPS_5080_20261020'
SRC_BOOT = '6928f333-1036-473c-92d5-4c482d5fef59'
DST_BOOT = '5f46bb0c-b4ec-4802-8f6e-01ac611c0f2f'
GPU = 'GPU-ac315b09-0bb7-2072-efd1-cdfc5fae96a5'


def verify_Goal2027_crosshost_MPS(base):
    base=Path(base);path=base/(JOB+'_'+SIDE+'_state.json')
    if not path.exists():
        return []
    s=json.loads(path.read_text())
    if s.get('phase')!='handoff_completed_first_target_epoch_verified':
        raise ValueError('Goal2027 crosshost partial/failed runtime accounting unresolved')
    assert s['job']==JOB and s['seed']==2027 and s['source_commit']==SOURCE
    assert s['source_host']=='tjucta-a100' and s['target_host']=='5080'
    assert s['total_training_budget']==10000000
    assert s['scientific_parameters_changed'] is False and s['additional_evaluation'] is False
    assert s['source_default_CUDA_backend_verified'] is True
    assert s['source_paused'] is False and s['source_retired'] is True
    assert s['source_canonical_path_vacated_for_final_collection'] is True
    assert s['original_evidence_preserved'] is True
    assert s['retired_source_full_checkpoint_sha256_verified'] is True
    assert s['retired_source_full_checkpoint'].startswith(s['retired_source_archive']+'/')
    assert s['source_wrapper_retirement']['original_wrapper_exit_code']!=0
    assert s['source_wrapper_retirement']['retired_source_directory_preserved_at']==s['retired_source_archive']
    assert s['full_replay_and_rng_present'] is True and s['full_optimizers_and_finite_model_verified'] is True
    for filename,key in [('handoff_Goal2027_A100_to_5080_MPS_20261020.py','controller_sha256'),
                         ('handoff_point2029_MPS_5080_20261020.py','MPS_helper_sha256'),
                         ('mps_training_bootstrap_20261020.py','bootstrap_sha256')]:
        assert hashlib.sha256((base/filename).read_bytes()).hexdigest()==s[key]
    preservation=s['input_preservation'];sha=s['checkpoint_sha256']
    assert preservation['host_input_sha256']==preservation['archive_input_sha256']==sha
    assert preservation['archive_separate_4T_filesystem_verified'] is True
    assert hashlib.sha256(Path(preservation['host_input_path']).read_bytes()).hexdigest()==sha
    steps=s['last_logged_source_steps']-s['resume_steps']
    cost=s['last_logged_source_cumulative_training_cost']-s['resume_cumulative_training_cost']
    assert 0<=steps<=5000 and steps==s['known_repeated_environment_steps_lower_bound']
    assert math.isfinite(cost) and cost>=0 and cost==s['known_additional_training_cost_lower_bound']
    assert s['physical_repeated_steps_and_cost_unknown_not_zero'] is True
    assert s['unlogged_source_tail_may_exist'] is True
    src,dst=s['source_identity'],s['target_identity']
    assert src['boot_id']==SRC_BOOT and dst['boot_id']==DST_BOOT
    assert src['pid']==7777 and src['start_ticks']==74027
    assert src['device']==dst['device']=='cuda:0'
    assert src['cuda_visible_devices']==dst['cuda_visible_devices']=='0'
    runtime=s['runtime_change']
    assert runtime['from_torch']=='2.7.1+cu118' and runtime['to_torch']=='2.7.1+cu128'
    assert runtime['target_backend']=='private_MPS' and runtime['bitwise_equivalence_claimed'] is False
    proof=s['MPS_connection_proof'];q=s['plan']['rebalance_qualification']
    assert proof['connection_verified'] is True and proof['gpu_uuid']==s['target_gpu_uuid_verified']==GPU
    assert proof['client_pid']==dst['pid'] and dst['pid'] in proof['client_pids']
    assert proof['server_pid']==q['MPS_server_pid']
    assert set(q['owned_gpu_pids'])<=set(proof['client_pids'])
    assert len(q['owned_gpu_pids'])==8 and not q['foreign_pids']
    assert q['observed_at_unix']-q['previous_observed_at_unix']>=1800
    assert s['first_verified_target_steps']>=s['resume_steps']+5000
    assert s['first_verified_target_cost']>=s['resume_cumulative_training_cost']
    native=json.loads((base/s['target_first_epoch_native_observation']).read_text())
    rows=[w for w in native['training'] if w['job']==JOB];assert len(rows)==1
    w=rows[0]
    assert w['pid']==dst['pid'] and w['start_ticks']==dst['start_ticks'] and w['boot_id']==DST_BOOT
    assert w['mps_gpu_uuid_verified']==GPU and w['cuda_execution_backend']=='mps_verified'
    assert w['steps']>=s['first_verified_target_steps'] and w['progress_csv']==s['first_verified_target_csv']
    return [{'job':JOB,'runtime_placement':'existing_A100_Goal2027_to_5080_private_MPS',
             'receipt':s,'first_target_native_observation':native,
             'known_additional_physical_steps':steps,
             'known_additional_physical_training_cost':cost,
             'unknown_unlogged_tail_not_zero':True}]
