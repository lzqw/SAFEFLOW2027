"""Account new existing-Point GPU placements, without waiving final gates."""
import hashlib
import json
import math
from pathlib import Path

SIDE = 'point_5080_to_4080_MPS_20261021'
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
GPU = 'GPU-66ed7e12-a0c3-9273-463d-56a5e5f0a4f7'


def verify_Point_5080_to_4080_MPS(base):
    base = Path(base)
    records = []
    for path in sorted(base.glob('point-*_' + SIDE + '_state.json')):
        s = json.loads(path.read_text())
        if s['phase'] != 'completed_first_MPS_epoch_source_retired':
            raise ValueError('Unresolved Point GPU placement: ' + path.name)
        assert s['source_commit'] == SOURCE and s['total_training_budget'] == 10000000
        assert s['source_host'] == '5080' and s['target_host'] == '4080'
        assert s['job'] in {slug + '-' + str(seed) for slug in ('point-goal', 'point-button') for seed in range(2027, 2032)}
        assert s['scientific_parameters_changed'] is False and s['additional_evaluation'] is False
        assert s['additional_experiment_cells'] == 0 and s['other_projects_modified'] is False
        assert s['source_retired'] is True and s['source_paused'] is False
        assert s['source_canonical_path_vacated'] is True and s['original_evidence_preserved'] is True
        assert s['retired_source_full_checkpoint_SHA_verified'] is True and s['source_wrapper_exit_code'] != 0
        assert s['retired_source_full_checkpoint'].startswith(s['retired_source_archive'] + '/')
        assert s['full_replay_rng_optimizers_and_finite_state_verified'] is True
        assert hashlib.sha256((base / 'handoff_Point_5080_to_4080_MPS_20261021.py').read_bytes()).hexdigest() == s['controller_sha256']
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == s['plan']['accounting_module_sha256']
        assert s['plan']['accounting_hook_integrated_before_any_transition'] is True
        assert s['plan']['regular_gpu_status'] == 'available_for_rebalance'
        assert s['plan']['regular_frame_at_unix'] - s['plan']['candidate_since_unix'] >= 1800
        src, dst = s['source_identity'], s['target_identity']
        assert src['boot_id'] == '5f46bb0c-b4ec-4802-8f6e-01ac611c0f2f'
        assert dst['boot_id'] == '3b362c03-4ed3-4e03-aea4-cf7339356026'
        assert src['device'] == dst['device'] == 'cuda:0'
        assert src['cuda_visible_devices'] == dst['cuda_visible_devices'] == '0'
        assert all(src[k] == s['plan']['expected_source_identity'][k] for k in ('pid', 'start_ticks', 'boot_id', 'device'))
        assert s['runtime_change'] == {'from_torch': '2.7.1+cu128', 'to_torch': '2.7.1+cu118',
                                       'from_backend': 'private_MPS', 'to_backend': 'private_MPS',
                                       'cuda_build_changed': True, 'bitwise_equivalence_claimed': False}
        assert s['source_MPS_proof']['connection_verified'] is True
        assert s['source_MPS_proof']['gpu_uuid'] == 'GPU-ac315b09-0bb7-2072-efd1-cdfc5fae96a5'
        assert s['source_MPS_proof']['client_pid'] == src['pid']
        proof = s['target_MPS_proof']
        assert proof['connection_verified'] is True and proof['gpu_uuid'] == GPU
        assert proof['client_pid'] == dst['pid'] and dst['pid'] in proof['client_pids']
        for cap in (s['target_capacity_before_pause'], s['target_capacity_at_launch'], s['target_capacity_immediately_before_client_launch']):
            assert cap['GPU']['uuid'] == GPU and cap['GPU']['free_mib'] >= 8192
            assert cap['GPU']['utilization_percent'] <= 85 and cap['GPU']['foreign_gpu_processes'] == 0
            assert cap['CPU']['cpu_affinity_count'] >= 24 and cap['CPU']['actual_cpu_idle_fraction'] >= .4
            assert cap['CPU']['memory_available_bytes'] >= 8*1024**3
        keep = s['input_preservation']
        assert keep['host_input_sha256'] == keep['archive_input_sha256'] == s['checkpoint_sha256']
        assert keep['archive_separate_4T_filesystem_verified'] is True
        assert hashlib.sha256(Path(keep['host_input_path']).read_bytes()).hexdigest() == s['checkpoint_sha256']
        steps = s['last_logged_source_steps'] - s['resume_steps']
        cost = s['last_logged_source_cost'] - s['resume_cumulative_training_cost']
        assert 0 <= steps <= 5000 and steps == s['known_repeated_environment_steps_lower_bound']
        assert math.isfinite(cost) and cost >= 0 and cost == s['known_additional_training_cost_lower_bound']
        assert s['unknown_unlogged_source_tail_not_zero'] is True
        assert s['first_verified_target_steps'] >= s['resume_steps'] + 5000
        native = json.loads((base / s['target_first_epoch_native_observation']).read_text())
        w = next(w for w in native['training'] if w['job'] == s['job'])
        assert (w['pid'], w['start_ticks'], w['boot_id']) == (dst['pid'], dst['start_ticks'], dst['boot_id'])
        assert w['mps_gpu_uuid_verified'] == GPU and w['cuda_execution_backend'] == 'mps_verified'
        assert w['steps'] >= s['first_verified_target_steps']
        records.append({'job': s['job'], 'runtime_placement': 'existing_Point_5080_to_4080_private_MPS',
                        'receipt': s, 'first_target_native_observation': native,
                        'known_additional_physical_steps': steps,
                        'known_additional_physical_training_cost': cost,
                        'unknown_unlogged_tail_not_zero': True})
    return records
