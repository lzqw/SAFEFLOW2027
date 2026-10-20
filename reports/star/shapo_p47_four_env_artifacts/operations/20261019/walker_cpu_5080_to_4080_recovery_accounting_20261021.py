"""Strict accounting for existing Walker CPU placements; no scientific gate waiver."""
import hashlib
import json
import math
from pathlib import Path

SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
SIDE = 'walker_cpu_5080_to_4080_20261021'
SRC_BOOT = '5f46bb0c-b4ec-4802-8f6e-01ac611c0f2f'
DST_BOOT = '3b362c03-4ed3-4e03-aea4-cf7339356026'


def verify_Walker_CPU_5080_to_4080(base):
    base = Path(base)
    result = []
    for job, pid, ticks in [('walker2d-2028', 42764, 2320568), ('walker2d-2029', 44234, 2374389)]:
        prefix = job + '_' + SIDE
        path = base / (prefix + '_state.json')
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        if s.get('phase') != 'handoff_completed_first_CPU_epoch_verified':
            raise ValueError('Unresolved partial/failed CPU placement: ' + job)
        assert s['job'] == job and s['seed'] == int(job.rsplit('-', 1)[1])
        assert s['environment'] == 'Walker2d-v4' and s['source_commit'] == SOURCE
        assert s['source_host'] == '5080' and s['target_host'] == '4080'
        assert s['total_training_budget'] == 5000000
        assert s['scientific_parameters_changed'] is False and s['additional_evaluation'] is False
        assert s['additional_experiment_cells'] == 0 and s['no_other_projects_stopped'] is True
        assert s['GPU_capacity_not_claimed_or_used'] is True
        assert s['source_paused'] is False and s['source_retired'] is True
        assert s['source_canonical_path_vacated'] is True and s['original_evidence_preserved'] is True
        assert s['retired_source_archive'].startswith('/home/autodrive/LZQW/SAFEFLOW2027/recovery_backups/retired_cpu_workers/')
        assert s['retired_source_full_checkpoint'].startswith(s['retired_source_archive'] + '/')
        assert s['retired_source_full_checkpoint_SHA_verified'] is True
        assert s['source_wrapper_exit_code'] != 0
        assert s['full_replay_and_rng_present'] is True
        assert s['model_parameter_finiteness_verified'] is True
        assert s['full_optimizers_and_finite_model_verified'] is True
        sha = hashlib.sha256((base / 'handoff_Walker_CPU_5080_to_4080_20261021.py').read_bytes()).hexdigest()
        assert s['controller_sha256'] == s['plan']['controller_sha256'] == sha
        assert s['plan']['accounting_module_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        assert s['rpc_helper_sha256'] == hashlib.sha256((base / s['rpc_helper_filename']).read_bytes()).hexdigest()
        moved = json.loads((base / (prefix + '_receipt.json')).read_text())
        for key in ('checkpoint_sha256', 'resume_steps', 'resume_cumulative_training_cost',
                    'last_logged_source_steps', 'last_logged_source_cumulative_training_cost',
                    'known_repeated_environment_steps_lower_bound', 'known_additional_training_cost_lower_bound'):
            assert moved[key] == s[key]
        src, dst = s['source_identity'], s['target_identity']
        assert (src['pid'], src['start_ticks'], src['boot_id']) == (pid, ticks, SRC_BOOT)
        assert src['device'] == dst['device'] == 'cpu'
        assert src['cuda_visible_devices'] == dst['cuda_visible_devices'] == '-1'
        assert dst['boot_id'] == DST_BOOT
        runtime = s['runtime_change']
        assert runtime['from_device'] == runtime['to_device'] == 'cpu'
        assert runtime['from_torch'] == '2.7.1+cu128' and runtime['to_torch'] == '2.7.1+cu118'
        assert runtime['cuda_build_changed'] is True and runtime['bitwise_equivalence_claimed'] is False
        preservation = s['input_preservation']
        assert preservation['host_input_sha256'] == preservation['archive_input_sha256'] == s['checkpoint_sha256']
        assert preservation['archive_separate_4T_filesystem_verified'] is True
        assert hashlib.sha256(Path(preservation['host_input_path']).read_bytes()).hexdigest() == s['checkpoint_sha256']
        steps = s['last_logged_source_steps'] - s['resume_steps']
        cost = s['last_logged_source_cumulative_training_cost'] - s['resume_cumulative_training_cost']
        assert 0 <= steps <= 5000 and steps == s['known_repeated_environment_steps_lower_bound']
        assert math.isfinite(cost) and cost >= 0 and cost == s['known_additional_training_cost_lower_bound']
        assert s['unknown_unlogged_source_tail_not_zero'] is True
        for c in (s['target_capacity_before_pause'], s['target_capacity_at_launch']):
            assert c['boot_id'] == DST_BOOT and c['cpu_affinity_count'] >= 24
            assert c['load_1m'] < .5 * c['cpu_affinity_count'] and c['maximum_load_affinity_fraction'] == .5
            assert c['actual_cpu_idle_fraction'] >= .4 and c['cpu_idle_sample_seconds'] >= 1
            assert c['memory_available_bytes'] >= 8 * 1024**3 and c['project_free_bytes'] >= 5 * 1024**3
            assert c['minimum_cpu_quota_cores_or_null'] is None or c['minimum_cpu_quota_cores_or_null'] >= 8
            assert c['target_device'] == 'cpu' and c['target_cuda_visible_devices'] == '-1'
            assert c['GPU_capacity_not_claimed_or_used'] is True
        assert s['resume_steps'] + 5000 <= s['first_verified_target_steps'] <= 5000000
        assert s['first_verified_target_cost'] >= s['resume_cumulative_training_cost']
        native = json.loads((base / s['target_first_epoch_native_observation']).read_text())
        w = next(w for w in native['training'] if w['job'] == job)
        assert (w['pid'], w['start_ticks'], w['boot_id']) == (dst['pid'], dst['start_ticks'], DST_BOOT)
        assert w['device'] == 'cpu' and w['steps'] >= s['first_verified_target_steps']
        assert w['progress_csv'] == s['first_verified_target_csv']
        result.append({'job': job, 'runtime_placement': 'existing_Walker_CPU_5080_to_4080',
                       'receipt': s, 'first_target_native_observation': native,
                       'known_additional_physical_steps': steps,
                       'known_additional_physical_training_cost': cost,
                       'unknown_unlogged_tail_not_zero': True})
    return result
