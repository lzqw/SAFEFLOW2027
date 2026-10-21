#!/usr/bin/env python3
"""Publish existing capacity/log facts without new remote observations."""
import hashlib
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = Path('/home/lzqw/SAFEFLOW2027_delivery/training-status-20261019/reports/star/shapo_p47_four_env_artifacts/operations/20261019')
NAME = 'Point_shared_GPU_slowdown_context_20261021_0922.json'


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def main():
    assert not (OUT / NAME).exists() and not (BASE / NAME).exists()
    samples = {}
    for stamp in ('0822', '0852', '0922'):
        raw = read(BASE / ('capacity_progress_observation_20261021_' + stamp + '.json'))
        gpu, = raw['hosts']['4080']['gpus']
        assert gpu['safeflow_gpu_processes'] == 3 and not gpu['MPS_accounting_unproven']
        samples[stamp] = {'native_observation': 'ordinary_observation_20261021_' + stamp + '.json',
                          'GPU_total_utilization_percent': gpu['utilization_percent'],
                          'foreign_or_unknown_GPU_pids': gpu['foreign_pids'],
                          'verified_SAFEFLOW_MPS_clients': gpu['safeflow_mps_client_pids']}
    assert not samples['0822']['foreign_or_unknown_GPU_pids']
    assert len(samples['0852']['foreign_or_unknown_GPU_pids']) == 3
    assert not samples['0922']['foreign_or_unknown_GPU_pids']
    proof_name = 'CPU_4080_Point_slowdown_20261021_0922.json'
    proof = read(BASE / proof_name)
    rows = proof['remote_observation']['workers']
    assert len(rows) == 8 and len({r['pid'] for r in rows}) == 8
    points = {}
    for w in rows:
        if not w['job'].startswith('point-'):
            continue
        last = w['recent_progress_rows'][-1]
        points[w['job']] = {'last_logged_steps': float(last['TotalEnvSteps']),
                            'last_logged_epoch_FPS': float(last['Time/FPS']),
                            'native_4M_file_in_current_segment_at_sample': w['native_4M_file_exists']}
    status = read(OUT / 'training_status_20261021_0922.json')
    slow = next(w for w in status['active_training'] if w['job'] == 'point-button-2028')
    diag = {'kind': 'shared_GPU_runtime_context_not_a_scientific_result',
            'native_capacity_frames': samples,
            'one_shot_OS_log_proof': proof_name,
            'one_shot_OS_log_proof_sha256': hashlib.sha256((BASE / proof_name).read_bytes()).hexdigest(),
            'most_recent_bounded_log_rows': points,
            'interpretation': [
                'A transient multi-client slowdown overlaps with three foreign-or-unknown GPU processes at08:52, absent at09:22. This supports resource-contention as an explanation, not a controlled causal measurement.',
                'Two Point clients have subsequent18-20FPS complete epochs. Button2028 has not yet logged a subsequent complete epoch in this observation; its recovery is not assumed.',
                'The08:52 totalGPU100percent was not SAFEFLOW-only utilization or proof of useful SAFEFLOW throughput.',
                'Eight existing learners each used approximately one CPU core in a one-second sample. This is not evidence of sustained utilization or permission to oversubscribe additional workers.',
                'The raw last-epoch Button2028 estimate remains7.86days in the09:22 snapshot. It extrapolates a contention-affected epoch, not a reliable steady-state deadline. Wait for normal scheduled frames rather than replace measured FPS with a preferred number.',
                'Button2028 native4M JSON was not yet present in its current segment at this sample. No extra evaluation was launched and no task was declared stopped.'
            ],
            'raw_last_epoch_Button2028_training_only_days': slow['estimated_training_only_days_at_last_epoch_fps'],
            'additional_remote_observations_by_this_publisher': 0,
            'GPU_queries_by_diagnostic': 0, 'training_restarts': 0,
            'foreign_processes_signaled_or_modified': False,
            'scientific_parameters_or_budgets_changed': False, 'additional_evaluation_rollouts': 0,
            'original_full_parent_gaps_repaired': False, 'final_scientific_audit_executed': False}
    for destination in (BASE / NAME, OUT / NAME):
        write_new(destination, diag)
    for name in (proof_name, Path(__file__).name, 'observe_Point_slowdown_4080_20261021_0922.py'):
        assert not (OUT / name).exists()
        shutil.copyfile(BASE / name, OUT / name)
    root_path = OUT / 'training_status.json'
    root = read(root_path)
    assert root['superseded_by'] == 'training_status_20261021_0922.json'
    root['latest_runtime_contention_context'] = NAME
    part = root_path.with_suffix('.runtime_context.tmp')
    with part.open('x') as stream:
        stream.write(json.dumps(root, indent=2) + '\n')
    part.replace(root_path)
    print(json.dumps({'published_context': NAME, 'training_restarts': 0, 'extra_rollouts': 0}))


if __name__ == '__main__':
    main()
