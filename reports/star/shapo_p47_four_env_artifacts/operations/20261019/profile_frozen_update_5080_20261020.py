#!/usr/bin/env python3
"""Profile a disposable checkpoint copy's updates; zero simulator/training env steps."""
import functools
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ['PYTHONNOUSERSITE'] = '1'
os.sched_setaffinity(0, set(range(16)))

ROOT = Path('/home/autodrive/LZQW/SAFEFLOW2027')
INPUT = ROOT / 'runs/shapo_p47_four_env/runs/point-goal-2028/training/STAR-{SafetyPointGoal1-v0}/seed-2028-2026-10-19-23-39-24/checkpoints/latest_full.pt'
EXPECTED_SHA = '754b15ee61bac6329c48615efb2d42b6c60d77a4475af595f6c822a239fbcb62'
WORK = ROOT / 'operations/performance_20261020'


def main():
    import torch
    import yaml
    import omnisafe

    assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == EXPECTED_SHA
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
    torch.set_num_threads(1)
    output = Path(tempfile.mkdtemp(prefix='disposable_update_profile_', dir=WORK))
    config = yaml.safe_load((ROOT / 'runs/shapo_p47_four_env/runs/point-goal-2028/resolved_formal_config.yaml').read_text())
    custom = config['custom_cfgs']
    custom['seed'] = 2028
    custom.setdefault('train_cfgs', {}).update(device='cuda:0', torch_threads=1,
        vector_env_nums=1, parallel=1, total_steps=10000000)
    custom.setdefault('algo_cfgs', {}).update(size=400000, batch_size=256,
        steps_per_epoch=5000, resume_from=str(INPUT))
    custom.setdefault('logger_cfgs', {}).update(log_dir=str(output / 'disposable_logs'),
        use_tensorboard=False, use_wandb=False)
    wrapper = omnisafe.Agent('STAR', config['env_id'], custom_cfgs=custom)
    actor = wrapper.agent
    initial_env_steps = actor.env_steps
    initial_run_env_steps = actor.run_env_steps
    initial_updates = actor.update_count
    for _ in range(2):
        actor._update()
    torch.cuda.synchronize()

    phases = {}
    def instrument(obj, name):
        original = getattr(obj, name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            with torch.profiler.record_function('phase::' + name):
                result = original(*args, **kwargs)
            phases.setdefault(name, []).append(time.perf_counter() - started)
            return result
        setattr(obj, name, wrapped)
    for method in ('_update_cost_critic', '_update_violation_critic', '_sample_reward_batch',
                   '_update_reward_critics', '_update_actor'):
        instrument(actor, method)
    instrument(actor._buf, 'sample_batch')
    instrument(actor._actor_critic, 'polyak_update')
    started = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(4):
            actor._update()
        torch.cuda.synchronize()
    wall = time.perf_counter() - started
    assert actor.env_steps == initial_env_steps and actor.run_env_steps == initial_run_env_steps
    assert actor.update_count == initial_updates + 6
    events = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
    result = {
        'kind': 'disposable_frozen_update_performance_profile_not_scientific_result',
        'observed_at_unix': time.time(), 'input': str(INPUT), 'input_sha256': EXPECTED_SHA,
        'source_commit': '62ca7399357c10645a955dcbead7ed2153a82c69',
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
        'warmup_updates': 2, 'profiled_updates': 4, 'profile_wall_seconds': wall,
        'simulator_environment_steps': 0, 'formal_trainers_modified_or_restarted': False,
        'formal_checkpoints_or_counts_modified': False,
        'temporary_optimizer_updates_not_formal_experiment_data': 6,
        'phase_wall_seconds': {k: {'calls': len(v), 'total': sum(v), 'mean': sum(v)/len(v)} for k,v in phases.items()},
        'top_self_cpu_events': [{'name': e.key, 'calls': e.count,
            'self_cpu_total_ms': e.self_cpu_time_total/1000,
            'cpu_total_ms': e.cpu_time_total/1000,
            'self_device_total_ms': getattr(e, 'self_device_time_total', 0)/1000}
            for e in events[:30]],
        'output_directory': str(output)
    }
    (output / 'profile.json').write_text(json.dumps(result, indent=2) + '\n')
    print('SAFEFLOW_PROFILE_RESULT=' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
