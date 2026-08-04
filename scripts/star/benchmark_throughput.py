#!/usr/bin/env python3
"""Benchmark end-to-end SACLag throughput across process/vector-env configurations.

Each worker pre-fills replay, performs a real update warm-up, waits at a file barrier,
then measures rollout + backward updates + periodic evaluation.  For E vector
environments, the worker performs E optimizer updates per vector step and scales the
OmniSafe per-environment batch so that the global batch remains constant.  Therefore
the measured update-to-data ratio is explicitly UTD=1 for every matrix entry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import psutil


DEFAULT_CONFIGS = ('1x1', '1x4', '2x2', '3x1', '4x1')
CSV_FIELDS = (
    'captured_at_utc',
    'git_commit',
    'env_id',
    'processes',
    'vector_envs_per_process',
    'requested_duration_s',
    'measurement_wall_s',
    'warmup_s',
    'aggregate_env_steps',
    'aggregate_updates',
    'aggregate_env_fps',
    'aggregate_updates_per_s',
    'actual_utd',
    'global_batch_size',
    'evaluation_episodes',
    'gpu_util_mean_pct',
    'gpu_util_peak_pct',
    'gpu_vram_peak_mib',
    'worker_cuda_peak_mib',
    'gpu_temperature_peak_c',
    'gpu_power_mean_w',
    'cpu_mean_pct_of_quota',
    'cpu_peak_pct_of_quota',
    'worker_rss_peak_mib',
    'cgroup_ram_peak_gib',
    'cost_per_mstep_cny',
    'min_actor_parameter_delta_linf',
    'resource_samples',
    'valid',
    'notes',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


def read_number(path: Path) -> float | None:
    try:
        value = path.read_text(encoding='utf-8').strip()
    except OSError:
        return None
    return None if value == 'max' else float(value)


def cpu_quota() -> float:
    try:
        quota_text, period_text = (
            Path('/sys/fs/cgroup/cpu.max')
            .read_text(
                encoding='utf-8',
            )
            .split()
        )
    except (OSError, ValueError):
        return float(psutil.cpu_count() or 1)
    return (
        float(psutil.cpu_count() or 1)
        if quota_text == 'max'
        else float(quota_text) / float(period_text)
    )


def run_worker(args: argparse.Namespace) -> None:  # pylint: disable=too-many-locals
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(args.safe_root / 'cache/torchinductor'))
    os.environ.setdefault('WANDB_DIR', str(args.safe_root / 'runs/active'))

    result_path = args.worker_result
    payload: dict[str, Any] = {
        'status': 'failed',
        'worker_index': args.worker_index,
        'seed': args.seed,
        'env_id': args.env_id,
        'vector_envs': args.vector_envs,
        'started_at_utc': utc_now(),
    }
    algo: Any = None
    try:
        import torch

        import omnisafe

        if args.global_batch_size % args.vector_envs:
            raise ValueError('global batch size must be divisible by vector env count')
        per_env_batch = args.global_batch_size // args.vector_envs
        steps_per_epoch = 1000 * args.vector_envs
        buffer_vector_slots = max(20_000, 100_000 // args.vector_envs)
        cost_limit = 10.0 if args.env_id.startswith('SafetyPoint') else 0.0
        custom_cfgs = {
            'seed': args.seed,
            'train_cfgs': {
                'device': args.device,
                'torch_threads': 1,
                'vector_env_nums': args.vector_envs,
                'parallel': 1,
                'total_steps': steps_per_epoch * 1_000_000,
                'eval_episodes': 1,
            },
            'algo_cfgs': {
                'steps_per_epoch': steps_per_epoch,
                'update_cycle': 1,
                'update_iters': args.vector_envs,
                'size': buffer_vector_slots,
                'batch_size': per_env_batch,
                'start_learning_steps': 0,
                'reward_normalize': False,
                'cost_normalize': False,
                'obs_normalize': False,
            },
            'logger_cfgs': {
                'log_dir': str(args.worker_log_dir),
                'use_tensorboard': False,
                'use_wandb': False,
                'save_model_freq': 1_000_000,
            },
            'lagrange_cfgs': {'cost_limit': cost_limit},
        }
        wrapper = omnisafe.Agent('SACLag', args.env_id, custom_cfgs=custom_cfgs)
        algo = wrapper.agent

        # One full vector episode gives SACLag a finite cost statistic and fills replay.
        for _ in range(args.prefill_vector_steps):
            algo._env.rollout(  # pylint: disable=protected-access
                rollout_step=1,
                agent=algo._actor_critic,
                buffer=algo._buf,
                logger=algo._logger,
                use_rand_action=True,
            )

        warmup_deadline = time.monotonic() + args.warmup_seconds
        warmup_updates = 0
        while time.monotonic() < warmup_deadline:
            algo._env.rollout(  # pylint: disable=protected-access
                rollout_step=1,
                agent=algo._actor_critic,
                buffer=algo._buf,
                logger=algo._logger,
                use_rand_action=False,
            )
            before = algo._update_count
            algo._update()
            warmup_updates += algo._update_count - before
        algo._env.eval_policy(1, algo._actor_critic, algo._logger)

        parameter_before = next(algo._actor_critic.actor.parameters()).detach().clone()
        ready_path = args.ready_dir / f'worker-{args.worker_index}.ready.json'
        atomic_json(
            ready_path,
            {
                'worker_index': args.worker_index,
                'warmup_updates': warmup_updates,
                'ready_at_utc': utc_now(),
            },
        )
        barrier_deadline = time.monotonic() + args.barrier_timeout_seconds
        while not args.start_marker.exists():
            if time.monotonic() >= barrier_deadline:
                raise TimeoutError('timed out waiting for benchmark start barrier')
            time.sleep(0.1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        measurement_started = time.monotonic()
        measurement_deadline = measurement_started + args.duration_seconds
        env_steps = 0
        updates = 0
        evaluation_episodes = 0
        next_evaluation = args.eval_interval_transitions

        while time.monotonic() < measurement_deadline:
            algo._env.rollout(  # pylint: disable=protected-access
                rollout_step=1,
                agent=algo._actor_critic,
                buffer=algo._buf,
                logger=algo._logger,
                use_rand_action=False,
            )
            env_steps += args.vector_envs
            before = algo._update_count
            algo._update()
            updates += algo._update_count - before
            if env_steps >= next_evaluation:
                algo._env.eval_policy(1, algo._actor_critic, algo._logger)
                evaluation_episodes += 1
                next_evaluation += args.eval_interval_transitions

        if evaluation_episodes == 0:
            algo._env.eval_policy(1, algo._actor_critic, algo._logger)
            evaluation_episodes = 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.monotonic() - measurement_started
        parameter_after = next(algo._actor_critic.actor.parameters()).detach()
        parameter_delta = (parameter_after - parameter_before).abs().max().item()
        payload.update(
            {
                'status': 'passed',
                'finished_at_utc': utc_now(),
                'elapsed_seconds': elapsed,
                'env_steps': env_steps,
                'updates': updates,
                'actual_utd': updates / env_steps,
                'env_fps': env_steps / elapsed,
                'updates_per_second': updates / elapsed,
                'evaluation_episodes': evaluation_episodes,
                'global_batch_size': args.global_batch_size,
                'per_env_batch_size': per_env_batch,
                'actor_parameter_delta_linf': parameter_delta,
                'peak_cuda_memory_mib': (
                    torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
                ),
                'log_dir': algo._logger.log_dir,
            },
        )
        if not math.isclose(payload['actual_utd'], 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f'worker UTD drifted from 1: {payload["actual_utd"]}')
        if parameter_delta <= 0.0:
            raise RuntimeError('actor parameters did not change during measured backward updates')
    except Exception as error:  # pylint: disable=broad-exception-caught
        payload.update(
            {
                'status': 'failed',
                'finished_at_utc': utc_now(),
                'error': f'{type(error).__name__}: {error}',
                'traceback': traceback.format_exc(),
            },
        )
        atomic_json(result_path, payload)
        raise
    finally:
        if algo is not None:
            with suppress(Exception):
                algo._logger.close()
            with suppress(Exception):
                algo._env.close()

    atomic_json(result_path, payload)


def parse_matrix_config(value: str) -> tuple[int, int]:
    match = re.fullmatch(r'([1-9][0-9]*)x([1-9][0-9]*)', value)
    if match is None:
        raise ValueError(f'invalid matrix configuration {value!r}; expected PxE, for example 2x2')
    return int(match.group(1)), int(match.group(2))


def gpu_sample() -> dict[str, float] | None:
    result = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits',
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    values = [float(value.strip()) for value in result.stdout.splitlines()[0].split(',')]
    return dict(zip(('gpu_util', 'vram_mib', 'temperature_c', 'power_w'), values, strict=True))


def processes_for(roots: list[psutil.Process]) -> list[psutil.Process]:
    found: dict[int, psutil.Process] = {}
    for root in roots:
        try:
            candidates = [root, *root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        for process in candidates:
            found[process.pid] = process
    return list(found.values())


def resource_sample(roots: list[psutil.Process], quota_cpus: float) -> dict[str, float]:
    processes = processes_for(roots)
    cpu_sum = 0.0
    rss_sum = 0
    for process in processes:
        try:
            cpu_sum += process.cpu_percent(interval=None)
            rss_sum += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    sample = {
        'cpu_pct_of_quota': cpu_sum / quota_cpus,
        'worker_rss_mib': rss_sum / 2**20,
        'cgroup_ram_gib': (read_number(Path('/sys/fs/cgroup/memory.current')) or 0.0) / 2**30,
    }
    gpu = gpu_sample()
    if gpu is not None:
        sample.update(gpu)
    return sample


def mean(samples: list[dict[str, float]], key: str) -> float:
    values = [sample[key] for sample in samples if key in sample]
    return fmean(values) if values else float('nan')


def peak(samples: list[dict[str, float]], key: str) -> float:
    values = [sample[key] for sample in samples if key in sample]
    return max(values) if values else float('nan')


def git_commit() -> str:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


def terminate_workers(workers: list[subprocess.Popen[str]]) -> None:
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
    deadline = time.monotonic() + 20.0
    while any(worker.poll() is None for worker in workers) and time.monotonic() < deadline:
        time.sleep(0.2)
    for worker in workers:
        if worker.poll() is None:
            worker.kill()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_recommendations(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if str(row.get('valid', '')).lower() == 'true']
    by_env: dict[str, dict[str, Any]] = {}
    for row in valid:
        env_id = str(row['env_id'])
        if env_id not in by_env or float(row['aggregate_env_fps']) > float(
            by_env[env_id]['aggregate_env_fps'],
        ):
            by_env[env_id] = row

    def best_family(point: bool) -> dict[str, Any] | None:
        candidates = [row for row in valid if str(row['env_id']).startswith('SafetyPoint') is point]
        return (
            max(candidates, key=lambda row: float(row['aggregate_env_fps'])) if candidates else None
        )

    def compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            'env_id': row['env_id'],
            'processes': int(row['processes']),
            'vector_envs_per_process': int(row['vector_envs_per_process']),
            'aggregate_env_fps': float(row['aggregate_env_fps']),
            'aggregate_updates_per_s': float(row['aggregate_updates_per_s']),
            'cost_per_mstep_cny': float(row['cost_per_mstep_cny']),
        }

    payload = {
        'generated_at_utc': utc_now(),
        'selection_rule': 'maximum valid aggregate end-to-end FPS at UTD=1',
        'point': compact(best_family(True)),
        'mujoco': compact(best_family(False)),
        'by_environment': {env_id: compact(row) for env_id, row in sorted(by_env.items())},
    }
    atomic_json(csv_path.with_name('recommended_concurrency.json'), payload)


def run_matrix(  # pylint: disable=too-many-locals,too-many-statements
    args: argparse.Namespace,
) -> None:
    configs = [parse_matrix_config(value) for value in args.configs]
    csv_rows = load_csv(args.csv) if args.append else []
    run_root = args.output_dir / args.run_id
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f'run directory is not empty: {run_root}')
    run_root.mkdir(parents=True, exist_ok=True)
    quota_cpus = cpu_quota()
    current_commit = git_commit()

    for config_index, (process_count, vector_envs) in enumerate(configs):
        config_name = f'P{process_count}_E{vector_envs}'
        config_dir = run_root / config_name
        ready_dir = config_dir / 'barrier'
        start_marker = ready_dir / 'start.json'
        ready_dir.mkdir(parents=True, exist_ok=True)
        workers: list[subprocess.Popen[str]] = []
        logs: list[Any] = []
        print(f'[{utc_now()}] launching {args.env_id} {config_name}', flush=True)
        try:
            for worker_index in range(process_count):
                worker_dir = config_dir / f'worker-{worker_index}'
                worker_dir.mkdir(parents=True, exist_ok=True)
                log_stream = (worker_dir / 'worker.log').open('w', encoding='utf-8')
                logs.append(log_stream)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    '--worker',
                    '--env-id',
                    args.env_id,
                    '--device',
                    args.device,
                    '--safe-root',
                    str(args.safe_root),
                    '--worker-index',
                    str(worker_index),
                    '--vector-envs',
                    str(vector_envs),
                    '--seed',
                    str(args.seed + config_index * 100 + worker_index),
                    '--global-batch-size',
                    str(args.global_batch_size),
                    '--prefill-vector-steps',
                    str(args.prefill_vector_steps),
                    '--warmup-seconds',
                    str(args.warmup_seconds),
                    '--duration-seconds',
                    str(args.duration_seconds),
                    '--eval-interval-transitions',
                    str(args.eval_interval_transitions),
                    '--barrier-timeout-seconds',
                    str(args.ready_timeout_seconds),
                    '--ready-dir',
                    str(ready_dir),
                    '--start-marker',
                    str(start_marker),
                    '--worker-result',
                    str(worker_dir / 'result.json'),
                    '--worker-log-dir',
                    str(worker_dir / 'omnisafe'),
                ]
                worker_env = os.environ.copy()
                worker_env.update(
                    {
                        'OMP_NUM_THREADS': '1',
                        'MKL_NUM_THREADS': '1',
                        'OPENBLAS_NUM_THREADS': '1',
                        'PYTHONUNBUFFERED': '1',
                        'TORCHINDUCTOR_CACHE_DIR': str(args.safe_root / 'cache/torchinductor'),
                        'WANDB_DIR': str(args.safe_root / 'runs/active'),
                    },
                )
                workers.append(
                    subprocess.Popen(
                        command,
                        stdout=log_stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=worker_env,
                    ),
                )

            ready_deadline = time.monotonic() + args.ready_timeout_seconds
            ready_paths = [
                ready_dir / f'worker-{index}.ready.json' for index in range(process_count)
            ]
            while not all(path.exists() for path in ready_paths):
                failed = [worker.returncode for worker in workers if worker.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f'worker exited before barrier with return codes {failed}')
                if time.monotonic() >= ready_deadline:
                    raise TimeoutError(f'workers did not reach barrier for {config_name}')
                time.sleep(1.0)

            roots = [psutil.Process(worker.pid) for worker in workers]
            for process in processes_for(roots):
                with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    process.cpu_percent(interval=None)
            atomic_json(start_marker, {'start_at_utc': utc_now(), 'config': config_name})
            measurement_wall_started = time.monotonic()
            samples: list[dict[str, float]] = []
            hard_deadline = (
                measurement_wall_started + args.duration_seconds + args.ready_timeout_seconds
            )
            while any(worker.poll() is None for worker in workers):
                samples.append(resource_sample(roots, quota_cpus))
                if time.monotonic() >= hard_deadline:
                    raise TimeoutError(f'workers exceeded measurement deadline for {config_name}')
                time.sleep(args.sample_interval_seconds)
            samples.append(resource_sample(roots, quota_cpus))
            measurement_wall = time.monotonic() - measurement_wall_started
            return_codes = [worker.wait() for worker in workers]
            if any(return_codes):
                raise RuntimeError(f'workers failed for {config_name}: {return_codes}')

            results = [
                json.loads((config_dir / f'worker-{index}/result.json').read_text(encoding='utf-8'))
                for index in range(process_count)
            ]
            if any(result.get('status') != 'passed' for result in results):
                raise RuntimeError(f'invalid worker result for {config_name}')
            aggregate_steps = sum(int(result['env_steps']) for result in results)
            aggregate_updates = sum(int(result['updates']) for result in results)
            denominator = max(float(result['elapsed_seconds']) for result in results)
            aggregate_fps = aggregate_steps / denominator
            aggregate_updates_s = aggregate_updates / denominator
            actual_utd = aggregate_updates / aggregate_steps
            minimum_delta = min(float(result['actor_parameter_delta_linf']) for result in results)
            valid = (
                denominator >= args.duration_seconds
                and math.isclose(actual_utd, 1.0, rel_tol=0.0, abs_tol=1e-9)
                and minimum_delta > 0.0
                and sum(int(result['evaluation_episodes']) for result in results) >= process_count
            )
            row = {
                'captured_at_utc': utc_now(),
                'git_commit': current_commit,
                'env_id': args.env_id,
                'processes': process_count,
                'vector_envs_per_process': vector_envs,
                'requested_duration_s': args.duration_seconds,
                'measurement_wall_s': measurement_wall,
                'warmup_s': args.warmup_seconds,
                'aggregate_env_steps': aggregate_steps,
                'aggregate_updates': aggregate_updates,
                'aggregate_env_fps': aggregate_fps,
                'aggregate_updates_per_s': aggregate_updates_s,
                'actual_utd': actual_utd,
                'global_batch_size': args.global_batch_size,
                'evaluation_episodes': sum(
                    int(result['evaluation_episodes']) for result in results
                ),
                'gpu_util_mean_pct': mean(samples, 'gpu_util'),
                'gpu_util_peak_pct': peak(samples, 'gpu_util'),
                'gpu_vram_peak_mib': peak(samples, 'vram_mib'),
                'worker_cuda_peak_mib': max(
                    float(result['peak_cuda_memory_mib']) for result in results
                ),
                'gpu_temperature_peak_c': peak(samples, 'temperature_c'),
                'gpu_power_mean_w': mean(samples, 'power_w'),
                'cpu_mean_pct_of_quota': mean(samples, 'cpu_pct_of_quota'),
                'cpu_peak_pct_of_quota': peak(samples, 'cpu_pct_of_quota'),
                'worker_rss_peak_mib': peak(samples, 'worker_rss_mib'),
                'cgroup_ram_peak_gib': peak(samples, 'cgroup_ram_gib'),
                'cost_per_mstep_cny': args.hourly_rate * 1_000_000 / (3600 * aggregate_fps),
                'min_actor_parameter_delta_linf': minimum_delta,
                'resource_samples': len(samples),
                'valid': valid,
                'notes': 'rollout+backward+evaluation; UTD=1; global batch fixed',
            }
            csv_rows.append(row)
            write_csv(args.csv, csv_rows)
            write_recommendations(args.csv, csv_rows)
            atomic_json(config_dir / 'aggregate.json', row)
            print(
                f'[{utc_now()}] {config_name}: {aggregate_fps:.2f} env FPS, '
                f'{aggregate_updates_s:.2f} update/s, valid={valid}',
                flush=True,
            )
            if not valid:
                raise RuntimeError(f'benchmark validity checks failed for {config_name}')
        finally:
            terminate_workers(workers)
            for log_stream in logs:
                log_stream.close()
        if args.cooldown_seconds > 0 and config_index + 1 < len(configs):
            time.sleep(args.cooldown_seconds)

    print(json.dumps({'csv': str(args.csv), 'rows': len(csv_rows)}, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--env-id', default='SafetyPointGoal1-v0')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--safe-root',
        type=Path,
        default=Path(os.environ.get('STAR_SAFE_ROOT', '/root/autodl-tmp/SAFE')),
    )
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--global-batch-size', type=int, default=256)
    parser.add_argument('--prefill-vector-steps', type=int, default=1000)
    parser.add_argument('--warmup-seconds', type=float, default=60.0)
    parser.add_argument('--duration-seconds', type=float, default=600.0)
    parser.add_argument('--eval-interval-transitions', type=int, default=5000)
    parser.add_argument('--ready-timeout-seconds', type=float, default=1800.0)
    parser.add_argument('--sample-interval-seconds', type=float, default=1.0)
    parser.add_argument('--cooldown-seconds', type=float, default=5.0)
    parser.add_argument('--hourly-rate', type=float, default=1.48)
    parser.add_argument('--configs', nargs='+', default=list(DEFAULT_CONFIGS))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--append', action='store_true')

    parser.add_argument('--worker-index', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--vector-envs', type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        '--barrier-timeout-seconds',
        type=float,
        default=1800.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--ready-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--start-marker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-result', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-log-dir', type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker:
        required = (args.ready_dir, args.start_marker, args.worker_result, args.worker_log_dir)
        if any(value is None for value in required):
            parser.error('internal worker paths are required')
        run_worker(args)
        return
    if args.output_dir is None:
        parser.error('--output-dir is required')
    if args.csv is None:
        args.csv = args.output_dir / 'throughput_benchmark.csv'
    if args.duration_seconds <= 0 or args.warmup_seconds < 0:
        parser.error('duration must be positive and warmup non-negative')
    if args.global_batch_size <= 0:
        parser.error('--global-batch-size must be positive')
    run_matrix(args)


if __name__ == '__main__':
    main()
