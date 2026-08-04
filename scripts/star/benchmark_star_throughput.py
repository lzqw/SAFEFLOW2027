#!/usr/bin/env python3
"""Benchmark full STAR W3 updates at 1x1, 2x1, 3x1, and 4x1 concurrency."""

# Launchers intentionally share atomic receipt and commit-capture helpers.
# pylint: disable=duplicate-code

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


DEFAULT_CONFIGS = ('1x1', '2x1', '3x1', '4x1')
CSV_FIELDS = (
    'captured_at_utc',
    'git_commit',
    'env_id',
    'processes',
    'vector_envs_per_process',
    'requested_duration_s',
    'measurement_wall_s',
    'prefill_steps_per_worker',
    'warmup_steps_per_worker',
    'aggregate_env_steps',
    'aggregate_updates',
    'aggregate_env_fps',
    'aggregate_updates_per_s',
    'actual_utd',
    'global_batch_size_per_worker',
    'gpu_util_mean_pct',
    'gpu_util_peak_pct',
    'gpu_vram_peak_mib',
    'worker_cuda_peak_mib',
    'gpu_temperature_peak_c',
    'gpu_power_mean_w',
    'worker_cpu_peak_pct',
    'worker_rss_peak_mib',
    'cost_per_mstep_cny',
    'min_actor_parameter_delta_linf',
    'resource_samples',
    'valid',
    'notes',
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{json.dumps(payload, indent=2, sort_keys=True)}\n', encoding='utf-8')
    os.replace(temporary, path)


def git_commit(repo_root: Path) -> str:
    """Return the benchmarked worktree commit."""
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


def parse_config(value: str) -> tuple[int, int]:
    """Parse ``processes x vector-envs`` and enforce this benchmark's 1-env workers."""
    match = re.fullmatch(r'([1-9][0-9]*)x([1-9][0-9]*)', value)
    if match is None:
        raise ValueError(f'invalid concurrency configuration {value!r}')
    processes, vector_envs = int(match.group(1)), int(match.group(2))
    if vector_envs != 1:
        raise ValueError('STAR P3 throughput must use exactly one environment per process')
    return processes, vector_envs


def run_worker(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    args: argparse.Namespace,
) -> None:
    """Measure one synchronized process using the complete W3 update closure."""
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    payload: dict[str, Any] = {
        'status': 'failed',
        'worker_index': args.worker_index,
        'seed': args.seed,
        'started_at_utc': utc_now(),
    }
    algorithm: Any = None
    try:
        import torch  # pylint: disable=import-outside-toplevel

        import omnisafe  # pylint: disable=import-outside-toplevel

        custom = {
            'seed': args.seed,
            'train_cfgs': {
                'device': args.device,
                'torch_threads': 1,
                'vector_env_nums': 1,
                'parallel': 1,
                'total_steps': 1_000_000_000,
                'eval_episodes': 0,
            },
            'algo_cfgs': {
                'steps_per_epoch': 1_000_000,
                'size': max(20_000, args.prefill_steps * 4),
                'batch_size': args.batch_size,
                'random_steps': args.prefill_steps,
                'terminal_utility_steps': args.prefill_steps,
                'geometry_min_steps': args.prefill_steps,
                'eval_milestones': [],
                'milestone_checkpoints': False,
                'checkpoint_replay': False,
            },
            'logger_cfgs': {
                'log_dir': str(args.worker_log_dir),
                'use_tensorboard': False,
                'use_wandb': False,
            },
        }
        wrapper = omnisafe.Agent('STAR', args.env_id, custom_cfgs=custom)
        algorithm = wrapper.agent
        for _ in range(args.prefill_steps):
            algorithm._collect_once()  # pylint: disable=protected-access
        if algorithm.env_steps != args.prefill_steps or algorithm.update_count != 0:
            raise RuntimeError('random replay prefill changed the expected counters')

        # This is a workload benchmark, not an activation claim. Force W3 only after a real
        # random replay prefill so every measured update executes the full geometry path.
        algorithm._warmup.stage = 'W3'  # pylint: disable=protected-access
        algorithm._warmup.geometry_activation_step = (  # pylint: disable=protected-access
            algorithm.env_steps
        )
        for _ in range(args.warmup_steps):
            algorithm._collect_once()  # pylint: disable=protected-access

        atomic_json(
            args.ready_path,
            {
                'worker_index': args.worker_index,
                'ready_at_utc': utc_now(),
                'env_steps': algorithm.env_steps,
                'updates': algorithm.update_count,
            },
        )
        deadline = time.monotonic() + args.barrier_timeout_seconds
        while not args.start_marker.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('timed out waiting for the synchronized start marker')
            time.sleep(0.05)

        parameter_before = (
            next(
                algorithm._actor_critic.actor.parameters(),  # pylint: disable=protected-access
            )
            .detach()
            .clone()
        )
        env_before = algorithm.env_steps
        updates_before = algorithm.update_count
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        measurement_deadline = started + args.duration_seconds
        while time.monotonic() < measurement_deadline:
            algorithm._collect_once()  # pylint: disable=protected-access
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        env_steps = algorithm.env_steps - env_before
        updates = algorithm.update_count - updates_before
        parameter_after = next(
            algorithm._actor_critic.actor.parameters(),  # pylint: disable=protected-access
        ).detach()
        parameter_delta = float((parameter_after - parameter_before).abs().max().item())
        actual_utd = updates / max(env_steps, 1)
        if not math.isclose(actual_utd, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f'measured UTD drifted from 1: {actual_utd}')
        if parameter_delta <= 0.0:
            raise RuntimeError('actor parameters did not change during the measurement')
        payload.update(
            {
                'status': 'passed',
                'finished_at_utc': utc_now(),
                'elapsed_seconds': elapsed,
                'env_steps': env_steps,
                'updates': updates,
                'actual_utd': actual_utd,
                'env_fps': env_steps / elapsed,
                'updates_per_second': updates / elapsed,
                'batch_size': args.batch_size,
                'warmup_stage': algorithm._warmup.stage,  # pylint: disable=protected-access
                'actor_parameter_delta_linf': parameter_delta,
                'peak_cuda_memory_mib': (
                    torch.cuda.max_memory_allocated() / 2**20
                    if torch.cuda.is_available()
                    else 0.0
                ),
            },
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        payload.update(
            {
                'finished_at_utc': utc_now(),
                'error': f'{type(error).__name__}: {error}',
                'traceback': traceback.format_exc(),
            },
        )
        atomic_json(args.worker_result, payload)
        raise
    finally:
        if algorithm is not None:
            with suppress(Exception):
                algorithm._logger.close()  # pylint: disable=protected-access
            with suppress(Exception):
                algorithm._env.close()  # pylint: disable=protected-access
    atomic_json(args.worker_result, payload)


def gpu_sample() -> dict[str, float] | None:
    """Read one GPU telemetry sample."""
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
    if len(values) != 4:
        return None
    return {
        'gpu_util': values[0],
        'vram_mib': values[1],
        'temperature_c': values[2],
        'power_w': values[3],
    }


def process_sample(
    workers: list[subprocess.Popen[str]],
    tracked: dict[int, psutil.Process],
) -> dict[str, float]:
    """Measure aggregate worker CPU and resident memory."""
    for worker in workers:
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            root = psutil.Process(worker.pid)
            for process in (root, *root.children(recursive=True)):
                if process.pid not in tracked:
                    tracked[process.pid] = process
                    process.cpu_percent(interval=None)
    cpu = 0.0
    rss = 0
    for process in tracked.values():
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            cpu += process.cpu_percent(interval=None)
            rss += process.memory_info().rss
    result = {'worker_cpu_pct': cpu, 'worker_rss_mib': rss / 2**20}
    gpu = gpu_sample()
    if gpu is not None:
        result.update(gpu)
    return result


def mean(samples: list[dict[str, float]], key: str) -> float:
    """Return a telemetry mean or zero when unavailable."""
    values = [sample[key] for sample in samples if key in sample]
    return fmean(values) if values else 0.0


def peak(samples: list[dict[str, float]], key: str) -> float:
    """Return a telemetry maximum or zero when unavailable."""
    values = [sample[key] for sample in samples if key in sample]
    return max(values) if values else 0.0


def terminate_workers(workers: list[subprocess.Popen[str]]) -> None:
    """Stop only benchmark workers after a failed configuration."""
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
    deadline = time.monotonic() + 10.0
    while any(worker.poll() is None for worker in workers) and time.monotonic() < deadline:
        time.sleep(0.1)
    for worker in workers:
        if worker.poll() is None:
            worker.kill()


def wait_until_ready(
    workers: list[subprocess.Popen[str]],
    ready_paths: list[Path],
    timeout_seconds: float,
) -> None:
    """Wait for every worker's post-warmup barrier receipt."""
    deadline = time.monotonic() + timeout_seconds
    while not all(path.exists() for path in ready_paths):
        failures = [worker.returncode for worker in workers if worker.poll() not in (None, 0)]
        if failures:
            raise RuntimeError(f'worker exited before the barrier: {failures}')
        if time.monotonic() >= deadline:
            raise TimeoutError('workers did not reach the throughput barrier in time')
        time.sleep(0.1)


def run_configuration(  # pylint: disable=too-many-locals,too-many-statements
    args: argparse.Namespace,
    repo_root: Path,
    run_root: Path,
    config: str,
    commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one synchronized process-count configuration."""
    processes, vector_envs = parse_config(config)
    config_dir = run_root / config
    ready_dir = config_dir / 'ready'
    ready_dir.mkdir(parents=True, exist_ok=True)
    start_marker = config_dir / 'start.marker'
    workers: list[subprocess.Popen[str]] = []
    streams: list[Any] = []
    ready_paths: list[Path] = []
    result_paths: list[Path] = []
    script = Path(__file__).resolve()
    try:
        for worker_index in range(processes):
            worker_dir = config_dir / f'worker-{worker_index}'
            worker_dir.mkdir(parents=True, exist_ok=True)
            ready_path = ready_dir / f'worker-{worker_index}.json'
            result_path = worker_dir / 'result.json'
            stream = (worker_dir / 'worker.log').open('w', encoding='utf-8')
            command = [
                str(args.python),
                str(script),
                '--worker',
                '--worker-index',
                str(worker_index),
                '--worker-result',
                str(result_path),
                '--worker-log-dir',
                str(worker_dir / 'runs'),
                '--ready-path',
                str(ready_path),
                '--start-marker',
                str(start_marker),
                '--env-id',
                args.env_id,
                '--seed',
                str(args.seed + worker_index),
                '--device',
                args.device,
                '--batch-size',
                str(args.batch_size),
                '--prefill-steps',
                str(args.prefill_steps),
                '--warmup-steps',
                str(args.warmup_steps),
                '--duration-seconds',
                str(args.duration_seconds),
                '--barrier-timeout-seconds',
                str(args.barrier_timeout_seconds),
            ]
            workers.append(
                subprocess.Popen(  # pylint: disable=consider-using-with
                    command,
                    cwd=repo_root,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                ),
            )
            streams.append(stream)
            ready_paths.append(ready_path)
            result_paths.append(result_path)

        wait_until_ready(workers, ready_paths, args.barrier_timeout_seconds)
        start_marker.touch()
        samples: list[dict[str, float]] = []
        tracked_processes: dict[int, psutil.Process] = {}
        while any(worker.poll() is None for worker in workers):
            samples.append(process_sample(workers, tracked_processes))
            time.sleep(args.sample_interval_seconds)
        return_codes = [worker.wait() for worker in workers]
        if any(code != 0 for code in return_codes):
            tails = []
            for index in range(processes):
                log = config_dir / f'worker-{index}/worker.log'
                tails.append('\n'.join(log.read_text(encoding='utf-8').splitlines()[-30:]))
            raise RuntimeError(f'{config} workers failed with {return_codes}:\n' + '\n'.join(tails))

        results = [json.loads(path.read_text(encoding='utf-8')) for path in result_paths]
        if any(result['status'] != 'passed' for result in results):
            raise RuntimeError(f'{config} has a failed worker receipt')
        env_steps = sum(int(result['env_steps']) for result in results)
        updates = sum(int(result['updates']) for result in results)
        measurement_wall = max(float(result['elapsed_seconds']) for result in results)
        actual_utd = updates / env_steps
        aggregate_fps = env_steps / measurement_wall
        parameter_delta = min(float(result['actor_parameter_delta_linf']) for result in results)
        valid = (
            math.isclose(actual_utd, 1.0, rel_tol=0.0, abs_tol=1e-9)
            and parameter_delta > 0.0
            and all(result['warmup_stage'] == 'W3' for result in results)
        )
        row = {
            'captured_at_utc': utc_now(),
            'git_commit': commit,
            'env_id': args.env_id,
            'processes': processes,
            'vector_envs_per_process': vector_envs,
            'requested_duration_s': args.duration_seconds,
            'measurement_wall_s': measurement_wall,
            'prefill_steps_per_worker': args.prefill_steps,
            'warmup_steps_per_worker': args.warmup_steps,
            'aggregate_env_steps': env_steps,
            'aggregate_updates': updates,
            'aggregate_env_fps': aggregate_fps,
            'aggregate_updates_per_s': updates / measurement_wall,
            'actual_utd': actual_utd,
            'global_batch_size_per_worker': args.batch_size,
            'gpu_util_mean_pct': mean(samples, 'gpu_util'),
            'gpu_util_peak_pct': peak(samples, 'gpu_util'),
            'gpu_vram_peak_mib': peak(samples, 'vram_mib'),
            'worker_cuda_peak_mib': max(
                float(result['peak_cuda_memory_mib']) for result in results
            ),
            'gpu_temperature_peak_c': peak(samples, 'temperature_c'),
            'gpu_power_mean_w': mean(samples, 'power_w'),
            'worker_cpu_peak_pct': peak(samples, 'worker_cpu_pct'),
            'worker_rss_peak_mib': peak(samples, 'worker_rss_mib'),
            'cost_per_mstep_cny': args.hourly_rate_cny * 1_000_000 / (3600 * aggregate_fps),
            'min_actor_parameter_delta_linf': parameter_delta,
            'resource_samples': len(samples),
            'valid': valid,
            'notes': 'full_W3_forced_after_real_random_prefill_for_workload_measurement',
        }
        details = {'configuration': config, 'workers': results, 'telemetry': samples}
        atomic_json(config_dir / 'result.json', {'row': row, **details})
        if not valid:
            raise RuntimeError(f'{config} failed the fixed-batch, UTD, W3, or actor-update gate')
        return row, details
    finally:
        terminate_workers(workers)
        for stream in streams:
            stream.close()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write the public throughput table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    """Build parent and hidden worker arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--configs', nargs='+', default=list(DEFAULT_CONFIGS))
    result.add_argument('--env-id', default='SafetyPointGoal1-v0')
    result.add_argument('--seed', type=int, default=3027)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--batch-size', type=int, default=256)
    result.add_argument('--prefill-steps', type=int, default=256)
    result.add_argument('--warmup-steps', type=int, default=64)
    result.add_argument('--duration-seconds', type=float, default=30.0)
    result.add_argument('--sample-interval-seconds', type=float, default=0.5)
    result.add_argument('--barrier-timeout-seconds', type=float, default=300.0)
    result.add_argument('--hourly-rate-cny', type=float, default=1.48)
    result.add_argument('--python', type=Path, default=Path(sys.executable))
    result.add_argument('--output-dir', type=Path)
    result.add_argument('--csv', type=Path)
    result.add_argument('--summary', type=Path)
    result.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    result.add_argument('--worker-index', type=int, default=0, help=argparse.SUPPRESS)
    result.add_argument('--worker-result', type=Path, help=argparse.SUPPRESS)
    result.add_argument('--worker-log-dir', type=Path, help=argparse.SUPPRESS)
    result.add_argument('--ready-path', type=Path, help=argparse.SUPPRESS)
    result.add_argument('--start-marker', type=Path, help=argparse.SUPPRESS)
    return result


def main() -> None:  # pylint: disable=too-many-locals
    """Run a worker or the complete synchronized concurrency matrix."""
    args = parser().parse_args()
    if args.worker:
        required = (
            args.worker_result,
            args.worker_log_dir,
            args.ready_path,
            args.start_marker,
        )
        if any(value is None for value in required):
            raise ValueError('worker paths are required in worker mode')
        run_worker(args)
        return
    if args.output_dir is None or args.csv is None or args.summary is None:
        raise ValueError('--output-dir, --csv, and --summary are required')
    if (
        min(
            args.batch_size,
            args.prefill_steps,
            args.warmup_steps,
            args.duration_seconds,
            args.sample_interval_seconds,
        )
        <= 0
    ):
        raise ValueError('benchmark sizes and durations must be positive')

    repo_root = Path(__file__).resolve().parents[2]
    captured = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_root = args.output_dir / captured
    commit = git_commit(repo_root)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    started = time.monotonic()
    for config in args.configs:
        print(f'[{utc_now()}] benchmarking STAR {config}', flush=True)
        row, detail = run_configuration(args, repo_root, run_root, config, commit)
        rows.append(row)
        details.append(detail)
        write_csv(args.csv, rows)
        atomic_json(
            args.summary,
            {
                'status': 'running',
                'captured_at_utc': utc_now(),
                'git_commit': commit,
                'rows': rows,
            },
        )
        print(
            f'[{utc_now()}] {config}: {float(row["aggregate_env_fps"]):.2f} aggregate FPS',
            flush=True,
        )

    best = max(rows, key=lambda row: row['aggregate_env_fps'])
    payload = {
        'status': 'passed',
        'captured_at_utc': utc_now(),
        'git_commit': commit,
        'elapsed_seconds': time.monotonic() - started,
        'rows': rows,
        'details': details,
        'recommended_configuration': (f'{best["processes"]}x{best["vector_envs_per_process"]}'),
    }
    write_csv(args.csv, rows)
    atomic_json(args.summary, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
