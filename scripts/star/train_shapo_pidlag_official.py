#!/usr/bin/env python3
"""Run pinned official SHAPO-PIDLag while capturing evaluation-critical state.

The official optimizer and environment path are used unchanged.  This wrapper only
locks resource arguments, runs W&B offline, keeps CUDA diagnostics loggable, and
preserves the observation running statistics that the upstream script otherwise
omits from its final checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_SHAPO_COMMIT = '1bb770437bbbe40a89938661343744243e164bf9'
SUPPORTED_ENVIRONMENTS = ('SafetyPointGoal1-v0', 'SafetyPointCircle1-v0')
CUDA_DIAGNOSTIC_KEYS = frozenset(
    {
        'Misc/CosineSimilarity',
        'Misc/EffectiveRho',
        'Misc/ScaleAlongGrad',
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def last_progress_row(path: Path) -> dict[str, str]:
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f'official SHAPO progress is empty: {path}')
    return rows[-1]


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--official-source', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--phase', choices=('throughput', 'formal'), required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    return parser.parse_known_args()


def git_identity(source: Path) -> tuple[str, str]:
    commit = subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, status


def main() -> None:  # pylint: disable=too-many-locals
    wrapper_args, official_argv = parse_wrapper_args()
    official_source = wrapper_args.official_source.resolve()
    commit, status = git_identity(official_source)
    if commit != EXPECTED_SHAPO_COMMIT:
        raise RuntimeError(f'expected SHAPO {EXPECTED_SHAPO_COMMIT}, observed {commit}')
    if status:
        raise RuntimeError('official SHAPO source must be clean')

    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('WANDB_MODE', 'offline')
    os.environ.setdefault('WANDB_SILENT', 'true')
    sys.path.insert(0, str(official_source))
    sys.path.insert(0, str(official_source / 'safepo' / 'single_agent'))
    sys.argv = [sys.argv[0], *official_argv]

    import numpy as np
    import torch
    import wandb

    from safepo.single_agent import trpo_pid
    from safepo.utils.config import single_agent_args

    args, cfg_env = single_agent_args()
    required = {
        'cost_limit': 10.0,
        'device': 'cuda',
        'device_id': 0,
        'num_envs': 1,
        'perturbation_target_kl': 0.00001,
        'use_eval': False,
        'use_shapo_actor': True,
    }
    observed = {key: getattr(args, key) for key in required}
    if observed != required:
        raise RuntimeError(f'official SHAPO arguments do not match the frozen protocol: {observed}')
    if args.task not in SUPPORTED_ENVIRONMENTS:
        raise RuntimeError(f'unsupported frozen Point-task environment: {args.task}')
    if args.total_steps <= 0 or args.total_steps % args.steps_per_epoch:
        raise RuntimeError('SHAPO total steps must be positive and divisible by steps per epoch')

    base_log_dir = Path(args.log_dir).resolve()
    run_log_dir = (
        base_log_dir
        / args.experiment
        / args.task
        / 'trpo_pid'
        / f'seed-{args.seed}'
    )
    if run_log_dir.exists() and any(run_log_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite an existing SHAPO run: {run_log_dir}')
    run_log_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = str(run_log_dir)

    captured_training_envs: list[Any] = []
    original_make_env = trpo_pid.make_sa_safetygym_env

    def capture_make_env(*capture_args: Any, **capture_kwargs: Any) -> Any:
        result = original_make_env(*capture_args, **capture_kwargs)
        if not captured_training_envs:
            captured_training_envs.append(result[0])
        return result

    trpo_pid.make_sa_safetygym_env = capture_make_env
    original_set_num_threads = torch.set_num_threads
    original_set_num_threads(1)

    def keep_one_torch_thread(_requested: int) -> None:
        original_set_num_threads(1)

    trpo_pid.torch.set_num_threads = keep_one_torch_thread

    # Upstream passes three scalar CUDA tensors directly to NumPy's logger,
    # which raises before the first epoch can be written.  Detaching only these
    # diagnostic copies fixes serialization; the tensors used by the optimizer
    # and the environment remain untouched.
    original_logger_store = trpo_pid.EpochLogger.store

    def store_cuda_diagnostics(logger: Any, add_value: bool = False, **values: Any) -> None:
        for key in CUDA_DIAGNOSTIC_KEYS.intersection(values):
            value = values[key]
            if torch.is_tensor(value):
                host_value = value.detach().cpu()
                values[key] = host_value.item() if host_value.numel() == 1 else host_value.numpy()
        original_logger_store(logger, add_value=add_value, **values)

    trpo_pid.EpochLogger.store = store_cuda_diagnostics
    run = wandb.init(
        project='safeflow2027-shapo-pidlag',
        name=f'{wrapper_args.phase}-seed-{args.seed}',
        dir=str(run_log_dir),
        config=vars(args),
        mode='offline',
        settings=wandb.Settings(_service_wait=60),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        trpo_pid.main(args, cfg_env)
    finally:
        run.finish()
    elapsed = time.monotonic() - started
    if not captured_training_envs:
        raise RuntimeError('failed to capture the official SHAPO training environment')

    train_env = captured_training_envs[0]
    if not hasattr(train_env, 'obs_rms'):
        raise RuntimeError('official SHAPO observation normalizer has no obs_rms state')
    normalizer_path = run_log_dir / 'obs_normalizer.pt'
    torch.save(
        {
            'count': float(train_env.obs_rms.count),
            'mean': torch.as_tensor(np.asarray(train_env.obs_rms.mean)).cpu(),
            'var': torch.as_tensor(np.asarray(train_env.obs_rms.var)).cpu(),
        },
        normalizer_path,
    )

    checkpoint = run_log_dir / 'policy.pt'
    progress = run_log_dir / 'progress.csv'
    if not checkpoint.is_file() or not progress.is_file():
        raise RuntimeError('official SHAPO run did not produce its final policy and progress')
    final_row = last_progress_row(progress)
    total_cost = float(final_row['Metrics/TotalCost'])
    payload = {
        'algorithm': 'SHAPO-PIDLag',
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256_file(checkpoint),
        'elapsed_seconds': elapsed,
        'end_to_end_fps': args.total_steps / elapsed,
        'environment': args.task,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'instrumentation': {
            'cuda_diagnostic_logging_compatibility': sorted(CUDA_DIAGNOSTIC_KEYS),
            'observation_normalizer_preserved': True,
            'optimizer_or_environment_semantics_changed': False,
            'torch_threads_locked': 1
        },
        'last_progress_row': final_row,
        'learner_vector_envs': args.num_envs,
        'normalizer': str(normalizer_path),
        'normalizer_sha256': sha256_file(normalizer_path),
        'official_arguments': vars(args),
        'official_source_commit': commit,
        'phase': wrapper_args.phase,
        'protocol_sha256': wrapper_args.protocol_sha256,
        'seed': args.seed,
        'source_commit': wrapper_args.source_commit,
        'status': 'complete',
        'steps': args.total_steps,
        'training_cumulative_cost': total_cost,
    }
    if torch.cuda.is_available():
        payload['peak_cuda_memory_mib'] = torch.cuda.max_memory_allocated() / 2**20
    atomic_json(wrapper_args.result, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
