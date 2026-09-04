#!/usr/bin/env python3
"""Run PointCircle SACLag/SHAPO-PIDLag after the preregistered STAR screen passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENVIRONMENT = 'SafetyPointCircle1-v0'
METHODS = ('SACLag', 'SHAPO-PIDLag')
SEEDS = (2027, 2028, 2029)
FORMAL_STEPS = 500_000
REPLAY_SIZE = 1_000_000
EXPECTED_PROTOCOL_SHA256 = '0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8'
EXPECTED_MATRIX_SHA256 = '0c4fe4b21cd086bde6e703c486022716056c844b5c3b0d79a05a062b784aac88'
EXPECTED_SHAPO_COMMIT = '1bb770437bbbe40a89938661343744243e164bf9'
PROTOCOL_RELATIVE = Path('reports/star/single_gpu_paper_core_preregistration.json')
MATRIX_RELATIVE = Path('configs/star/single_gpu_paper_core.yaml')


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} does not contain a JSON object')
    return payload


def git_output(source: Path, *arguments: str) -> str:
    """Run one read-only Git identity command."""
    return subprocess.run(
        ['git', '-C', str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_sources(source: Path, official_source: Path) -> dict[str, str]:
    """Require clean pinned sources and unchanged global protocol identities."""
    if git_output(source, 'status', '--porcelain'):
        raise ValueError('SAFEFLOW2027 PointCircle prior source must be clean')
    if git_output(official_source, 'status', '--porcelain'):
        raise ValueError('official SHAPO source must be clean')
    official_commit = git_output(official_source, 'rev-parse', 'HEAD')
    if official_commit != EXPECTED_SHAPO_COMMIT:
        raise ValueError(f'expected SHAPO {EXPECTED_SHAPO_COMMIT}, observed {official_commit}')
    protocol_sha = sha256_file(source / PROTOCOL_RELATIVE)
    matrix_sha = sha256_file(source / MATRIX_RELATIVE)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256 or matrix_sha != EXPECTED_MATRIX_SHA256:
        raise ValueError('PointCircle prior queue does not match the frozen single-GPU protocol')
    return {
        'source_commit': git_output(source, 'rev-parse', 'HEAD'),
        'official_source_commit': official_commit,
        'protocol_sha256': protocol_sha,
        'matrix_sha256': matrix_sha,
    }


def wait_for_pointcircle(path: Path, poll_seconds: int) -> bool:
    """Wait for STAR completion, returning false only for the preregistered screen failure."""
    while True:
        if path.is_file():
            payload = read_json(path)
            if payload.get('protocol_sha256') != EXPECTED_PROTOCOL_SHA256:
                raise ValueError('PointCircle prerequisite has the wrong protocol identity')
            state = payload.get('state')
            if state == 'complete':
                return True
            if state == 'screen_not_passed':
                return False
        time.sleep(poll_seconds)


def run_directory(root: Path, method: str, seed: int) -> Path:
    """Return one stable method/seed directory."""
    return root / 'formal' / method.lower().replace('-', '_') / f'seed-{seed}'


def receipt_complete(
    path: Path,
    method: str,
    seed: int,
    identities: dict[str, str],
) -> bool:
    """Return whether one receipt proves exact formal completion."""
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return (
        payload.get('status') == 'complete'
        and payload.get('phase') == 'formal'
        and payload.get('algorithm') == method
        and payload.get('environment') == ENVIRONMENT
        and int(payload.get('seed', -1)) == seed
        and int(payload.get('steps', -1)) == FORMAL_STEPS
        and payload.get('protocol_sha256') == identities['protocol_sha256']
        and payload.get('source_commit') == identities['source_commit']
        and (
            method != 'SHAPO-PIDLag'
            or payload.get('official_source_commit') == identities['official_source_commit']
        )
    )


def command_for(
    args: argparse.Namespace,
    identities: dict[str, str],
    method: str,
    seed: int,
) -> tuple[list[str], Path, Path]:
    """Build one exact 500k PointCircle prior-baseline command."""
    target = run_directory(args.run_root, method, seed)
    receipt = target / 'receipt.json'
    log = target / 'training.log'
    common = [
        '--result',
        str(receipt),
        '--phase',
        'formal',
        '--source-commit',
        identities['source_commit'],
        '--protocol-sha256',
        identities['protocol_sha256'],
    ]
    if method == 'SACLag':
        command = [
            str(args.python_bin),
            str(args.source / 'scripts/star/train_pointgoal_saclag_baseline.py'),
            '--environment',
            ENVIRONMENT,
            '--seed',
            str(seed),
            '--steps',
            str(FORMAL_STEPS),
            '--steps-per-epoch',
            '2000',
            '--replay-size',
            str(REPLAY_SIZE),
            '--device',
            'cuda:0',
            '--log-dir',
            str(target / 'training'),
            *common,
        ]
    else:
        command = [
            str(args.python_bin),
            str(args.source / 'scripts/star/train_shapo_pidlag_official.py'),
            '--official-source',
            str(args.official_shapo_source),
            *common,
            '--seed',
            str(seed),
            '--task',
            ENVIRONMENT,
            '--num-envs',
            '1',
            '--total-steps',
            str(FORMAL_STEPS),
            '--steps-per-epoch',
            '20000',
            '--device',
            'cuda',
            '--device-id',
            '0',
            '--use-eval',
            'False',
            '--use-shapo-actor',
            'True',
            '--perturbation-target-kl',
            '0.00001',
            '--cost-limit',
            '10.0',
            '--experiment',
            'formal_pointcircle',
            '--log-dir',
            str(target / 'training'),
        ]
    return command, receipt, log


def start_process(command: list[str], log: Path, source: Path) -> tuple[subprocess.Popen[str], Any]:
    """Start one learner under the frozen single-process resource settings."""
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open('a', encoding='utf-8')
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        command,
        cwd=source,
        env={
            **os.environ,
            'CUDA_VISIBLE_DEVICES': '0',
            'MKL_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'PYTHONDONTWRITEBYTECODE': '1',
            'WANDB_MODE': 'offline',
            'WANDB_SILENT': 'true',
        },
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, stream


def run_pair(
    args: argparse.Namespace,
    identities: dict[str, str],
    seed: int,
) -> dict[str, Any]:
    """Run one two-method wave and fail fast on engineering errors."""
    pending: list[tuple[str, list[str], Path, Path]] = []
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        command, receipt, log = command_for(args, identities, method, seed)
        if receipt_complete(receipt, method, seed, identities):
            rows.append({'method': method, 'receipt': str(receipt), 'state': 'already_complete'})
        else:
            pending.append((method, command, receipt, log))
    started = time.monotonic()
    running: dict[str, tuple[subprocess.Popen[str], Any, Path, Path]] = {}
    for method, command, receipt, log in pending:
        process, stream = start_process(command, log, args.source)
        running[method] = (process, stream, receipt, log)
    failures: list[str] = []
    while running:
        for method, (process, stream, receipt, log) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            del running[method]
            if return_code or not receipt_complete(receipt, method, seed, identities):
                failures.append(f'{method} exit={return_code} log={log}')
            else:
                rows.append({'method': method, 'receipt': str(receipt), 'state': 'completed'})
        if failures and running:
            for process, _, _, _ in running.values():
                process.terminate()
            for process, stream, _, _ in running.values():
                process.wait(timeout=30)
                stream.close()
            running.clear()
        elif running:
            time.sleep(5)
    timing = {
        'seed': seed,
        'steps_per_method': FORMAL_STEPS,
        'elapsed_seconds': time.monotonic() - started,
        'rows': sorted(rows, key=lambda row: row['method']),
        'protocol_sha256': identities['protocol_sha256'],
    }
    atomic_json(args.run_root / 'formal' / f'pair-seed-{seed}.json', timing)
    if failures:
        raise RuntimeError('; '.join(failures))
    return timing


def parser() -> argparse.ArgumentParser:
    """Build queue arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source', type=Path, required=True)
    result.add_argument('--official-shapo-source', type=Path, required=True)
    result.add_argument('--run-root', type=Path, required=True)
    result.add_argument('--python-bin', type=Path, required=True)
    result.add_argument('--pointcircle-state', type=Path, required=True)
    result.add_argument('--poll-seconds', type=int, default=30)
    return result


def main() -> None:
    """Wait for the STAR gate and conditionally run the exact 2x3 prior matrix."""
    args = parser().parse_args()
    for field in ('source', 'official_shapo_source', 'run_root', 'python_bin', 'pointcircle_state'):
        setattr(args, field, getattr(args, field).resolve())
    if args.poll_seconds <= 0:
        raise ValueError('poll interval must be positive')
    identities = validate_sources(args.source, args.official_shapo_source)
    args.run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_root / 'queue_identity.json',
        {
            **identities,
            'environment': ENVIRONMENT,
            'methods': list(METHODS),
            'seeds': list(SEEDS),
            'steps_per_run': FORMAL_STEPS,
            'maximum_concurrent_learners': 2,
            'vector_env_nums': 1,
            'saclag_utd_ratio': 1.0,
        },
    )
    if not wait_for_pointcircle(args.pointcircle_state, args.poll_seconds):
        atomic_json(
            args.run_root / 'queue_state.json',
            {
                'state': 'screen_not_passed',
                'finished_at_utc': utc_now(),
                'protocol_sha256': identities['protocol_sha256'],
                'scientific_training_launched': False,
            },
        )
        return
    for seed in SEEDS:
        run_pair(args, identities, seed)
    atomic_json(
        args.run_root / 'queue_state.json',
        {
            'state': 'complete',
            'finished_at_utc': utc_now(),
            'protocol_sha256': identities['protocol_sha256'],
            'environment': ENVIRONMENT,
            'methods': list(METHODS),
            'seeds': list(SEEDS),
            'steps_per_run': FORMAL_STEPS,
        },
    )


if __name__ == '__main__':
    main()
