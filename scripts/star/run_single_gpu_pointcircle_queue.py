#!/usr/bin/env python3
"""Run the preregistered PointCircle screen and conditional formal STAR matrix."""

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
EXPECTED_PROTOCOL_SHA256 = '0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8'
EXPECTED_MATRIX_SHA256 = '0c4fe4b21cd086bde6e703c486022716056c844b5c3b0d79a05a062b784aac88'
PROTOCOL_RELATIVE = Path('reports/star/single_gpu_paper_core_preregistration.json')
MATRIX_RELATIVE = Path('configs/star/single_gpu_paper_core.yaml')
ARMS = ('C0', 'P47')
CONFIG_RELATIVE = {
    'C0': Path('configs/star/single_gpu_pointcircle_c0.yaml'),
    'P47': Path('configs/star/single_gpu_pointcircle_p47.yaml'),
}
FORMAL_SEEDS = (2027, 2028, 2029)
SCREEN_STEPS = 300_000
FORMAL_STEPS = 500_000
REPLAY_SIZE = 1_000_000


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


def validate_source(source: Path) -> dict[str, Any]:
    """Require clean source and the locked global protocol/matrix identities."""
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError('PointCircle training source must be clean')
    commit = subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    protocol_sha = sha256_file(source / PROTOCOL_RELATIVE)
    matrix_sha = sha256_file(source / MATRIX_RELATIVE)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256 or matrix_sha != EXPECTED_MATRIX_SHA256:
        raise ValueError('PointCircle queue does not match the frozen single-GPU protocol')
    return {
        'source_commit': commit,
        'protocol_sha256': protocol_sha,
        'matrix_sha256': matrix_sha,
        'config_sha256': {
            arm: sha256_file(source / relative) for arm, relative in CONFIG_RELATIVE.items()
        },
    }


def wait_for_prerequisite(path: Path, poll_seconds: int) -> None:
    """Wait for the PointGoal postprocessor's authenticated completion state."""
    while True:
        if path.is_file():
            payload = read_json(path)
            if (
                payload.get('state') == 'complete'
                and payload.get('protocol_sha256') == EXPECTED_PROTOCOL_SHA256
            ):
                manifest = Path(payload['final_manifest'])
                if read_json(manifest).get('status') != 'complete':
                    raise ValueError('PointGoal prerequisite has no valid final manifest')
                return
        time.sleep(poll_seconds)


def run_directory(root: Path, phase: str, arm: str, seed: int) -> Path:
    """Return one stable PointCircle run container."""
    return root / phase / f'{arm}-{seed}'


def receipt_complete(
    path: Path,
    arm: str,
    seed: int,
    steps: int,
    identities: dict[str, Any],
) -> bool:
    """Return whether one STAR receipt proves exact completion."""
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return (
        payload.get('status') == 'passed'
        and payload.get('algorithm') == 'STAR'
        and payload.get('environment') == ENVIRONMENT
        and int(payload.get('seed', -1)) == seed
        and int(payload.get('steps', -1)) == steps
        and payload.get('source_commit') == identities['source_commit']
        and payload.get('source_dirty') is False
        and payload.get('config_sha256') == identities['config_sha256'][arm]
    )


def unique_final_checkpoint(container: Path) -> Path:
    """Resolve exactly one final full replay checkpoint below a completed run."""
    matches = sorted(container.glob(f'training/STAR-{{{ENVIRONMENT}}}/seed-*/checkpoints/final_full.pt'))
    if len(matches) != 1:
        raise ValueError(f'expected one final_full.pt below {container}, found {len(matches)}')
    return matches[0]


def training_command(
    args: argparse.Namespace,
    identities: dict[str, Any],
    phase: str,
    arm: str,
    seed: int,
    steps: int,
    resume: Path | None,
) -> tuple[list[str], Path, Path]:
    """Build one exact STAR learner command and its receipt/log paths."""
    container = run_directory(args.run_root, phase, arm, seed)
    receipt = container / 'receipt.json'
    log = container / 'train.log'
    command = [
        str(args.python_bin),
        str(args.source / 'scripts/star/train_star.py'),
        '--config',
        str(args.source / CONFIG_RELATIVE[arm]),
        '--seed',
        str(seed),
        '--steps',
        str(steps),
        '--steps-per-epoch',
        '5000',
        '--device',
        'cuda:0',
        '--vector-envs',
        '1',
        '--replay-size',
        str(REPLAY_SIZE),
        '--checkpoint-replay',
        '--full-checkpoint-freq',
        '250000',
        '--log-dir',
        str(container / 'training'),
        '--result',
        str(receipt),
        '--no-tensorboard',
    ]
    if resume is not None:
        command.extend(['--resume', str(resume)])
    if identities['config_sha256'][arm] != sha256_file(args.source / CONFIG_RELATIVE[arm]):
        raise ValueError(f'{arm} config changed after queue validation')
    return command, receipt, log


def start_process(command: list[str], log: Path, source: Path) -> tuple[subprocess.Popen[str], Any]:
    """Start one learner with frozen single-process resource settings."""
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
            'PYTHONPATH': str(source),
        },
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, stream


def run_pair(  # pylint: disable=too-many-arguments
    args: argparse.Namespace,
    identities: dict[str, Any],
    phase: str,
    seed: int,
    steps: int,
    resumes: dict[str, Path | None],
) -> dict[str, Any]:
    """Run C0/P47 concurrently, stopping the counterpart on an engineering failure."""
    pending: list[tuple[str, list[str], Path, Path]] = []
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        command, receipt, log = training_command(
            args,
            identities,
            phase,
            arm,
            seed,
            steps,
            resumes[arm],
        )
        if receipt_complete(receipt, arm, seed, steps, identities):
            rows.append({'arm': arm, 'receipt': str(receipt), 'state': 'already_complete'})
        else:
            pending.append((arm, command, receipt, log))
    started_at = utc_now()
    started = time.monotonic()
    running: dict[str, tuple[subprocess.Popen[str], Any, Path, Path]] = {}
    for arm, command, receipt, log in pending:
        process, stream = start_process(command, log, args.source)
        running[arm] = (process, stream, receipt, log)
    failures: list[str] = []
    while running:
        for arm, (process, stream, receipt, log) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            del running[arm]
            if return_code or not receipt_complete(receipt, arm, seed, steps, identities):
                failures.append(f'{arm} exit={return_code} log={log}')
            else:
                rows.append({'arm': arm, 'receipt': str(receipt), 'state': 'completed'})
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
        'phase': phase,
        'seed': seed,
        'steps_per_method': steps,
        'started_at_utc': started_at,
        'finished_at_utc': utc_now(),
        'elapsed_seconds': time.monotonic() - started,
        'rows': sorted(rows, key=lambda row: row['arm']),
        'protocol_sha256': identities['protocol_sha256'],
    }
    atomic_json(args.run_root / phase / f'pair-seed-{seed}.json', timing)
    if failures:
        raise RuntimeError('; '.join(failures))
    return timing


def run_screen(args: argparse.Namespace, identities: dict[str, Any]) -> dict[str, Any]:
    """Train and evaluate the paired seed-2027 300k screen."""
    run_pair(
        args,
        identities,
        'screen',
        2027,
        SCREEN_STEPS,
        {arm: None for arm in ARMS},
    )
    decision_path = args.run_root / 'screen_decision.json'
    command = [
        str(args.python_bin),
        str(args.source / 'scripts/star/decide_pointcircle_screen.py'),
        '--source',
        str(args.source),
        '--screen-root',
        str(args.run_root / 'screen'),
        '--protocol-sha256',
        identities['protocol_sha256'],
        '--output',
        str(decision_path),
    ]
    with (args.run_root / 'screen-decision.log').open('a', encoding='utf-8') as stream:
        subprocess.run(
            command,
            cwd=args.source,
            env={
                **os.environ,
                'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONPATH': str(args.source),
            },
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    decision = read_json(decision_path)
    if decision.get('status') != 'complete':
        raise RuntimeError('PointCircle screen decision is incomplete')
    return decision


def run_formal(args: argparse.Namespace, identities: dict[str, Any]) -> None:
    """Continue seed 2027 and run fresh seeds 2028/2029 to 500k."""
    screen_resumes = {
        arm: unique_final_checkpoint(run_directory(args.run_root, 'screen', arm, 2027))
        for arm in ARMS
    }
    run_pair(args, identities, 'formal', 2027, FORMAL_STEPS, screen_resumes)
    for seed in (2028, 2029):
        run_pair(
            args,
            identities,
            'formal',
            seed,
            FORMAL_STEPS,
            {arm: None for arm in ARMS},
        )


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source', type=Path, required=True)
    result.add_argument('--run-root', type=Path, required=True)
    result.add_argument('--python-bin', type=Path, required=True)
    result.add_argument('--prerequisite-state', type=Path, required=True)
    result.add_argument('--poll-seconds', type=int, default=30)
    return result


def main() -> None:
    """Wait for PointGoal, run the screen, and conditionally finish PointCircle."""
    args = parser().parse_args()
    for field in ('source', 'run_root', 'python_bin', 'prerequisite_state'):
        setattr(args, field, getattr(args, field).resolve())
    if args.poll_seconds <= 0:
        raise ValueError('poll interval must be positive')
    identities = validate_source(args.source)
    args.run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_root / 'queue_identity.json',
        {
            **identities,
            'environment': ENVIRONMENT,
            'screen_steps': SCREEN_STEPS,
            'formal_steps': FORMAL_STEPS,
            'maximum_concurrent_learners': 2,
        },
    )
    wait_for_prerequisite(args.prerequisite_state, args.poll_seconds)
    decision = run_screen(args, identities)
    if not bool(decision['decision']['screen_pass']):
        atomic_json(
            args.run_root / 'queue_state.json',
            {
                'state': 'screen_not_passed',
                'finished_at_utc': utc_now(),
                'protocol_sha256': identities['protocol_sha256'],
                'screen_decision': str(args.run_root / 'screen_decision.json'),
            },
        )
        return
    run_formal(args, identities)
    atomic_json(
        args.run_root / 'queue_state.json',
        {
            'state': 'complete',
            'finished_at_utc': utc_now(),
            'protocol_sha256': identities['protocol_sha256'],
            'screen_decision': str(args.run_root / 'screen_decision.json'),
            'formal_seeds': list(FORMAL_SEEDS),
        },
    )


if __name__ == '__main__':
    main()
