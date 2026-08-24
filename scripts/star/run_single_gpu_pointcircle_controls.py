#!/usr/bin/env python3
"""Run the two PointCircle mechanism controls and one 0.25 parameter point."""

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
SEED = 2027
STEPS = 300_000
REPLAY_SIZE = 1_000_000
EXPECTED_PROTOCOL_SHA256 = '0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8'
PROTOCOL_RELATIVE = Path('reports/star/single_gpu_paper_core_preregistration.json')
CONFIGS = {
    'Iso-Low': Path('configs/star/single_gpu_pointcircle_iso_low.yaml'),
    'Random-Aniso': Path('configs/star/single_gpu_pointcircle_random_aniso.yaml'),
    'Safety-Aniso-0.25': Path('configs/star/single_gpu_pointcircle_ratio_025.yaml'),
}


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


def validate_source(source: Path) -> dict[str, Any]:
    """Require clean source, frozen protocol, and exact control configs."""
    if git_output(source, 'status', '--porcelain'):
        raise ValueError('PointCircle control source must be clean')
    protocol_sha = sha256_file(source / PROTOCOL_RELATIVE)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise ValueError('PointCircle controls do not match the frozen protocol')
    return {
        'source_commit': git_output(source, 'rev-parse', 'HEAD'),
        'protocol_sha256': protocol_sha,
        'config_sha256': {label: sha256_file(source / path) for label, path in CONFIGS.items()},
    }


def wait_for_prerequisite(path: Path, poll_seconds: int) -> bool:
    """Wait for the four-method PointCircle result or its registered screen failure."""
    while True:
        if path.is_file():
            payload = read_json(path)
            if payload.get('protocol_sha256') != EXPECTED_PROTOCOL_SHA256:
                raise ValueError('control prerequisite has a different protocol identity')
            state = payload.get('state')
            if state == 'complete':
                manifest = Path(payload['final_manifest'])
                if read_json(manifest).get('status') != 'complete':
                    raise ValueError('control prerequisite final manifest is invalid')
                return True
            if state == 'screen_not_passed':
                return False
        time.sleep(poll_seconds)


def run_directory(root: Path, label: str) -> Path:
    """Return one stable control run directory."""
    phase = 'parameter' if label == 'Safety-Aniso-0.25' else 'mechanism'
    return root / phase / f'{label}-{SEED}'


def receipt_complete(
    path: Path,
    label: str,
    identities: dict[str, Any],
) -> bool:
    """Return whether one receipt proves exact control completion."""
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
        and int(payload.get('seed', -1)) == SEED
        and int(payload.get('steps', -1)) == STEPS
        and payload.get('source_commit') == identities['source_commit']
        and payload.get('source_dirty') is False
        and payload.get('config_sha256') == identities['config_sha256'][label]
    )


def command_for(
    args: argparse.Namespace,
    identities: dict[str, Any],
    label: str,
) -> tuple[list[str], Path, Path]:
    """Build one exact control learner command."""
    target = run_directory(args.run_root, label)
    receipt = target / 'receipt.json'
    log = target / 'training.log'
    if identities['config_sha256'][label] != sha256_file(args.source / CONFIGS[label]):
        raise ValueError(f'{label} config changed after queue validation')
    command = [
        str(args.python_bin),
        str(args.source / 'scripts/star/train_star.py'),
        '--config',
        str(args.source / CONFIGS[label]),
        '--seed',
        str(SEED),
        '--steps',
        str(STEPS),
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
        str(STEPS),
        '--log-dir',
        str(target / 'training'),
        '--result',
        str(receipt),
        '--no-tensorboard',
    ]
    return command, receipt, log


def start(command: list[str], log: Path, source: Path) -> tuple[subprocess.Popen[str], Any]:
    """Start one control learner with the frozen single-process resources."""
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


def run_wave(
    args: argparse.Namespace,
    identities: dict[str, Any],
    labels: tuple[str, ...],
) -> None:
    """Run one one- or two-learner control wave with engineering fail-fast."""
    if not 1 <= len(labels) <= 2:
        raise ValueError('control wave must contain one or two learners')
    running: dict[str, tuple[subprocess.Popen[str], Any, Path, Path]] = {}
    completed: list[str] = []
    for label in labels:
        command, receipt, log = command_for(args, identities, label)
        if receipt_complete(receipt, label, identities):
            completed.append(label)
        else:
            process, stream = start(command, log, args.source)
            running[label] = (process, stream, receipt, log)
    failures: list[str] = []
    while running:
        for label, (process, stream, receipt, log) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            del running[label]
            if return_code or not receipt_complete(receipt, label, identities):
                failures.append(f'{label} exit={return_code} log={log}')
            else:
                completed.append(label)
        if failures and running:
            for process, _, _, _ in running.values():
                process.terminate()
            for process, stream, _, _ in running.values():
                process.wait(timeout=30)
                stream.close()
            running.clear()
        elif running:
            time.sleep(5)
    atomic_json(
        args.run_root / f'wave-{labels[0].lower().replace("-", "_")}.json',
        {
            'labels': list(labels),
            'completed': sorted(completed),
            'steps_per_run': STEPS,
            'protocol_sha256': identities['protocol_sha256'],
        },
    )
    if failures:
        raise RuntimeError('; '.join(failures))


def parser() -> argparse.ArgumentParser:
    """Build control-queue arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source', type=Path, required=True)
    result.add_argument('--run-root', type=Path, required=True)
    result.add_argument('--python-bin', type=Path, required=True)
    result.add_argument('--prerequisite-state', type=Path, required=True)
    result.add_argument('--poll-seconds', type=int, default=30)
    return result


def main() -> None:
    """Conditionally run exactly the preregistered three new single-seed controls."""
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
            'seed': SEED,
            'steps_per_run': STEPS,
            'mechanism_arms': ['Iso-Low', 'Random-Aniso'],
            'parameter_arm': 'Safety-Aniso-0.25',
            'maximum_concurrent_learners': 2,
        },
    )
    if not wait_for_prerequisite(args.prerequisite_state, args.poll_seconds):
        atomic_json(
            args.run_root / 'queue_state.json',
            {
                'state': 'screen_not_passed',
                'finished_at_utc': utc_now(),
                'scientific_training_launched': False,
                'protocol_sha256': identities['protocol_sha256'],
            },
        )
        return
    run_wave(args, identities, ('Iso-Low', 'Random-Aniso'))
    run_wave(args, identities, ('Safety-Aniso-0.25',))
    atomic_json(
        args.run_root / 'queue_state.json',
        {
            'state': 'complete',
            'finished_at_utc': utc_now(),
            'protocol_sha256': identities['protocol_sha256'],
            'completed_controls': list(CONFIGS),
        },
    )


if __name__ == '__main__':
    main()
