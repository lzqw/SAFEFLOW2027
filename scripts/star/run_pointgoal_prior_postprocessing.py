#!/usr/bin/env python3
"""Wait for PointGoal prior training, then evaluate and finalize it unattended."""

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


PROTOCOL_RELATIVE = Path('reports/star/single_gpu_paper_core_preregistration.json')
EXPECTED_PROTOCOL_VERSION = 'star-single-gpu-paper-core-v1'
METHOD_DIRECTORIES = ('saclag', 'shapo_pidlag')
TRAINING_SEEDS = (2027, 2028, 2029)


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(source: Path) -> str:
    """Require a clean immutable source and return its commit."""
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError('postprocessing source must be clean')
    return subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} does not contain a JSON object')
    return payload


def wait_for_training(
    training_root: Path,
    protocol_sha256: str,
    poll_seconds: int,
    timeout_seconds: int,
) -> None:
    """Wait until the formal queue writes its authenticated completion state."""
    started = time.monotonic()
    state_path = training_root / 'queue_state.json'
    while True:
        if state_path.is_file():
            state = read_json(state_path)
            if (
                state.get('state') == 'complete'
                and state.get('mode') == 'formal'
                and state.get('protocol_sha256') == protocol_sha256
            ):
                return
            raise ValueError(f'invalid training queue completion state: {state}')
        if timeout_seconds and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f'training queue did not finish within {timeout_seconds} seconds')
        time.sleep(poll_seconds)


def formal_receipts(training_root: Path, method_directory: str) -> list[Path]:
    """Return the exact three formal receipt paths for one method."""
    if method_directory not in METHOD_DIRECTORIES:
        raise ValueError(f'unsupported method directory {method_directory}')
    return [
        training_root
        / 'formal_pointgoal'
        / method_directory
        / f'seed-{seed}'
        / 'receipt.json'
        for seed in TRAINING_SEEDS
    ]


def evaluation_command(
    args: argparse.Namespace,
    method_directory: str,
    protocol_sha256: str,
) -> tuple[list[str], Path]:
    """Build one method-level three-seed evaluator command."""
    output = args.output_root / 'evaluations' / method_directory
    command = [
        str(args.python_bin),
        str(args.source / 'scripts/star/evaluate_pointgoal_prior_baselines.py'),
        '--receipts',
        *(str(path) for path in formal_receipts(args.training_root, method_directory)),
        '--official-source',
        str(args.official_shapo_source),
        '--output-dir',
        str(output),
        '--expected-protocol-sha256',
        protocol_sha256,
    ]
    return command, output


def start_logged(command: list[str], log: Path, source: Path) -> tuple[subprocess.Popen[str], Any]:
    """Start one unbuffered CPU postprocessing subprocess."""
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open('a', encoding='utf-8')
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        command,
        cwd=source,
        env={
            **os.environ,
            'CUDA_VISIBLE_DEVICES': '',
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


def run_evaluations(args: argparse.Namespace, protocol_sha256: str) -> list[Path]:
    """Evaluate SACLag and SHAPO concurrently, with three seeds sequentially per method."""
    running: list[tuple[str, subprocess.Popen[str], Any, Path]] = []
    for method_directory in METHOD_DIRECTORIES:
        command, output = evaluation_command(args, method_directory, protocol_sha256)
        process, stream = start_logged(
            command,
            args.output_root / 'logs' / f'evaluate-{method_directory}.log',
            args.source,
        )
        running.append((method_directory, process, stream, output))
    failures: list[str] = []
    outputs: list[Path] = []
    for method_directory, process, stream, output in running:
        return_code = process.wait()
        stream.close()
        if return_code:
            failures.append(f'{method_directory} evaluation exit={return_code}')
        else:
            outputs.append(output)
    if failures:
        raise RuntimeError('; '.join(failures))
    result_paths = [
        output / f'{method_directory}_seed_{seed}_step_200000.json'
        for method_directory, output in zip(METHOD_DIRECTORIES, outputs)
        for seed in TRAINING_SEEDS
    ]
    missing = [str(path) for path in result_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f'evaluator exited without results: {missing}')
    return result_paths


def run_finalizer(
    args: argparse.Namespace,
    protocol_sha256: str,
    evaluations: list[Path],
) -> None:
    """Run the strict four-method merger after all six evaluations exist."""
    command = [
        str(args.python_bin),
        str(args.source / 'scripts/star/finalize_pointgoal_four_method_comparison.py'),
        '--prior-evaluations',
        *(str(path) for path in evaluations),
        '--protocol',
        str(args.source / PROTOCOL_RELATIVE),
        '--expected-protocol-sha256',
        protocol_sha256,
        '--output-dir',
        str(args.output_root / 'final'),
    ]
    process, stream = start_logged(
        command,
        args.output_root / 'logs' / 'finalize.log',
        args.source,
    )
    return_code = process.wait()
    stream.close()
    if return_code:
        raise RuntimeError(f'PointGoal four-method finalizer exited with {return_code}')
    manifest = args.output_root / 'final' / 'pointgoal_four_method_manifest.json'
    if read_json(manifest).get('status') != 'complete':
        raise RuntimeError('PointGoal finalizer did not create a complete manifest')


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source', type=Path, required=True)
    result.add_argument('--official-shapo-source', type=Path, required=True)
    result.add_argument('--training-root', type=Path, required=True)
    result.add_argument('--output-root', type=Path, required=True)
    result.add_argument('--python-bin', type=Path, required=True)
    result.add_argument('--poll-seconds', type=int, default=30)
    result.add_argument('--timeout-seconds', type=int, default=0)
    return result


def main() -> None:
    """Wait, evaluate two methods concurrently, and finalize the PointGoal table."""
    args = parser().parse_args()
    for field in ('source', 'official_shapo_source', 'training_root', 'output_root', 'python_bin'):
        setattr(args, field, getattr(args, field).resolve())
    if args.poll_seconds <= 0 or args.timeout_seconds < 0:
        raise ValueError('poll interval must be positive and timeout non-negative')
    source_commit = git_identity(args.source)
    queue_identity = read_json(args.training_root / 'queue_identity.json')
    protocol_sha256 = str(queue_identity['protocol_sha256'])
    protocol_path = args.source / PROTOCOL_RELATIVE
    protocol = read_json(protocol_path)
    if protocol.get('protocol_version') != EXPECTED_PROTOCOL_VERSION:
        raise ValueError('unexpected single-GPU protocol version')
    if sha256_file(protocol_path) != protocol_sha256:
        raise ValueError('training queue and postprocessor protocol identities differ')

    state_path = args.output_root / 'postprocessing_state.json'
    if state_path.is_file():
        existing_state = read_json(state_path)
        if existing_state.get('state') == 'complete':
            final_manifest = Path(existing_state['final_manifest'])
            if read_json(final_manifest).get('status') != 'complete':
                raise RuntimeError('completed postprocessing state has no valid final manifest')
            print(state_path.read_text(encoding='utf-8'), end='')
            return
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        state_path,
        {
            'state': 'waiting_for_training',
            'source_commit': source_commit,
            'protocol_sha256': protocol_sha256,
            'started_at_utc': utc_now(),
        },
    )
    wait_for_training(
        args.training_root,
        protocol_sha256,
        args.poll_seconds,
        args.timeout_seconds,
    )
    atomic_json(
        state_path,
        {
            'state': 'evaluating',
            'source_commit': source_commit,
            'protocol_sha256': protocol_sha256,
            'updated_at_utc': utc_now(),
        },
    )
    evaluations = run_evaluations(args, protocol_sha256)
    atomic_json(
        state_path,
        {
            'state': 'finalizing',
            'source_commit': source_commit,
            'protocol_sha256': protocol_sha256,
            'updated_at_utc': utc_now(),
        },
    )
    run_finalizer(args, protocol_sha256, evaluations)
    complete = {
        'state': 'complete',
        'source_commit': source_commit,
        'protocol_sha256': protocol_sha256,
        'finished_at_utc': utc_now(),
        'final_manifest': str(
            args.output_root / 'final' / 'pointgoal_four_method_manifest.json'
        ),
    }
    atomic_json(state_path, complete)
    print(json.dumps(complete, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
