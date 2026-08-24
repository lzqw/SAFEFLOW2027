#!/usr/bin/env python3
"""Evaluate and finalize PointCircle mechanism/parameter controls."""

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
PROTOCOL_RELATIVE = Path('reports/star/single_gpu_paper_core_preregistration.json')
VARIANTS = ('Iso-High', 'Safety-Aniso', 'Iso-Low', 'Random-Aniso', 'Safety-Aniso-0.25')


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


def validate_source(source: Path) -> dict[str, str]:
    """Require a clean committed source under the frozen protocol."""
    if git_output(source, 'status', '--porcelain'):
        raise ValueError('PointCircle control postprocessing source must be clean')
    protocol_sha = sha256_file(source / PROTOCOL_RELATIVE)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise ValueError('control postprocessor has the wrong protocol identity')
    return {
        'source_commit': git_output(source, 'rev-parse', 'HEAD'),
        'protocol_sha256': protocol_sha,
    }


def wait_state(path: Path, poll_seconds: int) -> str:
    """Wait for a complete or screen-not-passed control queue state."""
    while True:
        if path.is_file():
            payload = read_json(path)
            if payload.get('protocol_sha256') != EXPECTED_PROTOCOL_SHA256:
                raise ValueError('control queue state has the wrong protocol identity')
            state = str(payload.get('state'))
            if state in ('complete', 'screen_not_passed'):
                return state
        time.sleep(poll_seconds)


def checkpoint(args: argparse.Namespace, variant: str) -> Path:
    """Resolve exactly one lightweight 300k checkpoint for a control variant."""
    if variant == 'Iso-High':
        container = args.star_root / 'screen/C0-2027'
    elif variant == 'Safety-Aniso':
        container = args.star_root / 'screen/P47-2027'
    elif variant in ('Iso-Low', 'Random-Aniso'):
        container = args.control_root / 'mechanism' / f'{variant}-2027'
    else:
        container = args.control_root / 'parameter/Safety-Aniso-0.25-2027'
    matches = sorted(
        container.glob(
            f'training/STAR-{{{ENVIRONMENT}}}/seed-*/checkpoints/step_300000.pt'
        )
    )
    if len(matches) != 1:
        raise ValueError(f'expected one {variant} 300k checkpoint, found {len(matches)}')
    return matches[0]


def start(command: list[str], log: Path, source: Path) -> tuple[subprocess.Popen[str], Any]:
    """Start one evaluation/finalization command."""
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


def evaluate_wave(args: argparse.Namespace, variants: tuple[str, ...]) -> None:
    """Evaluate one or two variants concurrently."""
    if not 1 <= len(variants) <= 2:
        raise ValueError('evaluation wave must contain one or two variants')
    running: list[tuple[str, subprocess.Popen[str], Any, Path]] = []
    for variant in variants:
        slug = variant.lower().replace('-', '_')
        output = args.output_root / 'evaluations' / f'{slug}.json'
        command = [
            str(args.python_bin),
            str(args.source / 'scripts/star/evaluate_pointcircle_control.py'),
            '--variant',
            variant,
            '--checkpoint',
            str(checkpoint(args, variant)),
            '--output',
            str(output),
            '--work-dir',
            str(args.output_root / 'work' / slug),
            '--device',
            'cuda:0',
        ]
        log = args.output_root / 'logs' / f'evaluate-{slug}.log'
        process, stream = start(command, log, args.source)
        running.append((variant, process, stream, log))
    failures: list[str] = []
    for variant, process, stream, log in running:
        return_code = process.wait()
        stream.close()
        if return_code:
            failures.append(f'{variant} exit={return_code} log={log}')
    if failures:
        raise RuntimeError('; '.join(failures))
    for variant in variants:
        output = args.output_root / 'evaluations' / f'{variant.lower().replace("-", "_")}.json'
        if read_json(output).get('status') != 'complete':
            raise ValueError(f'{variant} evaluation is incomplete')


def finalize(args: argparse.Namespace, protocol_sha: str) -> Path:
    """Run the evidence finalizer and return its manifest."""
    output = args.output_root / 'final'
    command = [
        str(args.python_bin),
        str(args.source / 'scripts/star/finalize_pointcircle_controls.py'),
        '--star-screen-root',
        str(args.star_root / 'screen'),
        '--control-root',
        str(args.control_root),
        '--evaluation-root',
        str(args.output_root / 'evaluations'),
        '--protocol',
        str(args.source / PROTOCOL_RELATIVE),
        '--expected-protocol-sha256',
        protocol_sha,
        '--output-dir',
        str(output),
    ]
    log = args.output_root / 'logs/finalize.log'
    process, stream = start(command, log, args.source)
    return_code = process.wait()
    stream.close()
    if return_code:
        raise RuntimeError(f'control finalizer exit={return_code}; see {log}')
    manifest = output / 'pointcircle_control_manifest.json'
    if read_json(manifest).get('status') != 'complete':
        raise ValueError('control final manifest is incomplete')
    return manifest


def parser() -> argparse.ArgumentParser:
    """Build postprocessing arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source', type=Path, required=True)
    result.add_argument('--star-root', type=Path, required=True)
    result.add_argument('--control-root', type=Path, required=True)
    result.add_argument('--output-root', type=Path, required=True)
    result.add_argument('--python-bin', type=Path, required=True)
    result.add_argument('--poll-seconds', type=int, default=30)
    return result


def main() -> None:
    """Wait for control training, evaluate all five variants, and finalize."""
    args = parser().parse_args()
    for field in ('source', 'star_root', 'control_root', 'output_root', 'python_bin'):
        setattr(args, field, getattr(args, field).resolve())
    if args.poll_seconds <= 0:
        raise ValueError('poll interval must be positive')
    identities = validate_source(args.source)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output_root / 'postprocessing_state.json',
        {'state': 'waiting_for_training', 'started_at_utc': utc_now(), **identities},
    )
    state = wait_state(args.control_root / 'queue_state.json', args.poll_seconds)
    if state == 'screen_not_passed':
        atomic_json(
            args.output_root / 'postprocessing_state.json',
            {'state': state, 'finished_at_utc': utc_now(), **identities},
        )
        return
    atomic_json(
        args.output_root / 'postprocessing_state.json',
        {'state': 'evaluating', 'updated_at_utc': utc_now(), **identities},
    )
    evaluate_wave(args, ('Iso-High', 'Safety-Aniso'))
    evaluate_wave(args, ('Iso-Low', 'Random-Aniso'))
    evaluate_wave(args, ('Safety-Aniso-0.25',))
    manifest = finalize(args, identities['protocol_sha256'])
    atomic_json(
        args.output_root / 'postprocessing_state.json',
        {
            'state': 'complete',
            'finished_at_utc': utc_now(),
            'final_manifest': str(manifest),
            'final_manifest_sha256': sha256_file(manifest),
            **identities,
        },
    )


if __name__ == '__main__':
    main()
