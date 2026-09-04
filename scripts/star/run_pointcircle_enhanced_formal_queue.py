#!/usr/bin/env python3
"""Conditionally run formal C0/enhanced-P47 PointCircle training after its gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.star.decide_pointcircle_geometry_enhancement import (
    ADDENDUM_RELATIVE,
    ADDENDUM_SHA256,
    ORIGINAL_DECISION_SHA256,
    PARENT_PROTOCOL_SHA256,
    validate_trigger,
)
from scripts.star.decide_pointcircle_screen import atomic_json, sha256_file


ENVIRONMENT = 'SafetyPointCircle1-v0'
ARMS = ('C0', 'P47')
SEEDS = (2027, 2028, 2029)
SCREEN_STEPS = 300_000
FORMAL_STEPS = 500_000
REPLAY_SIZE = 1_000_000
EXPECTED_MATRIX_SHA256 = '0c4fe4b21cd086bde6e703c486022716056c844b5c3b0d79a05a062b784aac88'
PROTOCOL_RELATIVE = Path('reports/star/single_gpu_paper_core_preregistration.json')
MATRIX_RELATIVE = Path('configs/star/single_gpu_paper_core.yaml')
CONFIG_RELATIVE = {
    'C0': Path('configs/star/single_gpu_pointcircle_c0.yaml'),
    'P47': Path('configs/star/single_gpu_pointcircle_p47_geometry_enhanced.yaml'),
}
CONFIG_SHA256 = {
    'C0': '5e8cda373b7c5f68ed34aa25d08179aa06597ee1899e694601d9e34575227f43',
    'P47': 'd0252cb0bc22c783153bae45874322e99364d9a1458048591e3301c56111a6c3',
}
ORIGINAL_C0_SOURCE_COMMIT = 'ac334bfbcbedf6641aa3b454ac59497c38f8ce0b'


def utc_now() -> str:
    """Return one UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


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


def validate_source(source: Path, expected_commit: str) -> dict[str, Any]:
    """Require exact clean source plus frozen protocol/config identities."""
    if git_output(source, 'status', '--porcelain'):
        raise ValueError('enhanced PointCircle formal source must be clean')
    source_commit = git_output(source, 'rev-parse', 'HEAD')
    if source_commit != expected_commit:
        raise ValueError(f'source commit mismatch: expected {expected_commit}, found {source_commit}')
    protocol_sha = sha256_file(source / PROTOCOL_RELATIVE)
    matrix_sha = sha256_file(source / MATRIX_RELATIVE)
    addendum_sha = sha256_file(source / ADDENDUM_RELATIVE)
    config_hashes = {
        arm: sha256_file(source / relative) for arm, relative in CONFIG_RELATIVE.items()
    }
    if protocol_sha != PARENT_PROTOCOL_SHA256 or matrix_sha != EXPECTED_MATRIX_SHA256:
        raise ValueError('enhanced formal queue does not match the frozen single-GPU protocol')
    if addendum_sha != ADDENDUM_SHA256 or config_hashes != CONFIG_SHA256:
        raise ValueError('enhanced formal queue does not match the locked candidate')
    return {
        'source_commit': source_commit,
        'protocol_sha256': protocol_sha,
        'matrix_sha256': matrix_sha,
        'enhancement_preregistration_sha256': addendum_sha,
        'config_sha256': config_hashes,
    }


def validate_terminal_gate(
    state: dict[str, Any],
    decision: dict[str, Any],
) -> bool:
    """Authenticate one terminal enhancement state and return its frozen outcome."""
    state_name = state.get('state')
    if state_name not in ('screen_passed', 'screen_not_passed'):
        raise ValueError('enhancement state is not terminal')
    passed = state_name == 'screen_passed'
    valid = (
        int(state.get('attempt', -1)) == 1
        and int(state.get('attempt_limit', -1)) == 1
        and state.get('scientific_training_launched') is True
        and state.get('no_further_pointcircle_method_changes_allowed') is True
        and decision.get('status') == 'complete'
        and decision.get('environment') == ENVIRONMENT
        and int(decision.get('attempt', -1)) == 1
        and int(decision.get('attempt_limit', -1)) == 1
        and decision.get('parent_protocol_sha256') == PARENT_PROTOCOL_SHA256
        and decision.get('enhancement_preregistration_sha256') == ADDENDUM_SHA256
        and bool(decision.get('decision', {}).get('screen_pass')) is passed
    )
    if not valid:
        raise ValueError('invalid terminal geometry-enhancement gate')
    return passed


def wait_for_gate(path: Path, poll_seconds: int) -> tuple[bool, dict[str, Any]]:
    """Wait for the single enhancement screen to reach a valid terminal state."""
    while True:
        if path.is_file():
            state = read_json(path)
            if state.get('state') == 'failed':
                raise RuntimeError(f'enhancement queue failed: {state.get("error")}')
            if state.get('state') in ('screen_passed', 'screen_not_passed'):
                decision = read_json(Path(state['screen_decision']))
                return validate_terminal_gate(state, decision), decision
        time.sleep(poll_seconds)


def validate_screen_receipt(
    path: Path,
    arm: str,
    expected_source_commit: str,
) -> None:
    """Require an exact completed seed-2027 300k screen receipt."""
    payload = read_json(path)
    valid = (
        payload.get('status') == 'passed'
        and payload.get('algorithm') == 'STAR'
        and payload.get('environment') == ENVIRONMENT
        and int(payload.get('seed', -1)) == 2027
        and int(payload.get('steps', -1)) == SCREEN_STEPS
        and payload.get('config_sha256') == CONFIG_SHA256[arm]
        and payload.get('source_commit') == expected_source_commit
        and payload.get('source_dirty') is False
    )
    if not valid:
        raise ValueError(f'invalid {arm} screen receipt {path}')


def unique_final_checkpoint(container: Path) -> Path:
    """Resolve exactly one full-replay final checkpoint."""
    matches = sorted(
        container.glob(f'training/STAR-{{{ENVIRONMENT}}}/seed-*/checkpoints/final_full.pt'),
    )
    if len(matches) != 1:
        raise ValueError(f'expected one final_full.pt below {container}, found {len(matches)}')
    return matches[0]


def screen_resumes(
    original_root: Path,
    enhancement_root: Path,
    enhancement_decision: dict[str, Any],
) -> dict[str, Path]:
    """Authenticate and resolve the two heterogeneous 300k screen checkpoints."""
    validate_trigger(original_root)
    c0_container = original_root / 'screen/C0-2027'
    p47_container = enhancement_root / 'P47-geometry-enhanced-2027'
    validate_screen_receipt(
        c0_container / 'receipt.json',
        'C0',
        ORIGINAL_C0_SOURCE_COMMIT,
    )
    validate_screen_receipt(
        p47_container / 'receipt.json',
        'P47',
        str(enhancement_decision['source_commit']),
    )
    return {
        'C0': unique_final_checkpoint(c0_container),
        'P47': unique_final_checkpoint(p47_container),
    }


def run_directory(root: Path, arm: str, seed: int) -> Path:
    """Return one stable formal run container."""
    return root / 'formal' / f'{arm}-{seed}'


def receipt_complete(
    path: Path,
    arm: str,
    seed: int,
    identities: dict[str, Any],
) -> bool:
    """Return whether one formal receipt proves exact completion."""
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
        and int(payload.get('steps', -1)) == FORMAL_STEPS
        and payload.get('source_commit') == identities['source_commit']
        and payload.get('source_dirty') is False
        and payload.get('config_sha256') == identities['config_sha256'][arm]
    )


def training_command(
    args: argparse.Namespace,
    arm: str,
    seed: int,
    resume: Path | None,
) -> tuple[list[str], Path, Path]:
    """Build one exact formal learner command."""
    container = run_directory(args.run_root, arm, seed)
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
        str(FORMAL_STEPS),
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
    return command, receipt, log


def start_process(command: list[str], log: Path, source: Path) -> tuple[subprocess.Popen[str], Any]:
    """Start one formal learner with fixed single-environment resources."""
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
    seed: int,
    resumes: dict[str, Path | None],
) -> dict[str, Any]:
    """Run one paired formal wave, stopping its counterpart on engineering failure."""
    pending: list[tuple[str, list[str], Path, Path]] = []
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        command, receipt, log = training_command(args, arm, seed, resumes[arm])
        if receipt_complete(receipt, arm, seed, identities):
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
            if return_code or not receipt_complete(receipt, arm, seed, identities):
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
        'seed': seed,
        'steps_per_method': FORMAL_STEPS,
        'started_at_utc': started_at,
        'finished_at_utc': utc_now(),
        'elapsed_seconds': time.monotonic() - started,
        'rows': sorted(rows, key=lambda row: row['arm']),
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
    result.add_argument('--original-root', type=Path, required=True)
    result.add_argument('--enhancement-root', type=Path, required=True)
    result.add_argument('--run-root', type=Path, required=True)
    result.add_argument('--python-bin', type=Path, required=True)
    result.add_argument('--expected-source-commit', required=True)
    result.add_argument('--poll-seconds', type=int, default=30)
    return result


def main() -> None:
    """Wait for the enhanced gate and conditionally run the 2x3 formal matrix."""
    args = parser().parse_args()
    for field in ('source', 'original_root', 'enhancement_root', 'run_root', 'python_bin'):
        setattr(args, field, getattr(args, field).resolve())
    if args.poll_seconds <= 0:
        raise ValueError('poll interval must be positive')
    identities = validate_source(args.source, args.expected_source_commit)
    args.run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_root / 'queue_identity.json',
        {
            **identities,
            'environment': ENVIRONMENT,
            'arms': list(ARMS),
            'seeds': list(SEEDS),
            'steps_per_run': FORMAL_STEPS,
            'maximum_concurrent_learners': 2,
            'vector_env_nums': 1,
            'utd_ratio': 1.0,
            'original_screen_decision_sha256': ORIGINAL_DECISION_SHA256,
        },
    )
    atomic_json(
        args.run_root / 'queue_state.json',
        {
            'state': 'waiting_for_enhancement_screen',
            'started_at_utc': utc_now(),
            'protocol_sha256': identities['protocol_sha256'],
            'scientific_training_launched': False,
        },
    )
    try:
        passed, decision = wait_for_gate(
            args.enhancement_root / 'queue_state.json',
            args.poll_seconds,
        )
        if not passed:
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
        resumes = screen_resumes(args.original_root, args.enhancement_root, decision)
        atomic_json(
            args.run_root / 'queue_state.json',
            {
                'state': 'training',
                'started_at_utc': utc_now(),
                'protocol_sha256': identities['protocol_sha256'],
                'scientific_training_launched': True,
                'screen_resumes': {
                    arm: {'path': str(path), 'sha256': sha256_file(path)}
                    for arm, path in resumes.items()
                },
            },
        )
        run_pair(args, identities, 2027, resumes)
        for seed in (2028, 2029):
            run_pair(args, identities, seed, dict.fromkeys(ARMS))
        atomic_json(
            args.run_root / 'queue_state.json',
            {
                'state': 'complete',
                'finished_at_utc': utc_now(),
                'protocol_sha256': identities['protocol_sha256'],
                'scientific_training_launched': True,
                'environment': ENVIRONMENT,
                'arms': list(ARMS),
                'seeds': list(SEEDS),
                'steps_per_run': FORMAL_STEPS,
            },
        )
    except Exception as error:
        atomic_json(
            args.run_root / 'queue_state.json',
            {
                'state': 'failed',
                'failed_at_utc': utc_now(),
                'protocol_sha256': identities['protocol_sha256'],
                'engineering_failure': True,
                'error': str(error),
            },
        )
        raise


if __name__ == '__main__':
    main()
