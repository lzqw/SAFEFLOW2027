#!/usr/bin/env python3
"""Run the single preregistered PointCircle geometry-enhancement screen."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.star.decide_pointcircle_geometry_enhancement import (
    ADDENDUM_RELATIVE,
    ADDENDUM_SHA256,
    CONFIG_RELATIVE,
    ORIGINAL_C0_EVALUATION_SHA256,
    ORIGINAL_DECISION_SHA256,
    validate_trigger,
)
from scripts.star.decide_pointcircle_screen import atomic_json, sha256_file


ENVIRONMENT = 'SafetyPointCircle1-v0'
TRAINING_SEED = 2027
STEPS = 300_000
REPLAY_SIZE = 1_000_000
CONFIG_SHA256 = 'd0252cb0bc22c783153bae45874322e99364d9a1458048591e3301c56111a6c3'


def utc_now() -> str:
    """Return one UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} does not contain a JSON object')
    return payload


def validate_source(source: Path, expected_commit: str) -> dict[str, str]:
    """Require the exact clean source and locked enhancement artifacts."""
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError('geometry-enhancement training source must be clean')
    commit = subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected_commit:
        raise ValueError(f'source commit mismatch: expected {expected_commit}, found {commit}')
    addendum_sha = sha256_file(source / ADDENDUM_RELATIVE)
    config_sha = sha256_file(source / CONFIG_RELATIVE)
    if addendum_sha != ADDENDUM_SHA256:
        raise ValueError('geometry-enhancement preregistration identity mismatch')
    if config_sha != CONFIG_SHA256:
        raise ValueError('geometry-enhancement config identity mismatch')
    return {
        'source_commit': commit,
        'enhancement_preregistration_sha256': addendum_sha,
        'config_sha256': config_sha,
    }


def receipt_complete(path: Path, source_commit: str) -> bool:
    """Return whether a receipt proves the exact enhanced 300k run."""
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
        and int(payload.get('seed', -1)) == TRAINING_SEED
        and int(payload.get('steps', -1)) == STEPS
        and payload.get('config_sha256') == CONFIG_SHA256
        and payload.get('source_commit') == source_commit
        and payload.get('source_dirty') is False
    )


def training_command(args: argparse.Namespace) -> tuple[list[str], Path, Path]:
    """Build the one allowed learner command."""
    container = args.run_root / 'P47-geometry-enhanced-2027'
    receipt = container / 'receipt.json'
    log = container / 'train.log'
    command = [
        str(args.python_bin),
        str(args.source / 'scripts/star/train_star.py'),
        '--config',
        str(args.source / CONFIG_RELATIVE),
        '--seed',
        str(TRAINING_SEED),
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
        '250000',
        '--log-dir',
        str(container / 'training'),
        '--result',
        str(receipt),
        '--no-tensorboard',
    ]
    return command, receipt, log


def run_training(args: argparse.Namespace, source_commit: str) -> Path:
    """Run or authenticate the single enhanced learner."""
    command, receipt, log = training_command(args)
    if receipt_complete(receipt, source_commit):
        return receipt
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        completed = subprocess.run(
            command,
            cwd=args.source,
            env={
                **os.environ,
                'CUDA_VISIBLE_DEVICES': '0',
                'MKL_NUM_THREADS': '1',
                'OMP_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1',
                'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONPATH': str(args.source),
            },
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode or not receipt_complete(receipt, source_commit):
        raise RuntimeError(f'enhanced learner failed with exit={completed.returncode}; log={log}')
    return receipt


def run_decision(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Apply the unchanged screen against the frozen original C0 evidence."""
    output = args.run_root / 'screen_decision.json'
    log = args.run_root / 'screen-decision.log'
    command = [
        str(args.python_bin),
        str(args.source / 'scripts/star/decide_pointcircle_geometry_enhancement.py'),
        '--source',
        str(args.source),
        '--original-root',
        str(args.original_root),
        '--enhancement-root',
        str(args.run_root),
        '--output',
        str(output),
    ]
    with log.open('a', encoding='utf-8') as stream:
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
    payload = read_json(output)
    if payload.get('status') != 'complete':
        raise RuntimeError('geometry-enhancement screen decision is incomplete')
    return output, payload


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source', type=Path, required=True)
    result.add_argument('--original-root', type=Path, required=True)
    result.add_argument('--run-root', type=Path, required=True)
    result.add_argument('--python-bin', type=Path, required=True)
    result.add_argument('--expected-source-commit', required=True)
    return result


def main() -> None:
    """Validate the trigger, train one learner, and freeze its decision."""
    args = parser().parse_args()
    for field in ('source', 'original_root', 'run_root', 'python_bin'):
        setattr(args, field, getattr(args, field).resolve())
    identities = validate_source(args.source, args.expected_source_commit)
    c0_path, _ = validate_trigger(args.original_root)
    args.run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_root / 'queue_identity.json',
        {
            **identities,
            'environment': ENVIRONMENT,
            'attempt': 1,
            'attempt_limit': 1,
            'training_seed': TRAINING_SEED,
            'steps': STEPS,
            'maximum_concurrent_learners': 1,
            'original_screen_decision_sha256': ORIGINAL_DECISION_SHA256,
            'frozen_c0_evaluation': str(c0_path),
            'frozen_c0_evaluation_sha256': ORIGINAL_C0_EVALUATION_SHA256,
        },
    )
    atomic_json(
        args.run_root / 'queue_state.json',
        {
            'state': 'training',
            'started_at_utc': utc_now(),
            'attempt': 1,
            'attempt_limit': 1,
            'scientific_training_launched': True,
        },
    )
    try:
        receipt = run_training(args, identities['source_commit'])
        decision_path, decision = run_decision(args)
        passed = bool(decision['decision']['screen_pass'])
        atomic_json(
            args.run_root / 'queue_state.json',
            {
                'state': 'screen_passed' if passed else 'screen_not_passed',
                'finished_at_utc': utc_now(),
                'attempt': 1,
                'attempt_limit': 1,
                'scientific_training_launched': True,
                'receipt': str(receipt),
                'screen_decision': str(decision_path),
                'no_further_pointcircle_method_changes_allowed': True,
            },
        )
    except Exception as error:
        atomic_json(
            args.run_root / 'queue_state.json',
            {
                'state': 'failed',
                'failed_at_utc': utc_now(),
                'attempt': 1,
                'attempt_limit': 1,
                'scientific_training_launched': True,
                'engineering_failure': True,
                'error': str(error),
            },
        )
        raise


if __name__ == '__main__':
    main()
