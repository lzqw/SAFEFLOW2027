#!/usr/bin/env python3
"""Decide the single permitted PointCircle geometry-enhancement screen."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.star.decide_pointcircle_screen import (
    ENVIRONMENT,
    atomic_json,
    git_commit,
    read_evaluation,
    screen_decision,
    sha256_file,
)


TRAINING_SEED = 2027
STEPS = 300_000
PARENT_PROTOCOL_SHA256 = '0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8'
ADDENDUM_SHA256 = 'fcde54b52e02c00f4c8997f676f5f963fd307dbae8d053559e55e1568c9ffc02'
ORIGINAL_DECISION_SHA256 = '62873e89e8584fbb6663a183f71dcca84d33ef7b63296308bb47e88af1221d86'
ORIGINAL_C0_EVALUATION_SHA256 = (
    'ac6e54f6086c6718209d415d552b0b97c5ee4e0958a858cc1aefd993091fbbcc'
)
CONFIG_RELATIVE = Path('configs/star/single_gpu_pointcircle_p47_geometry_enhanced.yaml')
ADDENDUM_RELATIVE = Path(
    'reports/star/single_gpu_pointcircle_geometry_enhancement_preregistration.json',
)


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} does not contain a JSON object')
    return payload


def validate_trigger(original_root: Path) -> tuple[Path, dict[str, float]]:
    """Authenticate the failed original screen and its frozen C0 endpoint."""
    decision_path = original_root / 'screen_decision.json'
    if sha256_file(decision_path) != ORIGINAL_DECISION_SHA256:
        raise ValueError('original PointCircle decision identity mismatch')
    payload = read_json(decision_path)
    original_decision = payload.get('decision', {})
    original_p47 = payload.get('development_curves', {}).get('P47', {}).get('300000', {})
    valid = (
        payload.get('status') == 'complete'
        and payload.get('environment') == ENVIRONMENT
        and payload.get('protocol_sha256') == PARENT_PROTOCOL_SHA256
        and original_decision.get('screen_pass') is False
        and float(original_p47.get('high_risk_vn_vt', -1.0)) > 0.9
    )
    if not valid:
        raise ValueError('the preregistered one-time enhancement trigger is not satisfied')
    matches = sorted(
        (original_root / 'screen/C0-2027').glob(
            'training/STAR-{SafetyPointCircle1-v0}/seed-*/artifacts/'
            'evaluation_step_300000.json',
        ),
    )
    if len(matches) != 1 or sha256_file(matches[0]) != ORIGINAL_C0_EVALUATION_SHA256:
        raise ValueError('frozen original C0 evaluation identity mismatch')
    return matches[0], read_evaluation(matches[0], STEPS, 'C0')


def validate_enhanced_receipt(
    path: Path,
    config_sha256: str,
    source_commit: str,
) -> dict[str, Any]:
    """Require one exact clean-source enhanced 300k training receipt."""
    payload = read_json(path)
    valid = (
        payload.get('status') == 'passed'
        and payload.get('algorithm') == 'STAR'
        and payload.get('environment') == ENVIRONMENT
        and int(payload.get('seed', -1)) == TRAINING_SEED
        and int(payload.get('steps', -1)) == STEPS
        and payload.get('config_sha256') == config_sha256
        and payload.get('source_commit') == source_commit
        and payload.get('source_dirty') is False
    )
    if not valid:
        raise ValueError('invalid enhanced PointCircle receipt')
    return payload


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source', type=Path, required=True)
    result.add_argument('--original-root', type=Path, required=True)
    result.add_argument('--enhancement-root', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main() -> None:
    """Apply the unchanged gate to enhanced P47 versus the frozen original C0."""
    args = parser().parse_args()
    args.source = args.source.resolve()
    args.original_root = args.original_root.resolve()
    args.enhancement_root = args.enhancement_root.resolve()
    args.output = args.output.resolve()
    source_commit = git_commit(args.source)
    if sha256_file(args.source / ADDENDUM_RELATIVE) != ADDENDUM_SHA256:
        raise ValueError('geometry-enhancement preregistration identity mismatch')
    config_sha256 = sha256_file(args.source / CONFIG_RELATIVE)
    c0_path, c0 = validate_trigger(args.original_root)
    run = args.enhancement_root / 'P47-geometry-enhanced-2027'
    receipt_path = run / 'receipt.json'
    receipt = validate_enhanced_receipt(receipt_path, config_sha256, source_commit)
    matches = sorted(
        run.glob(
            'training/STAR-{SafetyPointCircle1-v0}/seed-*/artifacts/'
            'evaluation_step_300000.json',
        ),
    )
    if len(matches) != 1:
        raise ValueError(f'expected one enhanced 300k evaluation, found {len(matches)}')
    enhanced_path = matches[0]
    enhanced = read_evaluation(enhanced_path, STEPS, 'P47-Geometry-Enhanced')
    decision = screen_decision(c0, enhanced)
    payload = {
        'status': 'complete',
        'environment': ENVIRONMENT,
        'attempt': 1,
        'attempt_limit': 1,
        'training_seed': TRAINING_SEED,
        'screen_milestone': STEPS,
        'source_commit': source_commit,
        'parent_protocol_sha256': PARENT_PROTOCOL_SHA256,
        'enhancement_preregistration_sha256': ADDENDUM_SHA256,
        'config_sha256': config_sha256,
        'decided_at_utc': datetime.now(timezone.utc).isoformat(),
        'trigger': {
            'original_screen_decision_sha256': ORIGINAL_DECISION_SHA256,
            'original_p47_high_risk_vn_vt_above_0.9': True,
        },
        'comparison': {
            'C0': c0,
            'P47-Geometry-Enhanced': enhanced,
        },
        'decision': decision,
        'scientific_early_stopping': False,
        'no_further_pointcircle_method_changes_allowed': True,
        'evidence': [
            {'path': str(c0_path), 'sha256': sha256_file(c0_path)},
            {'path': str(receipt_path), 'sha256': sha256_file(receipt_path)},
            {'path': str(enhanced_path), 'sha256': sha256_file(enhanced_path)},
        ],
        'training_summary': receipt.get('training_summary'),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
