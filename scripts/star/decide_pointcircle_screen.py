#!/usr/bin/env python3
"""Apply the preregistered PointCircle seed-2027 screen at 300k steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENVIRONMENT = 'SafetyPointCircle1-v0'
BASE_SEEDS = (22_027, 22_028, 22_029, 22_030, 22_031)
MILESTONES = (150_000, 300_000)
MAX_RETURN_DEGRADATION = 0.10
MINIMUM_SAFETY_IMPROVEMENT = 0.10
MAXIMUM_HIGH_RISK_VN_VT = 0.90
GATE_EPSILON = 1e-12


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


def finite(payload: dict[str, Any], key: str, label: str) -> float:
    """Read one required finite metric."""
    value = float(payload[key])
    if not math.isfinite(value):
        raise ValueError(f'{label}.{key} is not finite')
    return value


def read_evaluation(path: Path, milestone: int, arm: str) -> dict[str, float]:
    """Validate and compact one 20-episode PointCircle development evaluation."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    label = f'{arm}@{milestone}'
    if payload.get('environment') != ENVIRONMENT:
        raise ValueError(f'{label} belongs to another environment')
    if int(payload.get('milestone', -1)) != milestone:
        raise ValueError(f'{label} milestone mismatch')
    if int(payload.get('episodes', -1)) != 20:
        raise ValueError(f'{label} is not a 20-episode development evaluation')
    if [int(seed) for seed in payload.get('base_seeds', [])] != list(BASE_SEEDS):
        raise ValueError(f'{label} uses the wrong base seeds')
    if int(payload.get('episodes_per_base_seed', -1)) != 4:
        raise ValueError(f'{label} does not use four episodes per base seed')
    high_risk = payload.get('direction_geometry_high_risk_top_20_percent')
    if not isinstance(high_risk, dict):
        raise ValueError(f'{label} has no high-risk direction geometry')
    return {
        'mean_return': finite(payload, 'mean_return', label),
        'mean_cost': finite(payload, 'mean_cost', label),
        'cost_cvar_20': finite(payload, 'cost_cvar_20', label),
        'high_risk_vn_vt': finite(
            high_risk,
            'normal_to_tangent_variance_ratio',
            f'{label}.high_risk',
        ),
        'high_risk_dangerous_outward_velocity': finite(
            high_risk,
            'dangerous_outward_velocity',
            f'{label}.high_risk',
        ),
    }


def relative_reduction(baseline: float, candidate: float) -> float:
    """Return a scale-safe relative reduction, including the zero-cost case."""
    scale = max(abs(baseline), 1e-12)
    return (baseline - candidate) / scale


def screen_decision(c0: dict[str, float], ours: dict[str, float]) -> dict[str, Any]:
    """Apply exactly the return AND (safety OR geometry) screen."""
    return_relative_degradation = max(
        0.0,
        (c0['mean_return'] - ours['mean_return']) / max(abs(c0['mean_return']), 1e-12),
    )
    mean_cost_improvement = relative_reduction(c0['mean_cost'], ours['mean_cost'])
    cvar_improvement = relative_reduction(c0['cost_cvar_20'], ours['cost_cvar_20'])
    return_ok = return_relative_degradation <= MAX_RETURN_DEGRADATION + GATE_EPSILON
    safety_ok = (
        mean_cost_improvement >= MINIMUM_SAFETY_IMPROVEMENT - GATE_EPSILON
        or cvar_improvement >= MINIMUM_SAFETY_IMPROVEMENT - GATE_EPSILON
    )
    geometry_ok = (
        ours['high_risk_vn_vt'] <= MAXIMUM_HIGH_RISK_VN_VT + GATE_EPSILON
        and ours['high_risk_dangerous_outward_velocity']
        < c0['high_risk_dangerous_outward_velocity']
    )
    return {
        'screen_pass': return_ok and (safety_ok or geometry_ok),
        'return_ok': return_ok,
        'safety_ok': safety_ok,
        'geometry_ok': geometry_ok,
        'return_relative_degradation': return_relative_degradation,
        'mean_cost_relative_improvement': mean_cost_improvement,
        'cvar_20_relative_improvement': cvar_improvement,
        'thresholds': {
            'maximum_return_relative_degradation': MAX_RETURN_DEGRADATION,
            'minimum_mean_cost_or_cvar_improvement': MINIMUM_SAFETY_IMPROVEMENT,
            'maximum_high_risk_vn_vt': MAXIMUM_HIGH_RISK_VN_VT,
            'outward_velocity_rule': 'P47_strictly_lower_than_C0',
        },
        'logic': 'return_ok AND (safety_ok OR geometry_ok)',
    }


def validate_receipt(path: Path, config_sha256: str, source_commit: str) -> None:
    """Require one exact completed 300k seed-2027 training receipt."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    valid = (
        payload.get('status') == 'passed'
        and payload.get('environment') == ENVIRONMENT
        and int(payload.get('seed', -1)) == 2027
        and int(payload.get('steps', -1)) == 300_000
        and payload.get('config_sha256') == config_sha256
        and payload.get('source_commit') == source_commit
        and payload.get('source_dirty') is False
    )
    if not valid:
        raise ValueError(f'invalid PointCircle screen receipt {path}')


def git_commit(source: Path) -> str:
    """Require clean decision source and return its exact commit."""
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError('PointCircle screen decision requires a clean source')
    return subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source', type=Path, required=True)
    result.add_argument('--screen-root', type=Path, required=True)
    result.add_argument('--protocol-sha256', required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main() -> None:
    """Load both curves, apply the frozen gate, and write one immutable decision."""
    args = parser().parse_args()
    args.source = args.source.resolve()
    args.screen_root = args.screen_root.resolve()
    args.output = args.output.resolve()
    source_commit = git_commit(args.source)
    configs = {
        'C0': args.source / 'configs/star/single_gpu_pointcircle_c0.yaml',
        'P47': args.source / 'configs/star/single_gpu_pointcircle_p47.yaml',
    }
    config_hashes = {arm: sha256_file(path) for arm, path in configs.items()}
    curves: dict[str, dict[str, dict[str, float]]] = {}
    evidence: list[dict[str, Any]] = []
    for arm in ('C0', 'P47'):
        run = args.screen_root / f'{arm}-2027'
        receipt = run / 'receipt.json'
        validate_receipt(receipt, config_hashes[arm], source_commit)
        curves[arm] = {}
        evidence.append({'path': str(receipt), 'sha256': sha256_file(receipt)})
        for milestone in MILESTONES:
            matches = sorted(
                run.glob(f'training/STAR-{{{ENVIRONMENT}}}/seed-*/artifacts/evaluation_step_{milestone}.json'),
            )
            if len(matches) != 1:
                raise ValueError(f'expected one {arm}@{milestone} evaluation, found {len(matches)}')
            evaluation = matches[0]
            curves[arm][str(milestone)] = read_evaluation(evaluation, milestone, arm)
            evidence.append({'path': str(evaluation), 'sha256': sha256_file(evaluation)})
    decision = screen_decision(curves['C0']['300000'], curves['P47']['300000'])
    payload = {
        'status': 'complete',
        'environment': ENVIRONMENT,
        'training_seed': 2027,
        'screen_milestone': 300_000,
        'source_commit': source_commit,
        'protocol_sha256': args.protocol_sha256,
        'config_sha256': config_hashes,
        'decided_at_utc': datetime.now(timezone.utc).isoformat(),
        'development_curves': curves,
        'decision': decision,
        'evidence': evidence,
        'scientific_early_stopping': False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
