#!/usr/bin/env python3
"""Run the unchanged P4.5 cost-geometry audit for a sealed P4.6 candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.star import analyze_p4_5_cost_geometry as p45


PROTOCOL_VERSION = 'p4.6-direction-weight-pareto-v1'
PROTOCOL_SHA256 = 'c473da2ed99b31cde673a1cd038ec0477053b23ac63a130376ebccc4271ad93e'
ARM_WEIGHTS = {'C025': 0.25, 'C050': 0.50}


def arm_from_flags(algo_cfgs: dict[str, Any]) -> str:
    """Validate unchanged method flags and map the locked candidate weight to its arm."""
    required = {
        'unified_cost_geometry_enabled': True,
        'stochastic_flow_enabled': True,
        'safety_anisotropy_enabled': True,
        'safety_inward_drift_enabled': False,
        'safety_outward_rectification_enabled': False,
        'candidate_probing_enabled': False,
        'dynamic_geometry_gate': False,
        'directional_loss_enabled': True,
    }
    mismatches = {
        key: algo_cfgs.get(key)
        for key, expected in required.items()
        if algo_cfgs.get(key) != expected
    }
    if mismatches:
        raise ValueError(f'checkpoint violates P4.6 method flags: {mismatches}')
    weight = float(algo_cfgs.get('directional_loss_weight', math.nan))
    matches = [arm for arm, expected in ARM_WEIGHTS.items() if math.isclose(weight, expected)]
    if len(matches) != 1:
        raise ValueError(f'checkpoint has unregistered P4.6 direction weight: {weight!r}')
    return matches[0]


def audit_checkpoint(
    checkpoint: Path,
    output: Path,
    work_dir: Path,
    device: str,
    state_count: int = p45.DEFAULT_STATE_COUNT,
    sample_count: int = p45.DEFAULT_ACTION_SAMPLES,
    audit_seed: int = p45.DEFAULT_AUDIT_SEED,
) -> dict[str, Any]:
    """Delegate numerical integration to the frozen P4.5 audit and reseal its identity."""
    p45.PROTOCOL_VERSION = PROTOCOL_VERSION
    p45.PROTOCOL_SHA256 = PROTOCOL_SHA256
    p45._arm_from_flags = arm_from_flags  # pylint: disable=protected-access
    result = p45.audit_checkpoint(
        checkpoint,
        output,
        work_dir,
        device,
        state_count,
        sample_count,
        audit_seed,
    )
    result['stage'] = 'P4.6-final-cost-geometry-audit'
    result['directional_loss_weight'] = ARM_WEIGHTS[result['arm']]
    result['flags']['directional_loss_weight'] = ARM_WEIGHTS[result['arm']]
    p45.atomic_json(output, result)
    return result


def parse_args() -> argparse.Namespace:
    """Parse one final-checkpoint audit invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--state-count', type=int, default=p45.DEFAULT_STATE_COUNT)
    parser.add_argument('--action-samples', type=int, default=p45.DEFAULT_ACTION_SAMPLES)
    parser.add_argument('--audit-seed', type=int, default=p45.DEFAULT_AUDIT_SEED)
    args = parser.parse_args()
    if args.state_count <= 0 or args.action_samples <= 1:
        parser.error('state count must be positive and action samples must exceed one')
    return args


def main() -> None:
    """Audit one P4.6 checkpoint and print its compact mode summary."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result = audit_checkpoint(
        args.checkpoint,
        args.output,
        args.work_dir,
        args.device,
        args.state_count,
        args.action_samples,
        args.audit_seed,
    )
    print(
        json.dumps(
            {
                'output': str(args.output),
                'identity': f"{result['arm']}:{result['training_seed']}",
                'modes': result['modes'],
            },
        ),
    )


if __name__ == '__main__':
    main()
