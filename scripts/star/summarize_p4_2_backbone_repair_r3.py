#!/usr/bin/env python3
"""Apply the preregistered P4.2 B0-R3 three-seed reward-backbone gate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.star.summarize_p4_2_backbone import ENDPOINTS, SEEDS, atomic_json, read_json
from scripts.star.summarize_p4_2_backbone_repair_r2 import repair_summary as r2_repair_summary
from scripts.star.summarize_p4_2_backbone_repair_r3_pilot import (
    EXPECTED_TIMESCALE,
    R3_SOURCE_COMMIT,
    validate_r3_run,
)


PILOT_PROMOTION_DECISION = 'launch_remaining_r3_repair_seeds'


def validate_r3_pilot_parent(path: Path) -> dict[str, Any]:
    """Require the immutable passing R3 pilot before reading the other seeds."""
    payload = read_json(path)
    if payload.get('stage') != 'P4.2-B0-R3-seed-2028-feasibility':
        raise ValueError('three-seed R3 parent is not the registered pilot')
    if payload.get('repair_source_commit') != R3_SOURCE_COMMIT:
        raise ValueError('three-seed R3 parent has an unexpected repair source')
    if payload.get('target_timescale') != EXPECTED_TIMESCALE:
        raise ValueError('three-seed R3 parent has an unexpected target timescale')
    if payload.get('pilot', {}).get('pilot_decision') != PILOT_PROMOTION_DECISION:
        raise ValueError('remaining R3 seeds were not authorized by the pilot')
    if payload.get('final_p4_verdict') is not None:
        raise ValueError('R3 pilot parent must keep final_p4_verdict null')
    return payload


def repair_summary(
    run_dirs: list[Path],
    receipts: list[Path],
    evaluation_csv: Path,
) -> dict[str, Any]:
    """Validate all R3 timescale fields, then apply the unchanged R2 functional gate."""
    if len(run_dirs) != 3 or len(receipts) != 3:
        raise ValueError('B0-R3 requires exactly three run directories and receipts')
    for run_dir in run_dirs:
        summary = read_json(run_dir / 'artifacts/training_summary.json')
        manifest = read_json(run_dir / 'artifacts/run_manifest.json')
        validate_r3_run(summary, manifest, int(summary['seed']))
    result = r2_repair_summary(run_dirs, receipts, evaluation_csv)
    result['name'] = 'B0-R3'
    result['target_timescale'] = EXPECTED_TIMESCALE
    return result


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run-dirs', nargs=3, type=Path, required=True)
    result.add_argument('--receipts', nargs=3, type=Path, required=True)
    result.add_argument('--evaluation-csv', type=Path, required=True)
    result.add_argument('--r3-pilot-decision', type=Path, required=True)
    result.add_argument('--repair-source-commit', required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main() -> None:
    """Write the strict machine-readable R3 repair decision."""
    args = parser().parse_args()
    validate_r3_pilot_parent(args.r3_pilot_decision)
    if args.repair_source_commit != R3_SOURCE_COMMIT:
        raise ValueError('three-seed source is not the frozen B0-R3 repair commit')
    repair = repair_summary(args.run_dirs, args.receipts, args.evaluation_csv)
    decision = (
        'proceed_to_bellman_probability_critic'
        if repair['functional_gate_passed']
        else 'continue_backbone_repair_and_pause_safety'
    )
    payload = {
        'schema_version': 1,
        'stage': 'P4.2-B0-R3-reward-backbone-repair',
        'repair_source_commit': R3_SOURCE_COMMIT,
        'target_timescale': EXPECTED_TIMESCALE,
        'seeds': list(SEEDS),
        'target_steps': ENDPOINTS[-1],
        'external_evaluation_protocol': {
            'mode': 'fixed-random-flow',
            'episodes_per_checkpoint_per_seed': 20,
            'milestones': list(ENDPOINTS),
        },
        'reward_td_divergence_rule': {
            'tail': 'last four 5k diagnostic rows',
            'strictly_rising': True,
            'final_to_historical_minimum_ratio_at_least': 2.0,
            'arm_pass_requires_no_seed_divergence': True,
        },
        'functional_gate_rule': {
            'pooled_success_rate': 'strictly rises from 0 to 50k',
            'pooled_return': 'strictly improves from 0 to 50k',
            'pooled_stall': 'does not rise from 0 to 50k',
            'pooled_step_displacement': 'does not decline from 0 to 50k',
            'reward_td': 'no seed triggers the unchanged monotonic-divergence rule',
            'arm_pass': 'all five criteria must pass',
        },
        'repair_arm': repair,
        'backbone_repair_decision': decision,
        'final_p4_verdict': None,
        'formal_four_environment_experiment_launched': False,
        'continuation_beyond_50k_launched': False,
    }
    atomic_json(args.output, payload)
    print({'backbone_repair_decision': decision, 'output': str(args.output)})


if __name__ == '__main__':
    main()
