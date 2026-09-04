# Copyright 2026 OmniSafe Team. All Rights Reserved.
"""Tests for the locked P4.4 seed-2028 pilot decision."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.star.summarize_p4_4_pilot import summarize


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'reports/star/p4_4_outward_rectification_preregistration.json'
PROTOCOL_SHA256 = '005c42348a22f664448c688b3cecc16eba7cbbea55d7727537caeef5d3b69457'


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def _diagnostics(path: Path, *, unstable: bool) -> None:
    fields = (
        'TotalEnvSteps',
        'Critic/CostTD',
        'Critic/RewardTD',
        'Critic/RewardQAbsMax',
        'Critic/RewardTargetQAbsMax',
        'Numerics/CostGradientNorm',
        'Numerics/ViolationGradientNorm',
        'Numerics/RewardGradientNorm',
        'Numerics/ActorGradientNorm',
        'Reference/RawDangerousNormalVelocity',
        'Reference/DangerousNormalVelocity',
        'Reference/PreClippingVariance',
        'Reference/PostClippingVariance',
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, step in enumerate(range(5_000, 30_001, 5_000)):
            writer.writerow(
                {
                    'TotalEnvSteps': step,
                    'Critic/CostTD': 0.1,
                    'Critic/RewardTD': 0.021 if unstable and index == 5 else 0.004 + index * 0.0001,
                    'Critic/RewardQAbsMax': 1.0,
                    'Critic/RewardTargetQAbsMax': 1.0,
                    'Numerics/CostGradientNorm': 0.1,
                    'Numerics/ViolationGradientNorm': 0.1,
                    'Numerics/RewardGradientNorm': 0.1,
                    'Numerics/ActorGradientNorm': 0.1,
                    'Reference/RawDangerousNormalVelocity': 0.2,
                    'Reference/DangerousNormalVelocity': 0.1,
                    'Reference/PreClippingVariance': 0.08,
                    'Reference/PostClippingVariance': 0.07,
                },
            )


def _args(
    tmp_path: Path,
    *,
    r1_outward_speed: float = 0.09,
    unstable_r1: bool = False,
) -> argparse.Namespace:
    run_bindings = []
    receipt_bindings = []
    evaluation_bindings = []
    audit_bindings = []
    specs = {
        'A': (False, True, False),
        'B': (False, True, True),
        'C': (True, False, True),
        'D': (True, True, True),
    }
    for arm in ('R0', 'R1'):
        rectification = arm == 'R1'
        root = tmp_path / arm
        run = root / 'run'
        artifacts = run / 'artifacts'
        checkpoint = run / 'checkpoints/final_full.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'checkpoint')
        summary = {
            'status': 'completed',
            'seed': 2028,
            'env_steps': 30_000,
            'run_env_steps': 30_000,
            'updates': 28_000,
            'stochastic_flow_enabled': True,
            'safety_anisotropy_enabled': True,
            'safety_inward_drift_enabled': False,
            'safety_outward_rectification_enabled': rectification,
            'complete_heldout_evidence': True,
            'candidate_probing_enabled': False,
            'numerical_hard_stop_reason': None,
        }
        _write_json(artifacts / 'training_summary.json', summary)
        _write_json(
            artifacts / 'run_manifest.json',
            {
                'seed': 2028,
                'p4_4_outward_rectification': {
                    'protocol_version': 'p4.4-outward-rectification-v1',
                    'protocol_sha256': PROTOCOL_SHA256,
                    'outward_rectification_enabled': rectification,
                },
            },
        )
        _write_json(
            artifacts / 'heldout_probability_latest.json',
            {
                'PositiveLabels': 300,
                'NegativeLabels': 400,
                'evidence_status': 'sufficient',
            },
        )
        _diagnostics(artifacts / 'diagnostics.csv', unstable=unstable_r1 and arm == 'R1')
        receipt = root / 'receipt.json'
        _write_json(
            receipt,
            {
                'status': 'passed',
                'seed': 2028,
                'steps': 30_000,
                'run_steps': 30_000,
                'updates': 28_000,
                'source_dirty': False,
                'source_commit': 'a' * 40,
            },
        )
        task = {
            'R0': (100.0, 10.0, 12.0, 0.60, 0.20),
            'R1': (98.0, 9.0, 13.0, 0.55, 0.25),
        }[arm]
        evaluation = root / 'evaluation.json'
        _write_json(
            evaluation,
            {
                'training_seed': 2028,
                'milestone': 30_000,
                'seeds': list(range(12_027, 12_047)),
                'evaluation_mode': 'fixed-random-flow',
                'base_action_policy': 'uniform[-1,1], fixed generator',
                'wiener_increment_policy': 'standard normal fixed generator',
                'mean_return': task[0],
                'mean_cost': task[1],
                'cost_cvar_20': task[2],
                'goal_met_episode_rate': task[3],
                'stall_ratio': task[4],
            },
        )
        mode_d = {
            'R0': (0.40, 0.20, 0.050),
            'R1': (0.35, r1_outward_speed, 0.049),
        }[arm]
        audit = root / 'audit.json'
        _write_json(
            audit,
            {
                'stage': 'P4.4-checkpoint-outward-decomposition',
                'protocol_version': 'p4.4-outward-rectification-v1',
                'protocol_sha256': PROTOCOL_SHA256,
                'parent_arm': arm,
                'training_seed': 2028,
                'env_steps': 30_000,
                'state_count': 64,
                'action_samples_per_state': 256,
                'flags': {'outward_rectification_enabled': rectification},
                'heldout_evidence': {
                    'Heldout/AUROC': 0.8,
                    'Heldout/BrierSkill': 0.1,
                    'Heldout/EvidenceSufficient': 1.0,
                },
                'modes': {
                    mode: {
                        'specification': {
                            'learned_drift': spec[0],
                            'wiener_noise': spec[1],
                            'action_clipping': spec[2],
                        },
                        'metrics': {
                            'crossing_rate': mode_d[0] if mode == 'D' else 0.5,
                            'outward_speed': mode_d[1] if mode == 'D' else 0.0,
                            'tangent_variance': mode_d[2] if mode == 'D' else 0.05,
                        },
                    }
                    for mode, spec in specs.items()
                },
            },
        )
        run_bindings.append(f'{arm}={run}')
        receipt_bindings.append(f'{arm}={receipt}')
        evaluation_bindings.append(f'{arm}={evaluation}')
        audit_bindings.append(f'{arm}={audit}')
    return argparse.Namespace(
        protocol=PROTOCOL,
        runs=run_bindings,
        receipts=receipt_bindings,
        evaluations=evaluation_bindings,
        audits=audit_bindings,
        output=tmp_path / 'decision.json',
        report=tmp_path / 'decision.md',
    )


def test_pilot_pass_authorizes_only_the_preregistered_expansion(tmp_path: Path) -> None:
    result = summarize(_args(tmp_path))
    assert result['decision'] == 'continue_seed_2028_and_expand_to_three_seeds'
    assert result['expansion_authorized'] is True
    assert all(result['criteria'].values())


def test_pilot_stops_when_outward_speed_is_not_halved(tmp_path: Path) -> None:
    result = summarize(_args(tmp_path, r1_outward_speed=0.11))
    assert result['decision'] == 'stop_after_seed_2028_pilot'
    assert result['criteria']['r1_outward_speed_reduced_by_at_least_half'] is False


def test_pilot_stops_on_reward_td_absolute_threshold(tmp_path: Path) -> None:
    result = summarize(_args(tmp_path, unstable_r1=True))
    assert result['decision'] == 'stop_after_seed_2028_pilot'
    assert result['criteria']['r1_numerically_stable'] is False
