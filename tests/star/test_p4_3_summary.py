# Copyright 2026 OmniSafe Team. All Rights Reserved.
"""Tests for the locked P4.3 three-arm causal decision."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from scripts.star.summarize_p4_3_safety_anisotropy import summarize


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'reports/star/p4_3_safety_anisotropy_preregistration.json'
PROTOCOL_SHA = '271b65a6c1869de9d15cfc0037c5b0e61f7bf3864d022769dcc9ae256046e052'
FLAGS = {
    'S0': (True, False, False, False),
    'S1': (True, True, False, False),
    'S2': (True, True, True, False),
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def make_evidence(tmp_path: Path, *, s1_normal_variance: float = 0.02) -> argparse.Namespace:
    run_bindings = []
    receipt_bindings = []
    evaluation_bindings = []
    audit_bindings = []
    for arm in ('S0', 'S1', 'S2'):
        stochastic, anisotropy, drift, probing = FLAGS[arm]
        for seed in (2027, 2028, 2029):
            root = tmp_path / arm / str(seed)
            run_dir = root / 'run'
            artifacts = run_dir / 'artifacts'
            artifacts.mkdir(parents=True)
            checkpoint = run_dir / 'checkpoints/final_full.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'checkpoint')
            summary = {
                'status': 'completed',
                'seed': seed,
                'env_steps': 50_000,
                'updates': 48_000,
                'stochastic_flow_enabled': stochastic,
                'safety_anisotropy_enabled': anisotropy,
                'safety_inward_drift_enabled': drift,
                'candidate_probing_enabled': probing,
                'global_safety_constraint_enabled': True,
                'reward_target_actor_enabled': True,
                'actor_update_interval': 2,
                'reward_target_samples': 4,
                'reward_critic_updates_per_update': 2,
                'separate_reward_replay': True,
                'reward_replay_recent_fraction': 0.5,
                'reward_replay_recent_window': 10_000,
                'reward_target_polyak': 0.001,
                'bellman_probability_enabled': True,
                'probability_target_polyak': 0.005,
                'calibration_reliability': 0.8,
            }
            write_json(artifacts / 'training_summary.json', summary)
            write_json(
                artifacts / 'run_manifest.json',
                {
                    'seed': seed,
                    'p4_3_stochastic_safety_reference': {
                        'protocol_version': 'p4.3-safety-anisotropy-v1',
                        'protocol_sha256': PROTOCOL_SHA,
                        'stochastic_flow_enabled': stochastic,
                        'safety_anisotropy_enabled': anisotropy,
                        'safety_inward_drift_enabled': drift,
                        'candidate_probing_enabled': probing,
                    },
                },
            )
            diagnostics = artifacts / 'diagnostics.csv'
            fields = (
                'TotalEnvSteps',
                'Critic/CostTD',
                'Critic/RewardTD',
                'Loss/CostCritic',
                'Loss/RewardCritic',
                'Dual/GlobalSafety',
                'Time/FPS',
                'Reference/TraceRelativeError',
            )
            with diagnostics.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for step in range(5_000, 50_001, 5_000):
                    writer.writerow(
                        {
                            'TotalEnvSteps': step,
                            'Critic/CostTD': 1.0,
                            'Critic/RewardTD': 1.0,
                            'Loss/CostCritic': 1.0,
                            'Loss/RewardCritic': 1.0,
                            'Dual/GlobalSafety': 0.1,
                            'Time/FPS': 10.0,
                            'Reference/TraceRelativeError': ('' if step == 5_000 else 1e-7),
                        },
                    )
            receipt_path = root / 'receipt.json'
            write_json(
                receipt_path,
                {
                    'status': 'passed',
                    'seed': seed,
                    'steps': 50_000,
                    'updates': 48_000,
                    'source_dirty': False,
                    'source_commit': 'a' * 40,
                    'training_summary': summary,
                },
            )
            task = {
                'S0': (100.0, 20.0, 30.0, 0.60, 0.20),
                'S1': (105.0, 18.0, 27.0, 0.60, 0.20),
                'S2': (110.0, 15.0, 22.0, 0.62, 0.18),
            }[arm]
            evaluation_path = root / 'evaluation.json'
            write_json(
                evaluation_path,
                {
                    'training_seed': seed,
                    'milestone': 50_000,
                    'seeds': list(range(12_027, 12_047)),
                    'evaluation_mode': 'fixed-random-flow',
                    'wiener_increment_policy': 'standard normal fixed local generator',
                    'mean_return': task[0],
                    'mean_cost': task[1],
                    'cost_cvar_20': task[2],
                    'goal_met_episode_rate': task[3],
                    'stall_ratio': task[4],
                },
            )
            normal_variance = {
                'S0': 0.04,
                'S1': s1_normal_variance,
                'S2': 0.018,
            }[arm]
            tangent_variance = {'S0': 0.04, 'S1': 0.06, 'S2': 0.062}[arm]
            crossing = {'S0': 0.40, 'S1': 0.30, 'S2': 0.25}[arm]
            displacement = {'S0': 0.00, 'S1': -0.02, 'S2': -0.10}[arm]
            conditional = {
                'normal_variance': normal_variance,
                'tangent_variance': tangent_variance,
                'tangent_entropy': 0.5 * math.log(2 * math.pi * math.e * tangent_variance),
                'crossing_rate': crossing,
                'mean_normal_displacement': displacement,
            }
            audit_path = root / 'audit.json'
            write_json(
                audit_path,
                {
                    'stage': 'P4.3-safety-anisotropy-heldout-audit',
                    'protocol_version': 'p4.3-safety-anisotropy-v1',
                    'protocol_sha256': PROTOCOL_SHA,
                    'training_seed': seed,
                    'env_steps': 50_000,
                    'state_count': 64,
                    'action_samples_per_state': 256,
                    'flags': {
                        'stochastic_flow_enabled': stochastic,
                        'safety_anisotropy_enabled': anisotropy,
                        'safety_inward_drift_enabled': drift,
                        'candidate_probing_enabled': probing,
                    },
                    'valid_normal_state_count': 64,
                    'maximum_reference_trace_relative_error': 1e-7,
                    'selection': {'in_band_state_count': 64},
                    'heldout_evidence': {
                        'Heldout/AUROC': 0.8,
                        'Heldout/BrierSkill': 0.1,
                    },
                    'local_reference': {
                        'entropy_advantage_vs_matched_normal_isotropic_low': (
                            0.0 if arm == 'S0' else 0.3
                        ),
                    },
                    'full_policy': {'crossing_rate': crossing},
                    'conditional_wiener': conditional,
                },
            )
            identity = f'{arm}:{seed}'
            run_bindings.append(f'{identity}={run_dir}')
            receipt_bindings.append(f'{identity}={receipt_path}')
            evaluation_bindings.append(f'{identity}={evaluation_path}')
            audit_bindings.append(f'{identity}={audit_path}')
    return argparse.Namespace(
        protocol=PROTOCOL,
        runs=run_bindings,
        receipts=receipt_bindings,
        evaluations=evaluation_bindings,
        audits=audit_bindings,
        output=tmp_path / 'decision.json',
        report=tmp_path / 'decision.md',
    )


def test_summary_passes_the_complete_mechanism_chain(tmp_path: Path) -> None:
    result = summarize(make_evidence(tmp_path))
    assert result['decision'] == 's0_s1_s2_mechanism_chain_passed'
    assert all(result['criteria'].values())
    assert result['final_p5_verdict'] is None


def test_summary_rejects_weak_normal_contraction(tmp_path: Path) -> None:
    result = summarize(make_evidence(tmp_path, s1_normal_variance=0.039))
    assert result['criteria']['s1_normal_variance_reduced'] is False
    assert result['decision'] == 'reject_safety_anisotropy_mechanism'
