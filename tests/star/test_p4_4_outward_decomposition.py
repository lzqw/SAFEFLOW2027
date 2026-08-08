# Copyright 2026 OmniSafe Team. All Rights Reserved.
"""Tests for the locked P4.4 pretraining decomposition and critic gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from scripts.star.analyze_p4_4_outward_decomposition import _mode_metrics
from scripts.star.summarize_p4_4_outward_decomposition import summarize


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'reports/star/p4_4_outward_rectification_preregistration.json'
PROTOCOL_SHA256 = '005c42348a22f664448c688b3cecc16eba7cbbea55d7727537caeef5d3b69457'
FLAGS = {
    'S0': (True, False, False, False),
    'S1': (True, True, False, False),
    'S2': (True, True, True, False),
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def _metric(value: float) -> dict[str, float]:
    return {
        'normal_variance': value,
        'tangent_variance': value + 0.01,
        'total_empirical_variance': 2.0 * value + 0.01,
        'pre_clipping_normal_variance': value + 0.02,
        'pre_clipping_tangent_variance': value + 0.03,
        'pre_clipping_total_empirical_variance': 2.0 * value + 0.05,
        'crossing_rate': 0.25,
        'mean_normal_displacement': -0.01,
        'clipping_rate': 0.1,
        'final_endpoint_clipping_rate': 0.1,
        'outward_velocity_ratio': 0.4,
        'outward_speed': 0.05,
        'mean_learned_normal_velocity': -0.01,
        'path_valid_normal_ratio': 1.0,
        'mean_end_margin': 0.0,
    }


def _args(
    tmp_path: Path,
    *,
    negative_count: int = 31,
    short_identity: tuple[str, int] | None = None,
) -> argparse.Namespace:
    bindings = []
    specs = {
        'A': (False, True, False),
        'B': (False, True, True),
        'C': (True, False, True),
        'D': (True, True, True),
    }
    for arm in ('S0', 'S1', 'S2'):
        stochastic, anisotropy, inward, probing = FLAGS[arm]
        for seed in (2027, 2028, 2029):
            valid_count = 31 if short_identity == (arm, seed) else 64
            local_negative = min(negative_count, valid_count)
            path = tmp_path / arm / f'{seed}.json'
            payload = {
                'stage': 'P4.4-pretraining-outward-decomposition',
                'protocol_version': 'p4.4-outward-rectification-v1',
                'protocol_sha256': PROTOCOL_SHA256,
                'parent_arm': arm,
                'training_seed': seed,
                'env_steps': 50_000,
                'state_count': 64,
                'action_samples_per_state': 256,
                'audit_seed': 4_000_007,
                'checkpoint': f'/checkpoint/{arm}-{seed}.pt',
                'checkpoint_sha256': 'a' * 64,
                'valid_fixed_normal_state_count': 64,
                'selection': {'in_band_state_count': 64},
                'heldout_evidence': {},
                'flags': {
                    'stochastic_flow_enabled': stochastic,
                    'safety_anisotropy_enabled': anisotropy,
                    'checkpoint_inward_drift_enabled': inward,
                    'decomposition_inward_drift_enabled': False,
                    'candidate_probing_enabled': probing,
                },
                'modes': {
                    mode: {
                        'specification': {
                            'learned_drift': spec[0],
                            'wiener_noise': spec[1],
                            'action_clipping': spec[2],
                        },
                        'metrics': _metric(0.02 + 0.001 * index),
                    }
                    for index, (mode, spec) in enumerate(specs.items())
                },
                'critic_alignment': {
                    'valid_state_count': valid_count,
                    'invalid_state_count': 64 - valid_count,
                    'negative_count': local_negative,
                    'negative_fraction': local_negative / valid_count,
                    'mean_cosine': 0.1,
                    'median_cosine': 0.1,
                },
            }
            _write_json(path, payload)
            bindings.append(f'{arm}:{seed}={path}')
    return argparse.Namespace(
        protocol=PROTOCOL,
        audits=bindings,
        output=tmp_path / 'decision.json',
        report=tmp_path / 'decision.md',
    )


def test_p4_4_preregistration_has_the_locked_pretraining_seal() -> None:
    payload = PROTOCOL.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == PROTOCOL_SHA256
    protocol = json.loads(payload)
    assert protocol['status'] == 'locked_before_pilot_training'
    assert protocol['base_commit'].startswith('59b3493')
    assert protocol['decomposition_audit']['state_count'] == 64
    assert protocol['decomposition_audit']['action_samples_per_state'] == 256
    assert protocol['heldout']['minimum_positive_labels'] == 256
    assert protocol['heldout']['minimum_negative_labels'] == 256


def test_mode_metrics_record_pre_and_post_clipping_and_outward_velocity() -> None:
    actions = torch.tensor([[1.0, -0.5], [1.0, 0.5], [0.5, 0.0]])
    proposal = torch.tensor([[1.5, -0.5], [1.5, 0.5], [0.5, 0.0]])
    normal = torch.tensor([1.0, 0.0])
    normal_velocity = torch.tensor([[1.0, -1.0], [2.0, -2.0], [0.0, 0.0]])
    result = _mode_metrics(
        actions,
        proposal,
        normal,
        torch.zeros(1, 2),
        torch.tensor([0.1, -0.1, 0.0]),
        0.0,
        normal_velocity,
        normal_velocity,
        torch.ones_like(normal_velocity, dtype=torch.bool),
        clipping_events=2,
    )
    assert result['pre_clipping_normal_variance'] > result['normal_variance']
    assert math.isclose(result['clipping_rate'], 1.0 / 3.0)
    assert math.isclose(result['outward_velocity_ratio'], 1.0 / 3.0, abs_tol=1e-7)
    assert math.isclose(result['outward_speed'], 0.5)


def test_critic_alignment_gate_authorizes_less_than_half_negative(tmp_path: Path) -> None:
    result = summarize(_args(tmp_path, negative_count=31))
    assert result['decision'] == 'p4_4_training_authorized'
    assert result['critic_alignment_gate']['training_authorized'] is True
    assert result['critic_alignment_gate']['pooled_negative_fraction'] < 0.5


def test_critic_alignment_gate_blocks_half_negative(tmp_path: Path) -> None:
    result = summarize(_args(tmp_path, negative_count=32))
    assert result['decision'] == 'blocked_for_critic_repair'
    assert result['critic_alignment_gate']['pooled_orientation_pass'] is False


def test_critic_alignment_gate_blocks_a_short_checkpoint(tmp_path: Path) -> None:
    result = summarize(_args(tmp_path, short_identity=('S2', 2029)))
    assert result['decision'] == 'blocked_for_critic_repair'
    assert result['critic_alignment_gate']['per_checkpoint_validity_pass'] is False
