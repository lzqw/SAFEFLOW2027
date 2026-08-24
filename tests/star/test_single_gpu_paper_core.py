"""Tests for the resource-aligned single-GPU paper-core protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.star.run_single_gpu_pointgoal_queue import (
    DUAL_RATIO_THRESHOLD,
    EXPECTED_PROTOCOL_VERSION,
    PROBE_STEPS,
    command_for,
    concurrency_decision,
    receipt_complete,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'reports/star/single_gpu_paper_core_preregistration.json'
MATRIX = ROOT / 'configs/star/single_gpu_paper_core.yaml'


def test_protocol_locks_requested_single_gpu_scope() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    matrix = yaml.safe_load(MATRIX.read_text(encoding='utf-8'))

    assert protocol['protocol_version'] == EXPECTED_PROTOCOL_VERSION
    assert protocol['status'] == 'locked_before_any_new_single_gpu_scientific_training'
    assert protocol['global_invariants']['maximum_concurrent_learners'] == 2
    assert protocol['global_invariants']['learner_vector_envs'] == 1
    assert protocol['global_invariants']['training_seeds'] == [2027, 2028, 2029]
    assert protocol['global_invariants']['final_evaluation']['episodes_per_policy'] == 100
    assert protocol['stages']['A_pointgoal_prior_methods']['new_training_steps'] == 1_200_000
    assert protocol['stages']['B_pointcircle_screen']['pass_rule'] == {
        'and_at_least_one': [
            'mean_cost_or_CVaR_relative_improvement_at_least_0.10',
            'high_risk_vn_vt_at_most_0.9_and_outward_velocity_lower',
        ],
        'return_relative_degradation_at_most': 0.1,
    }
    assert matrix['throughput_gate']['keep_dual_minimum_aggregate_ratio'] == 1.2
    assert matrix['pointcircle']['milestones'] == [150000, 300000, 500000]
    assert matrix['secondary_extension']['walker2d'] == 'appendix_seed_2027_only'


def test_receipt_and_throughput_decision_are_engineering_only(tmp_path: Path) -> None:
    protocol_sha = 'a' * 64
    single_dir = tmp_path / 'throughput_single' / 'saclag' / 'seed-2027'
    single_dir.mkdir(parents=True)
    single_receipt = single_dir / 'receipt.json'
    single_receipt.write_text(
        json.dumps(
            {
                'algorithm': 'SACLag',
                'elapsed_seconds': 1000.0,
                'protocol_sha256': protocol_sha,
                'seed': 2027,
                'status': 'complete',
                'steps': PROBE_STEPS,
            },
        ),
        encoding='utf-8',
    )
    pair_dir = tmp_path / 'throughput_dual'
    pair_dir.mkdir()
    (pair_dir / 'pair-seed-2027.json').write_text(
        json.dumps(
            {
                'controller_elapsed_seconds': 1500.0,
                'newly_launched_methods': ['SACLag', 'SHAPO-PIDLag'],
            },
        ),
        encoding='utf-8',
    )
    args = argparse.Namespace(run_root=tmp_path)
    decision = concurrency_decision(args, {'protocol_sha256': protocol_sha})

    assert receipt_complete(single_receipt, 'SACLag', 2027, PROBE_STEPS, protocol_sha)
    assert decision['engineering_only'] is True
    assert decision['scientific_metrics_inspected'] is False
    assert decision['threshold'] == DUAL_RATIO_THRESHOLD
    assert decision['decision'] == 'two_learners'


def test_frozen_commands_keep_one_vector_env_and_official_shapo_identity(
    tmp_path: Path,
) -> None:
    source = ROOT
    args = argparse.Namespace(
        source=source,
        official_shapo_source=Path('/opt/SHAPO'),
        python_bin=Path('/opt/python'),
        run_root=tmp_path,
    )
    identities = {'protocol_sha256': 'b' * 64, 'source_commit': 'c' * 40}
    saclag, _, _ = command_for(
        args,
        identities,
        'formal_pointgoal',
        'SACLag',
        2027,
        200000,
    )
    shapo, _, _ = command_for(
        args,
        identities,
        'formal_pointgoal',
        'SHAPO-PIDLag',
        2027,
        200000,
    )

    assert saclag[saclag.index('--steps-per-epoch') + 1] == '2000'
    assert shapo[shapo.index('--num-envs') + 1] == '1'
    assert shapo[shapo.index('--use-shapo-actor') + 1] == 'True'
    assert shapo[shapo.index('--perturbation-target-kl') + 1] == '0.00001'
