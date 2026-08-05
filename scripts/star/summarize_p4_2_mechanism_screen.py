#!/usr/bin/env python3
"""Apply the preregistered P4.2 B1--B4 50k causal-screen decision."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
from typing import Any

from scripts.star.summarize_p4_2_backbone import (
    atomic_json,
    number,
    optional_number,
    read_csv,
    read_json,
)


ARMS = ('B1', 'B2', 'B3', 'B4')
SCREEN_SEED = 2028
ENDPOINTS = (0, 50_000)
EXPECTED_DIAGNOSTIC_STEPS = tuple(range(5_000, 50_001, 5_000))
CORE_TRAINING_COLUMNS = (
    'Loss/CostCritic',
    'Loss/RewardCritic',
    'Loss/Actor',
    'Critic/CostTD',
    'Critic/RewardTD',
    'Dual/GlobalSafety',
    'TotalEnvSteps',
    'TotalUpdates',
    'TotalRewardCriticUpdates',
)
EXTERNAL_METRICS = (
    'mean_return',
    'mean_cost',
    'cost_cvar_20',
    'goal_met_episode_rate',
    'stall_ratio',
    'mean_step_displacement',
    'mean_start_end_displacement',
    'mean_action_norm',
)
HELDOUT_METRICS = (
    'BrierScore',
    'BrierSkill',
    'AUROC',
    'BellmanError',
    'GradientNorm',
    'EnsembleDisagreement',
    'BoundarySamples',
    'PositiveLabels',
    'NegativeLabels',
    'Episodes',
    'Transitions',
)
PROBING_METRICS = (
    'Probing/CandidateRiskP10',
    'Probing/CandidateRiskP50',
    'Probing/CandidateRiskP90',
    'Probing/SelectedRisk',
    'Probing/SelectedDisagreement',
    'Probing/ActualCost',
    'Probing/ActualViolation',
    'Probing/EffectiveBoundarySamples',
    'Probing/TangentRatio',
    'Probing/DangerousNormalVelocity',
)


def keyed_paths(values: list[str], label: str) -> dict[str, Path]:
    """Parse four explicit ``ARM=PATH`` command-line bindings."""
    result: dict[str, Path] = {}
    for value in values:
        arm, separator, raw_path = value.partition('=')
        if separator != '=' or arm not in ARMS or not raw_path:
            raise ValueError(f'{label} must use one ARM=PATH binding for each of {ARMS}')
        if arm in result:
            raise ValueError(f'{label} contains duplicate arm {arm}')
        result[arm] = Path(raw_path)
    if set(result) != set(ARMS):
        raise ValueError(f'{label} must bind exactly {ARMS}')
    return result


def reward_td_trend(rows: list[dict[str, str]], arm: str) -> dict[str, Any]:
    """Apply the screen's strict 1.5-times-minimum and rising-tail rule."""
    ordered = sorted(rows, key=lambda row: int(number(row['TotalEnvSteps'], 'TotalEnvSteps')))
    steps = [int(number(row['TotalEnvSteps'], f'{arm}.TotalEnvSteps')) for row in ordered]
    if tuple(steps) != EXPECTED_DIAGNOSTIC_STEPS:
        raise ValueError(f'{arm} diagnostics must cover each 5k step through 50k')
    values = [number(row['Critic/RewardTD'], f'{arm}.Critic/RewardTD') for row in ordered]
    historical_minimum = min(values)
    if historical_minimum <= 0.0:
        raise ValueError(f'{arm} reward TD minimum must be positive')
    last_four = values[-4:]
    strictly_rising = all(after > before for before, after in zip(last_four, last_four[1:]))
    final_to_minimum = values[-1] / historical_minimum
    return {
        'historical_minimum': historical_minimum,
        'historical_minimum_step': steps[values.index(historical_minimum)],
        'final': values[-1],
        'final_to_historical_minimum_ratio': final_to_minimum,
        'last_four_steps': steps[-4:],
        'last_four_values': last_four,
        'strictly_rising_last_three_intervals': strictly_rising,
        'monotonic_divergence': strictly_rising and final_to_minimum > 1.5,
    }


def finite_training_core(rows: list[dict[str, str]], arm: str) -> dict[str, Any]:
    """Validate required optimizer signals while allowing undefined held-out diagnostics."""
    for row_index, row in enumerate(rows):
        for column in CORE_TRAINING_COLUMNS:
            number(row[column], f'{arm}.diagnostics[{row_index}].{column}')
    return {
        'columns': list(CORE_TRAINING_COLUMNS),
        'rows': len(rows),
        'all_finite': True,
        'undefined_heldout_metrics_excluded': True,
    }


def heldout_history(path: Path, arm: str) -> list[dict[str, Any]]:
    """Validate the ten held-out-only geometry evaluations and make them strict JSON."""
    result: list[dict[str, Any]] = []
    rows = read_csv(path)
    steps = [int(number(row['env_steps'], f'{arm}.heldout.env_steps')) for row in rows]
    if tuple(steps) != EXPECTED_DIAGNOSTIC_STEPS:
        raise ValueError(f'{arm} held-out evidence must cover each 5k step through 50k')
    for row, step in zip(rows, steps):
        if row['split'] != 'heldout_episode':
            raise ValueError(f'{arm} geometry evidence did not use the held-out split')
        if int(number(row['recent_episode_limit'], f'{arm}.recent_episode_limit')) != 20:
            raise ValueError(f'{arm} has the wrong held-out episode limit')
        if int(number(row['transition_limit'], f'{arm}.transition_limit')) != 4096:
            raise ValueError(f'{arm} has the wrong held-out transition limit')
        result.append(
            {
                'env_steps': step,
                'gate_passed': row['gate_passed'].lower() == 'true',
                'curriculum_stage': row['curriculum_stage'],
                'geometry_enabled': row['geometry_enabled'].lower() == 'true',
                **{metric: optional_number(row.get(metric)) for metric in HELDOUT_METRICS},
            },
        )
    return result


def external_evidence(
    path: Path,
    arm: str,
    evaluation_seeds: list[int],
) -> dict[str, dict[str, Any]]:
    """Validate isolated step-0/50k evaluations and their exact seed identities."""
    grouped: dict[int, dict[str, Any]] = {}
    for row in read_csv(path):
        seed = int(number(row['training_seed'], f'{arm}.training_seed'))
        milestone = int(number(row['milestone'], f'{arm}.milestone'))
        if seed != SCREEN_SEED:
            raise ValueError(f'{arm} evaluation must use training seed {SCREEN_SEED}')
        if row['evaluation_mode'] != 'fixed-random-flow' or row['evaluation_role'] != 'primary':
            raise ValueError(f'{arm} requires primary fixed-random-flow evaluation')
        if row.get('probability_split') != 'external_isolated_environment':
            raise ValueError(f'{arm} evaluation did not preserve the isolated probability split')
        if int(number(row['episodes'], f'{arm}.episodes')) != len(evaluation_seeds):
            raise ValueError(f'{arm} evaluation has the wrong episode count')
        if milestone in grouped:
            raise ValueError(f'{arm} has a duplicate evaluation at step {milestone}')
        grouped[milestone] = {
            metric: number(row[metric], f'{arm}.{milestone}.{metric}')
            for metric in EXTERNAL_METRICS
        }
        grouped[milestone].update(
            {
                'episodes': len(evaluation_seeds),
                'goal_met_total': int(number(row['goal_met_total'], f'{arm}.goal_met_total')),
                'external_brier_skill': optional_number(row.get('brier_skill')),
                'external_auroc': optional_number(row.get('auroc')),
            },
        )
    if set(grouped) != set(ENDPOINTS):
        raise ValueError(f'{arm} evaluation must contain exactly steps {ENDPOINTS}')

    for milestone in ENDPOINTS:
        raw = read_json(path.parent / f'evaluation_seed_{SCREEN_SEED}_step_{milestone}.json')
        if raw.get('probability_split') != 'external_isolated_environment':
            raise ValueError(f'{arm} raw evaluation at {milestone} has the wrong split')
        if raw.get('evaluation_mode') != 'fixed-random-flow':
            raise ValueError(f'{arm} raw evaluation at {milestone} has the wrong mode')
        if [int(seed) for seed in raw.get('seeds', [])] != evaluation_seeds:
            raise ValueError(f'{arm} raw evaluation at {milestone} has the wrong seeds')
    return {str(step): grouped[step] for step in ENDPOINTS}


def _checkpoint_bytes(run_dir: Path, arm: str) -> dict[str, int]:
    paths = {
        label: run_dir / 'checkpoints' / filename
        for label, filename in {
            'milestone_0': 'milestone_0.pt',
            'milestone_50000': 'milestone_50000.pt',
            'latest_full': 'latest_full.pt',
            'final_full': 'final_full.pt',
        }.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'{arm} is missing checkpoints: {missing}')
    return {label: path.stat().st_size for label, path in paths.items()}


def arm_summary(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    arm: str,
    run_dir: Path,
    receipt_path: Path,
    evaluation_csv: Path,
    preregistration: dict[str, Any],
    training_source_commit: str,
) -> dict[str, Any]:
    """Validate one causal-screen arm and collect its ranking evidence."""
    registered = preregistration['arms'][arm]
    receipt = read_json(receipt_path)
    summary = read_json(run_dir / 'artifacts/training_summary.json')
    manifest = read_json(run_dir / 'artifacts/run_manifest.json')
    diagnostics = read_csv(run_dir / 'artifacts/diagnostics.csv')
    expected_switches = {
        'global_safety_constraint_enabled': bool(registered['global_safety']),
        'local_safety_geometry_enabled': bool(registered['dynamic_geometry']),
        'reward_flow_enabled': True,
    }
    if manifest['p4_2_backbone_ablation'] != expected_switches:
        raise ValueError(f'{arm} backbone switches do not match preregistration')
    if manifest['p4_2_probability_critic']['bellman_probability_enabled'] is not bool(
        registered['bellman_probability'],
    ):
        raise ValueError(f'{arm} Bellman-probability switch does not match preregistration')
    if manifest['p4_2_dynamic_geometry']['enabled'] is not bool(
        registered['dynamic_geometry'],
    ):
        raise ValueError(f'{arm} dynamic-geometry switch does not match preregistration')
    if manifest['p4_2_candidate_probing']['enabled'] is not bool(
        registered['candidate_probing'],
    ):
        raise ValueError(f'{arm} candidate-probing switch does not match preregistration')
    if tuple(int(step) for step in manifest['eval_milestones']) != ENDPOINTS:
        raise ValueError(f'{arm} has incorrect milestone checkpoints')
    if registered['dynamic_geometry']:
        dynamic = manifest['p4_2_dynamic_geometry']
        if (
            int(dynamic['enable_after_consecutive_passes']) != 3
            or int(dynamic['disable_after_consecutive_failures']) != 2
            or dynamic['gate_evidence_split'] != 'heldout_episode'
        ):
            raise ValueError(f'{arm} has the wrong reversible geometry-gate contract')
    if registered['candidate_probing']:
        probing = manifest['p4_2_candidate_probing']
        if (
            int(probing['candidate_count']) != 16
            or probing['risk_selection'] != 'closest_trusted_safe_boundary_then_reward_q'
            or probing['tangent_requires_geometry_enabled'] is not True
        ):
            raise ValueError(f'{arm} has the wrong K=16 candidate-probing contract')

    if receipt.get('status') != 'passed' or summary.get('status') != 'completed':
        raise ValueError(f'{arm} did not complete successfully')
    if int(receipt['seed']) != SCREEN_SEED or int(summary['seed']) != SCREEN_SEED:
        raise ValueError(f'{arm} used the wrong screen seed')
    if int(receipt['steps']) != ENDPOINTS[-1] or int(summary['env_steps']) != ENDPOINTS[-1]:
        raise ValueError(f'{arm} did not stop at exactly 50k')
    if int(receipt['updates']) != 48_000 or int(summary['updates']) != 48_000:
        raise ValueError(f'{arm} has incorrect update accounting')
    if (
        receipt.get('source_commit') != training_source_commit
        or receipt.get('source_dirty') is not False
    ):
        raise ValueError(f'{arm} receipt has the wrong or dirty training source')
    if receipt.get('config_sha256') != registered['config_sha256']:
        raise ValueError(f'{arm} config fingerprint does not match preregistration')
    if Path(str(receipt.get('config_file'))).as_posix() != registered['config']:
        raise ValueError(f'{arm} receipt names the wrong config file')

    probability = manifest['p4_2_probability_critic']
    if registered['bellman_probability']:
        episode_split = probability['episode_split']
        expected_split = preregistration['heldout']
        if int(episode_split['heldout_period']) != int(expected_split['episode_period']):
            raise ValueError(f'{arm} has the wrong held-out episode period')
        if any(
            episode_split[key]
            for key in (
                'optimizer_access_to_heldout',
                'global_dual_access_to_heldout',
                'observation_normalizer_access_to_heldout',
            )
        ):
            raise ValueError(f'{arm} leaked held-out episodes into training')
        if summary['replay_split_counts']['heldout_transitions'] <= 0:
            raise ValueError(f'{arm} did not populate the held-out split')

    finite_core = finite_training_core(diagnostics, arm)
    trend = reward_td_trend(diagnostics, arm)
    evaluation = external_evidence(
        evaluation_csv,
        arm,
        [int(seed) for seed in preregistration['external_evaluation']['seeds']],
    )
    heldout = (
        summary.get('heldout_probability_latest') if registered['bellman_probability'] else None
    )
    if registered['bellman_probability'] and not isinstance(heldout, dict):
        raise ValueError(f'{arm} is missing held-out probability evidence')
    history = (
        heldout_history(run_dir / 'artifacts/heldout_probability.csv', arm)
        if registered['bellman_probability']
        else []
    )
    probing_final: dict[str, float] | None = None
    if registered['candidate_probing']:
        for row_index, row in enumerate(diagnostics):
            for metric in PROBING_METRICS:
                number(row[metric], f'{arm}.diagnostics[{row_index}].{metric}')
        probing_final = {
            metric: number(diagnostics[-1][metric], f'{arm}.final.{metric}')
            for metric in PROBING_METRICS
        }
    return {
        'arm': arm,
        'mechanisms': {
            'global_safety': bool(registered['global_safety']),
            'bellman_probability': bool(registered['bellman_probability']),
            'dynamic_geometry': bool(registered['dynamic_geometry']),
            'candidate_probing': bool(registered['candidate_probing']),
        },
        'source_commit': receipt['source_commit'],
        'config_sha256': receipt['config_sha256'],
        'resolved_overrides_sha256': receipt['resolved_overrides_sha256'],
        'training_seconds': number(summary['training_seconds'], f'{arm}.training_seconds'),
        'estimated_cost_cny': number(receipt['estimated_cost_cny'], f'{arm}.estimated_cost_cny'),
        'training_core_finiteness': finite_core,
        'reward_td_trend': trend,
        'external_evaluation': evaluation,
        'heldout_probability_final': heldout,
        'heldout_probability_history': history,
        'geometry_activation_step': summary.get('geometry_activation_step'),
        'geometry_enabled_at_end': bool(summary['training_geometry_enabled']),
        'candidate_probing_final': probing_final,
        'checkpoint_bytes': _checkpoint_bytes(run_dir, arm),
    }


def _compare_ranked(left: dict[str, Any], right: dict[str, Any]) -> int:
    """Compare arms by the locked lexicographic order."""
    left_final = left['external_evaluation'][str(ENDPOINTS[-1])]
    right_final = right['external_evaluation'][str(ENDPOINTS[-1])]
    left_primary = (
        left_final['cost_cvar_20'],
        left_final['mean_cost'],
        -left_final['goal_met_episode_rate'],
        left_final['stall_ratio'],
    )
    right_primary = (
        right_final['cost_cvar_20'],
        right_final['mean_cost'],
        -right_final['goal_met_episode_rate'],
        right_final['stall_ratio'],
    )
    if left_primary != right_primary:
        return -1 if left_primary < right_primary else 1
    if left['mechanisms']['bellman_probability'] and right['mechanisms']['bellman_probability']:
        left_heldout = left['heldout_probability_final'] or {}
        right_heldout = right['heldout_probability_final'] or {}
        left_brier = optional_number(left_heldout.get('Heldout/BrierSkill'))
        left_auroc = optional_number(left_heldout.get('Heldout/AUROC'))
        right_brier = optional_number(right_heldout.get('Heldout/BrierSkill'))
        right_auroc = optional_number(right_heldout.get('Heldout/AUROC'))
        left_secondary = (
            -left_brier if left_brier is not None else float('inf'),
            -left_auroc if left_auroc is not None else float('inf'),
        )
        right_secondary = (
            -right_brier if right_brier is not None else float('inf'),
            -right_auroc if right_auroc is not None else float('inf'),
        )
        if left_secondary != right_secondary:
            return -1 if left_secondary < right_secondary else 1
    return -1 if left['arm'] < right['arm'] else int(left['arm'] > right['arm'])


def screen_decision(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply B1-relative guards, rank arms, and select exactly two extensions."""
    baseline = arms['B1']['external_evaluation'][str(ENDPOINTS[-1])]
    for arm in ARMS:
        evidence = arms[arm]
        endpoint = evidence['external_evaluation'][str(ENDPOINTS[-1])]
        criteria = {
            'completed_exactly_50k': True,
            'finite_core_training': evidence['training_core_finiteness']['all_finite'],
            'goal_met_rate_within_0_10_of_B1': (
                endpoint['goal_met_episode_rate'] >= baseline['goal_met_episode_rate'] - 0.10
            ),
            'stall_ratio_within_0_10_of_B1': endpoint['stall_ratio']
            <= baseline['stall_ratio'] + 0.10,
            'reward_td_not_divergent': not evidence['reward_td_trend']['monotonic_divergence'],
        }
        evidence['performance_criteria'] = criteria
        evidence['performance_eligible'] = all(criteria.values())

    ranked_all = sorted(arms.values(), key=functools.cmp_to_key(_compare_ranked))
    ranked_eligible = [evidence for evidence in ranked_all if evidence['performance_eligible']]
    selected = ranked_eligible[:2]
    if len(selected) < 2:
        selected.extend(evidence for evidence in ranked_all if evidence not in selected)
        selected = selected[:2]
    selected_payload = [
        {
            'arm': evidence['arm'],
            'selection_basis': 'performance_eligible'
            if evidence['performance_eligible']
            else 'fallback',
        }
        for evidence in selected
    ]
    return {
        'baseline_arm': 'B1',
        'ranked_completed_arms': [evidence['arm'] for evidence in ranked_all],
        'ranked_performance_eligible_arms': [evidence['arm'] for evidence in ranked_eligible],
        'selected_for_100k_x3': selected_payload,
        'decision': 'extend_selected_arms_to_100k_x3',
    }


def summarize_screen(
    run_dirs: dict[str, Path],
    receipts: dict[str, Path],
    evaluation_csvs: dict[str, Path],
    preregistration_path: Path,
    training_source_commit: str,
) -> dict[str, Any]:
    """Build the complete strict screen decision payload."""
    preregistration = read_json(preregistration_path)
    if preregistration.get('protocol_status') != 'locked_before_training':
        raise ValueError('B1--B4 protocol was not locked before training')
    if int(preregistration['screen']['seed']) != SCREEN_SEED:
        raise ValueError('preregistration has the wrong screen seed')
    if int(preregistration['screen']['steps']) != ENDPOINTS[-1]:
        raise ValueError('preregistration has the wrong screen horizon')
    arms = {
        arm: arm_summary(
            arm,
            run_dirs[arm],
            receipts[arm],
            evaluation_csvs[arm],
            preregistration,
            training_source_commit,
        )
        for arm in ARMS
    }
    return {
        'schema_version': 1,
        'stage': 'P4.2-B1-B4-causal-screen',
        'protocol_version': preregistration['protocol_version'],
        'training_source_commit': training_source_commit,
        'screen_seed': SCREEN_SEED,
        'target_steps': ENDPOINTS[-1],
        'arms': arms,
        'selection': screen_decision(arms),
        'extension_protocol': preregistration['selection'],
        'final_p4_verdict': None,
        'formal_four_environment_experiment_launched': False,
        'continuation_beyond_100k_launched': False,
    }


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run', action='append', required=True, help='ARM=run-directory')
    result.add_argument('--receipt', action='append', required=True, help='ARM=receipt.json')
    result.add_argument('--evaluation', action='append', required=True, help='ARM=evaluation.csv')
    result.add_argument('--preregistration', type=Path, required=True)
    result.add_argument('--training-source-commit', required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main() -> None:
    """Write the strict machine-readable mechanism-screen decision."""
    args = parser().parse_args()
    payload = summarize_screen(
        keyed_paths(args.run, '--run'),
        keyed_paths(args.receipt, '--receipt'),
        keyed_paths(args.evaluation, '--evaluation'),
        args.preregistration,
        args.training_source_commit,
    )
    atomic_json(args.output, payload)
    print(
        {
            'decision': payload['selection']['decision'],
            'selected': payload['selection']['selected_for_100k_x3'],
            'output': str(args.output),
        },
    )


if __name__ == '__main__':
    main()
