#!/usr/bin/env python3
"""Summarize the preregistered P4.2 top-two 100k-by-three extension."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from scripts.star.summarize_p4_2_backbone import (
    atomic_json,
    number,
    optional_number,
    read_csv,
    read_json,
)
from scripts.star.summarize_p4_2_mechanism_screen import (
    CORE_TRAINING_COLUMNS,
    EXTERNAL_METRICS,
    HELDOUT_METRICS,
    PROBING_METRICS,
)


SEEDS = (2027, 2028, 2029)
TARGET_STEPS = 100_000
RESUME_SEED = 2028
LOG_INTERVAL = 5_000


def selected_arms(screen_decision: dict[str, Any]) -> tuple[str, str]:
    """Return the exact two preregistered extension arms."""
    selected = tuple(item['arm'] for item in screen_decision['selection']['selected_for_100k_x3'])
    if len(selected) != 2 or len(set(selected)) != 2:
        raise ValueError('screen decision must select exactly two distinct arms')
    return selected


def keyed_paths(
    values: list[str],
    arms: tuple[str, str],
    label: str,
) -> dict[tuple[str, int], Path]:
    """Parse six explicit ``ARM:SEED=PATH`` bindings."""
    expected = {(arm, seed) for arm in arms for seed in SEEDS}
    result: dict[tuple[str, int], Path] = {}
    for value in values:
        key, separator, raw_path = value.partition('=')
        arm, colon, raw_seed = key.partition(':')
        if separator != '=' or colon != ':' or not raw_path:
            raise ValueError(f'{label} entries must use ARM:SEED=PATH')
        try:
            identity = (arm, int(raw_seed))
        except ValueError as error:
            raise ValueError(f'{label} has a non-integer seed in {value!r}') from error
        if identity not in expected or identity in result:
            raise ValueError(f'{label} has an unexpected or duplicate identity {identity}')
        result[identity] = Path(raw_path)
    if set(result) != expected:
        raise ValueError(f'{label} must bind exactly the selected arms at seeds {SEEDS}')
    return result


def expected_diagnostic_steps(seed: int) -> tuple[int, ...]:
    """Return fresh or resumed 5k logging coverage."""
    start = 55_000 if seed == RESUME_SEED else LOG_INTERVAL
    return tuple(range(start, TARGET_STEPS + 1, LOG_INTERVAL))


def reward_td_trend(
    rows: list[dict[str, str]],
    arm: str,
    seed: int,
    screen_historical_minimum: float | None = None,
) -> dict[str, Any]:
    """Apply the registered 1.5-times-minimum and rising-tail diagnostic."""
    ordered = sorted(rows, key=lambda row: int(number(row['TotalEnvSteps'], 'TotalEnvSteps')))
    steps = [int(number(row['TotalEnvSteps'], f'{arm}.{seed}.TotalEnvSteps')) for row in ordered]
    if tuple(steps) != expected_diagnostic_steps(seed):
        raise ValueError(f'{arm} seed {seed} has incorrect 5k diagnostic coverage')
    values = [number(row['Critic/RewardTD'], f'{arm}.{seed}.Critic/RewardTD') for row in ordered]
    candidates = values + (
        [number(screen_historical_minimum, f'{arm}.screen_historical_minimum')]
        if screen_historical_minimum is not None
        else []
    )
    historical_minimum = min(candidates)
    if historical_minimum <= 0.0:
        raise ValueError(f'{arm} seed {seed} reward TD minimum must be positive')
    last_four = values[-4:]
    strictly_rising = all(after > before for before, after in zip(last_four, last_four[1:]))
    ratio = values[-1] / historical_minimum
    return {
        'historical_minimum': historical_minimum,
        'final': values[-1],
        'final_to_historical_minimum_ratio': ratio,
        'last_four_steps': steps[-4:],
        'last_four_values': last_four,
        'strictly_rising_last_three_intervals': strictly_rising,
        'monotonic_divergence': strictly_rising and ratio > 1.5,
        'includes_screen_history': screen_historical_minimum is not None,
    }


def finite_training_core(rows: list[dict[str, str]], arm: str, seed: int) -> dict[str, Any]:
    """Require finite optimizer signals while allowing undefined held-out statistics."""
    for row_index, row in enumerate(rows):
        for column in CORE_TRAINING_COLUMNS:
            number(row[column], f'{arm}.{seed}.diagnostics[{row_index}].{column}')
    return {'columns': list(CORE_TRAINING_COLUMNS), 'rows': len(rows), 'all_finite': True}


def heldout_history(path: Path, arm: str, seed: int) -> list[dict[str, Any]]:
    """Validate the extension's held-out-only evidence series."""
    rows = read_csv(path)
    steps = [int(number(row['env_steps'], f'{arm}.{seed}.heldout.env_steps')) for row in rows]
    if tuple(steps) != expected_diagnostic_steps(seed):
        raise ValueError(f'{arm} seed {seed} has incorrect held-out coverage')
    result: list[dict[str, Any]] = []
    for row, step in zip(rows, steps):
        if row['split'] != 'heldout_episode':
            raise ValueError(f'{arm} seed {seed} did not use held-out geometry evidence')
        if int(number(row['recent_episode_limit'], 'recent_episode_limit')) != 20:
            raise ValueError(f'{arm} seed {seed} has the wrong held-out episode limit')
        if int(number(row['transition_limit'], 'transition_limit')) != 4096:
            raise ValueError(f'{arm} seed {seed} has the wrong held-out transition limit')
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


def external_final(path: Path, arm: str, seed: int, evaluation_seeds: list[int]) -> dict[str, Any]:
    """Validate one isolated 100k checkpoint evaluation."""
    rows = read_csv(path)
    if len(rows) != 1:
        raise ValueError(f'{arm} seed {seed} evaluation must contain one 100k row')
    row = rows[0]
    if int(number(row['training_seed'], 'training_seed')) != seed:
        raise ValueError(f'{arm} seed {seed} evaluation has the wrong training seed')
    if int(number(row['milestone'], 'milestone')) != TARGET_STEPS:
        raise ValueError(f'{arm} seed {seed} evaluation has the wrong milestone')
    if row['evaluation_mode'] != 'fixed-random-flow' or row['evaluation_role'] != 'primary':
        raise ValueError(f'{arm} seed {seed} evaluation has the wrong mode')
    if row.get('probability_split') != 'external_isolated_environment':
        raise ValueError(f'{arm} seed {seed} evaluation has the wrong split')
    if int(number(row['episodes'], 'episodes')) != len(evaluation_seeds):
        raise ValueError(f'{arm} seed {seed} evaluation has the wrong episode count')
    raw = read_json(path.parent / f'evaluation_seed_{seed}_step_{TARGET_STEPS}.json')
    if [int(value) for value in raw.get('seeds', [])] != evaluation_seeds:
        raise ValueError(f'{arm} seed {seed} raw evaluation has the wrong seeds')
    if raw.get('probability_split') != 'external_isolated_environment':
        raise ValueError(f'{arm} seed {seed} raw evaluation has the wrong split')
    result: dict[str, Any] = {
        metric: number(row[metric], f'{arm}.{seed}.{metric}') for metric in EXTERNAL_METRICS
    }
    result.update(
        {
            'episodes': len(evaluation_seeds),
            'goal_met_total': int(number(row['goal_met_total'], 'goal_met_total')),
            'external_brier_skill': optional_number(row.get('brier_skill')),
            'external_auroc': optional_number(row.get('auroc')),
        },
    )
    return result


def checkpoint_bytes(run_dir: Path, seed: int) -> dict[str, int]:
    """Require fresh/resumed milestone and recovery checkpoints."""
    filenames = ['milestone_100000.pt', 'latest_full.pt', 'final_full.pt']
    if seed != RESUME_SEED:
        filenames = ['milestone_0.pt', 'milestone_50000.pt', *filenames]
    paths = {name[:-3]: run_dir / 'checkpoints' / name for name in filenames}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'seed {seed} is missing extension checkpoints: {missing}')
    return {label: path.stat().st_size for label, path in paths.items()}


def run_summary(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    arm: str,
    seed: int,
    run_dir: Path,
    receipt_path: Path,
    evaluation_csv: Path,
    screen_decision: dict[str, Any],
    preregistration: dict[str, Any],
    training_source_commit: str,
) -> dict[str, Any]:
    """Validate one selected-arm extension run."""
    receipt = read_json(receipt_path)
    summary = read_json(run_dir / 'artifacts/training_summary.json')
    manifest = read_json(run_dir / 'artifacts/run_manifest.json')
    config = read_json(run_dir / 'config.json')
    diagnostics = read_csv(run_dir / 'artifacts/diagnostics.csv')
    registered = preregistration['arms'][arm]
    expected_run_steps = 50_000 if seed == RESUME_SEED else TARGET_STEPS
    if receipt.get('status') != 'passed' or summary.get('status') != 'completed':
        raise ValueError(f'{arm} seed {seed} did not complete')
    if int(receipt['seed']) != seed or int(summary['seed']) != seed:
        raise ValueError(f'{arm} seed {seed} receipt identity mismatch')
    if int(receipt['steps']) != TARGET_STEPS or int(summary['env_steps']) != TARGET_STEPS:
        raise ValueError(f'{arm} seed {seed} did not stop at 100k')
    if (
        int(receipt['run_steps']) != expected_run_steps
        or int(summary['run_env_steps']) != expected_run_steps
    ):
        raise ValueError(f'{arm} seed {seed} has incorrect fresh/resume run counters')
    if int(receipt['updates']) != 98_000 or int(summary['updates']) != 98_000:
        raise ValueError(f'{arm} seed {seed} has incorrect update accounting')
    if (
        receipt.get('source_commit') != training_source_commit
        or receipt.get('source_dirty') is not False
    ):
        raise ValueError(f'{arm} seed {seed} has the wrong or dirty source')
    if receipt.get('config_sha256') != registered['config_sha256']:
        raise ValueError(f'{arm} seed {seed} config fingerprint mismatch')
    resume_from = str(config['algo_cfgs']['resume_from'])
    if seed == RESUME_SEED:
        resume_path = Path(resume_from)
        if (
            not resume_path.is_file()
            or resume_path.name != 'final_full.pt'
            or arm not in resume_path.parts
            or 'p4.2-b1-b4-screen-5e994d7' not in resume_path.parts
        ):
            raise ValueError(f'{arm} seed 2028 did not resume the exact screen checkpoint')
    elif resume_from:
        raise ValueError(f'{arm} seed {seed} should start fresh')
    if tuple(int(step) for step in manifest['eval_milestones']) != (0, 50_000, TARGET_STEPS):
        raise ValueError(f'{arm} seed {seed} has incorrect milestones')
    if manifest['p4_2_probability_critic']['bellman_probability_enabled'] is not bool(
        registered['bellman_probability'],
    ):
        raise ValueError(f'{arm} seed {seed} has the wrong Bellman switch')
    if manifest['p4_2_dynamic_geometry']['enabled'] is not bool(registered['dynamic_geometry']):
        raise ValueError(f'{arm} seed {seed} has the wrong dynamic-geometry switch')
    if manifest['p4_2_candidate_probing']['enabled'] is not bool(registered['candidate_probing']):
        raise ValueError(f'{arm} seed {seed} has the wrong probing switch')

    screen_minimum = (
        number(
            screen_decision['arms'][arm]['reward_td_trend']['historical_minimum'],
            f'{arm}.screen_historical_minimum',
        )
        if seed == RESUME_SEED
        else None
    )
    trend = reward_td_trend(diagnostics, arm, seed, screen_minimum)
    finite = finite_training_core(diagnostics, arm, seed)
    evaluation = external_final(
        evaluation_csv,
        arm,
        seed,
        [int(value) for value in preregistration['external_evaluation']['seeds']],
    )
    history = (
        heldout_history(run_dir / 'artifacts/heldout_probability.csv', arm, seed)
        if registered['bellman_probability']
        else []
    )
    probing_final: dict[str, float] | None = None
    dangerous_velocity_change: dict[str, float] | None = None
    if registered['candidate_probing']:
        for row_index, row in enumerate(diagnostics):
            for metric in PROBING_METRICS:
                number(row[metric], f'{arm}.{seed}.diagnostics[{row_index}].{metric}')
        probing_final = {
            metric: number(diagnostics[-1][metric], f'{arm}.{seed}.final.{metric}')
            for metric in PROBING_METRICS
        }
        initial_velocity = (
            number(
                screen_decision['arms'][arm]['candidate_probing_final'][
                    'Probing/DangerousNormalVelocity'
                ],
                f'{arm}.screen_dangerous_velocity',
            )
            if seed == RESUME_SEED
            else number(
                diagnostics[0]['Probing/DangerousNormalVelocity'],
                f'{arm}.{seed}.initial_dangerous_velocity',
            )
        )
        dangerous_velocity_change = {
            'initial': initial_velocity,
            'final': probing_final['Probing/DangerousNormalVelocity'],
            'change': probing_final['Probing/DangerousNormalVelocity'] - initial_velocity,
        }
    return {
        'seed': seed,
        'resume_mode': 'screen_50k_full_checkpoint' if seed == RESUME_SEED else 'fresh',
        'source_commit': receipt['source_commit'],
        'config_sha256': receipt['config_sha256'],
        'resolved_overrides_sha256': receipt['resolved_overrides_sha256'],
        'training_seconds': number(summary['training_seconds'], f'{arm}.{seed}.training_seconds'),
        'estimated_cost_cny': number(receipt['estimated_cost_cny'], f'{arm}.{seed}.cost'),
        'training_core_finiteness': finite,
        'reward_td_trend': trend,
        'external_evaluation_100k': evaluation,
        'heldout_probability_final': (
            summary.get('heldout_probability_latest') if registered['bellman_probability'] else None
        ),
        'heldout_probability_history': history,
        'geometry_activation_step': summary.get('geometry_activation_step'),
        'geometry_enabled_at_end': bool(summary['training_geometry_enabled']),
        'candidate_probing_final': probing_final,
        'dangerous_normal_velocity_change': dangerous_velocity_change,
        'checkpoint_bytes': checkpoint_bytes(run_dir, seed),
    }


def pooled_external(runs: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Pool one arm's three fixed-seed endpoint summaries."""
    metrics = (*EXTERNAL_METRICS, 'external_brier_skill', 'external_auroc')
    result: dict[str, float] = {}
    for metric in metrics:
        values = [
            optional_number(runs[str(seed)]['external_evaluation_100k'].get(metric))
            for seed in SEEDS
        ]
        finite = [value for value in values if value is not None]
        if finite:
            result[metric] = statistics.fmean(finite)
    return result


def p4_1_reference(path: Path) -> dict[str, float]:
    """Read the sealed P4.1 100k stall and dangerous-velocity references."""
    report = read_json(path)
    endpoint = next(
        row
        for row in report['external_evaluation']['pooled_by_milestone']
        if int(row['milestone']) == TARGET_STEPS
    )
    dangerous = [
        number(item['counterfactual']['final_dangerous_normal_velocity'], 'p4_1.dangerous')
        for item in report['seeds']
    ]
    return {
        'stall_ratio': number(endpoint['stall_ratio'], 'p4_1.stall_ratio'),
        'dangerous_normal_velocity': statistics.fmean(dangerous),
    }


def extension_decision(
    arms: dict[str, dict[str, Any]],
    screen_decision: dict[str, Any],
    p4_1: dict[str, float],
) -> dict[str, Any]:
    """Evaluate the preregistered B4-before-P5 guard without launching P5."""
    if 'B4' not in arms or 'B1' not in arms:
        return {'p5_entry_guard_applicable': False, 'p5_entry_guard_passed': False}
    b4 = arms['B4']
    b1 = arms['B1']
    b4_runs = b4['runs']
    screen_success = number(
        screen_decision['arms']['B4']['external_evaluation']['50000']['goal_met_episode_rate'],
        'B4.screen_success',
    )
    heldout_brier = [
        optional_number(b4_runs[str(seed)]['heldout_probability_final'].get('Heldout/BrierSkill'))
        for seed in SEEDS
    ]
    heldout_auroc = [
        optional_number(b4_runs[str(seed)]['heldout_probability_final'].get('Heldout/AUROC'))
        for seed in SEEDS
    ]
    boundaries = [
        optional_number(
            b4_runs[str(seed)]['heldout_probability_final'].get('Heldout/BoundarySamples'),
        )
        for seed in SEEDS
    ]
    final_velocities = [
        number(
            b4_runs[str(seed)]['candidate_probing_final']['Probing/DangerousNormalVelocity'],
            f'B4.{seed}.dangerous_velocity',
        )
        for seed in SEEDS
    ]
    criteria = {
        'success_does_not_collapse_vs_B4_50k': (
            b4['pooled_external_100k']['goal_met_episode_rate'] >= screen_success - 0.10
        ),
        'stall_lower_than_P4_1_100k': (
            b4['pooled_external_100k']['stall_ratio'] < p4_1['stall_ratio']
        ),
        'reward_td_not_divergent_every_seed': all(
            not b4_runs[str(seed)]['reward_td_trend']['monotonic_divergence'] for seed in SEEDS
        ),
        'heldout_brier_skill_positive_every_seed': all(
            value is not None and value > 0.0 for value in heldout_brier
        ),
        'heldout_auroc_at_least_0_65_every_seed': all(
            value is not None and value >= 0.65 for value in heldout_auroc
        ),
        'heldout_boundary_samples_at_least_128_every_seed': all(
            value is not None and value >= 128 for value in boundaries
        ),
        'dangerous_normal_velocity_lower_than_P4_1': (
            statistics.fmean(final_velocities) < p4_1['dangerous_normal_velocity']
        ),
        'cost_lower_than_B1': (
            b4['pooled_external_100k']['mean_cost'] < b1['pooled_external_100k']['mean_cost']
        ),
        'cost_cvar_20_lower_than_B1': (
            b4['pooled_external_100k']['cost_cvar_20'] < b1['pooled_external_100k']['cost_cvar_20']
        ),
        'success_loss_vs_B1_no_more_than_0_10': (
            b4['pooled_external_100k']['goal_met_episode_rate']
            >= b1['pooled_external_100k']['goal_met_episode_rate'] - 0.10
        ),
    }
    return {
        'p5_entry_guard_applicable': True,
        'criteria': criteria,
        'p5_entry_guard_passed': all(criteria.values()),
        'observations': {
            'B4_screen_50k_success': screen_success,
            'B4_heldout_brier_skill_by_seed': dict(
                zip((str(seed) for seed in SEEDS), heldout_brier),
            ),
            'B4_heldout_auroc_by_seed': dict(zip((str(seed) for seed in SEEDS), heldout_auroc)),
            'B4_boundary_samples_by_seed': dict(zip((str(seed) for seed in SEEDS), boundaries)),
            'B4_dangerous_normal_velocity_by_seed': dict(
                zip((str(seed) for seed in SEEDS), final_velocities),
            ),
            'P4_1_reference': p4_1,
        },
    }


def summarize_extension(  # pylint: disable=too-many-arguments
    run_dirs: dict[tuple[str, int], Path],
    receipts: dict[tuple[str, int], Path],
    evaluation_csvs: dict[tuple[str, int], Path],
    screen_decision_path: Path,
    preregistration_path: Path,
    p4_1_report_path: Path,
    training_source_commit: str,
) -> dict[str, Any]:
    """Build the strict top-two extension payload."""
    screen = read_json(screen_decision_path)
    preregistration = read_json(preregistration_path)
    arms = selected_arms(screen)
    evidence: dict[str, dict[str, Any]] = {}
    for arm in arms:
        runs = {
            str(seed): run_summary(
                arm,
                seed,
                run_dirs[(arm, seed)],
                receipts[(arm, seed)],
                evaluation_csvs[(arm, seed)],
                screen,
                preregistration,
                training_source_commit,
            )
            for seed in SEEDS
        }
        evidence[arm] = {
            'runs': runs,
            'pooled_external_100k': pooled_external(runs),
            'training_seconds': sum(run['training_seconds'] for run in runs.values()),
            'estimated_cost_cny': sum(run['estimated_cost_cny'] for run in runs.values()),
        }
    p4_1 = p4_1_reference(p4_1_report_path)
    decision = extension_decision(evidence, screen, p4_1)
    return {
        'schema_version': 1,
        'stage': 'P4.2-top-two-100k-three-seed-extension',
        'training_source_commit': training_source_commit,
        'selected_arms': list(arms),
        'seeds': list(SEEDS),
        'target_steps': TARGET_STEPS,
        'arms': evidence,
        'p5_entry_guard': decision,
        'final_p4_verdict': None,
        'continuation_beyond_100k_launched': False,
        'formal_four_environment_experiment_launched': False,
    }


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run', action='append', required=True, help='ARM:SEED=run-directory')
    result.add_argument('--receipt', action='append', required=True, help='ARM:SEED=receipt.json')
    result.add_argument(
        '--evaluation',
        action='append',
        required=True,
        help='ARM:SEED=evaluation.csv',
    )
    result.add_argument('--screen-decision', type=Path, required=True)
    result.add_argument('--preregistration', type=Path, required=True)
    result.add_argument('--p4-1-report', type=Path, required=True)
    result.add_argument('--training-source-commit', required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main() -> None:
    """Write the machine-readable extension and pre-P5 decision."""
    args = parser().parse_args()
    screen = read_json(args.screen_decision)
    arms = selected_arms(screen)
    payload = summarize_extension(
        keyed_paths(args.run, arms, '--run'),
        keyed_paths(args.receipt, arms, '--receipt'),
        keyed_paths(args.evaluation, arms, '--evaluation'),
        args.screen_decision,
        args.preregistration,
        args.p4_1_report,
        args.training_source_commit,
    )
    atomic_json(args.output, payload)
    print(
        {
            'selected_arms': payload['selected_arms'],
            'p5_entry_guard_passed': payload['p5_entry_guard']['p5_entry_guard_passed'],
            'output': str(args.output),
        },
    )


if __name__ == '__main__':
    main()
