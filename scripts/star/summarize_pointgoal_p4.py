#!/usr/bin/env python3
"""Aggregate STAR PointGoal P4 artifacts and evaluate the preregistered criteria."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


EVALUATION_METRICS = (
    'mean_return',
    'mean_cost',
    'cost_cvar_20',
    'violation_ratio',
    'calibration_error',
    'brier_score',
    'brier_skill',
    'auroc',
    'spearman',
    'goals_per_episode_mean',
    'stall_ratio',
    'mean_start_end_displacement',
    'mean_action_norm',
)


def read_json(path: Path) -> dict[str, Any]:
    """Read one strict JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'expected a JSON object in {path}')
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV artifact."""
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def number(value: Any) -> float:
    """Convert a value to a finite float."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'expected a finite number, got {value!r}')
    return result


def optional_number(value: Any) -> float | None:
    """Convert a present finite value while preserving undefined CSV cells."""
    return None if value is None or value == '' else number(value)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows atomically with their stable first-row field order."""
    if not rows:
        return
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON atomically."""
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def final_counterfactual(run_dir: Path) -> dict[str, Any]:
    """Return the counterfactual artifact at the largest milestone."""
    paths = list((run_dir / 'artifacts').glob('counterfactual_step_*.json'))
    if not paths:
        raise FileNotFoundError(f'no counterfactual artifacts in {run_dir}')
    path = max(paths, key=lambda item: int(item.stem.rsplit('_', maxsplit=1)[-1]))
    payload = read_json(path)
    if not payload.get('available'):
        raise ValueError(f'counterfactual is unavailable in {path}')
    return payload


def evaluation_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    """Collect fixed-seed evaluation rows from every training seed."""
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        manifest = read_json(run_dir / 'artifacts/run_manifest.json')
        seed = int(manifest['seed'])
        for source in read_csv(run_dir / 'artifacts/evaluation.csv'):
            row: dict[str, Any] = {
                'seed': seed,
                'milestone': int(source['milestone']),
                'episodes': int(source['episodes']),
            }
            row.update({metric: number(source[metric]) for metric in EVALUATION_METRICS})
            rows.append(row)
    return sorted(rows, key=lambda row: (row['milestone'], row['seed']))


def checkpoint_evaluation_rows(path: Path) -> list[dict[str, Any]]:
    """Read the tie-correct fixed-device checkpoint evaluation table."""
    rows = []
    for source in read_csv(path):
        row: dict[str, Any] = {
            'seed': int(source['training_seed']),
            'milestone': int(source['milestone']),
            'episodes': int(source['episodes']),
        }
        row.update({metric: optional_number(source[metric]) for metric in EVALUATION_METRICS})
        rows.append(row)
    return sorted(rows, key=lambda row: (row['milestone'], row['seed']))


def aggregate_evaluations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate run-level evaluation metrics by milestone."""
    milestones = sorted({int(row['milestone']) for row in rows})
    aggregate: list[dict[str, Any]] = []
    for milestone in milestones:
        selected = [row for row in rows if int(row['milestone']) == milestone]
        output: dict[str, Any] = {'milestone': milestone, 'seed_count': len(selected)}
        for metric in EVALUATION_METRICS:
            values = [number(row[metric]) for row in selected if row[metric] is not None]
            output[f'{metric}_count'] = len(values)
            output[f'{metric}_mean'] = statistics.fmean(values) if values else None
            output[f'{metric}_std'] = (
                statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
            )
            output[f'{metric}_min'] = min(values) if values else None
            output[f'{metric}_max'] = max(values) if values else None
        aggregate.append(output)
    return aggregate


def group_run_segments(run_dirs: list[Path]) -> dict[int, list[Path]]:
    """Group chronologically contiguous training segments by seed."""
    grouped: dict[int, list[tuple[int, Path, dict[str, Any]]]] = {}
    for run_dir in run_dirs:
        training = read_json(run_dir / 'artifacts/training_summary.json')
        seed = int(training['seed'])
        grouped.setdefault(seed, []).append((int(training['env_steps']), run_dir, training))
    if not grouped:
        raise ValueError('at least one run directory is required')

    result: dict[int, list[Path]] = {}
    for seed, entries in grouped.items():
        entries.sort(key=lambda item: item[0])
        endpoints = [entry[0] for entry in entries]
        if len(endpoints) != len(set(endpoints)):
            raise ValueError(f'run segments for seed {seed} have duplicate endpoints')
        previous_endpoint: int | None = None
        for endpoint, run_dir, training in entries:
            run_steps = training.get('run_env_steps')
            if len(entries) > 1 and run_steps is None:
                raise ValueError(f'run segment {run_dir} is missing run_env_steps')
            if run_steps is not None:
                start = endpoint - int(run_steps)
                if start < 0:
                    raise ValueError(f'run segment {run_dir} has invalid step counters')
                if previous_endpoint is None and len(entries) > 1 and start != 0:
                    raise ValueError(f'run chain for seed {seed} does not start at zero')
                if previous_endpoint is not None and start != previous_endpoint:
                    raise ValueError(f'run chain for seed {seed} is not contiguous at {run_dir}')
            previous_endpoint = endpoint
        result[seed] = [entry[1] for entry in entries]
    return result


def segment_diagnostics(run_dirs: list[Path]) -> list[dict[str, str]]:
    """Merge diagnostics from an ordered run chain without hiding duplicate steps."""
    rows = [row for run_dir in run_dirs for row in read_csv(run_dir / 'artifacts/diagnostics.csv')]
    if rows and all(row.get('TotalEnvSteps', '') != '' for row in rows):
        steps = [int(number(row['TotalEnvSteps'])) for row in rows]
        if len(steps) != len(set(steps)):
            raise ValueError('run chain diagnostics contain duplicate environment steps')
        rows = [row for _, row in sorted(zip(steps, rows), key=lambda item: item[0])]
    return rows


def mechanism_row(  # pylint: disable=too-many-locals
    run_dirs: list[Path],
    audit: dict[str, Any],
    receipts: list[dict[str, Any]],
    fixed_evaluations: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Collect one seed's mechanism evidence and evaluate its P4 conditions."""
    run_dir = run_dirs[-1]
    training_segments = [
        read_json(segment / 'artifacts/training_summary.json') for segment in run_dirs
    ]
    training = read_json(run_dir / 'artifacts/training_summary.json')
    warmup = read_json(run_dir / 'artifacts/warmup_status.json')
    counterfactual = final_counterfactual(run_dir)
    diagnostics = segment_diagnostics(run_dirs)
    evaluation_source = (
        fixed_evaluations
        if fixed_evaluations is not None
        else [row for segment in run_dirs for row in read_csv(segment / 'artifacts/evaluation.csv')]
    )
    evaluations = sorted(evaluation_source, key=lambda row: int(row['milestone']))
    if not diagnostics or len(evaluations) < 2:
        raise ValueError(f'incomplete P4 artifacts in {run_dir}')
    seed = int(training['seed'])
    if int(audit['seed']) != seed or any(int(receipt['seed']) != seed for receipt in receipts):
        raise ValueError(f'seed mismatch while aggregating run {seed}')

    w2_rows = [row for row in diagnostics if int(number(row['Train/WarmupStage'])) == 2]
    if not w2_rows:
        raise ValueError(f'run {seed} has no W2 diagnostic row')
    final_diagnostic = diagnostics[-1]
    last_w2 = w2_rows[-1]
    initial_evaluation = evaluations[0]
    final_evaluation = evaluations[-1]
    if int(initial_evaluation['milestone']) != 0 or int(final_evaluation['milestone']) != 300_000:
        raise ValueError(f'run {seed} is missing the fixed 0/300000 evaluation endpoints')

    boundary_gate = number(counterfactual['gate_boundary_mean'])
    interior_gate = number(counterfactual['gate_interior_mean'])
    gate_ratio = boundary_gate / max(interior_gate, 1e-12)
    w2_danger = number(last_w2['Geometry/DangerousNormalVelocity'])
    final_danger = number(final_diagnostic['Geometry/DangerousNormalVelocity'])
    replay_spearman = optional_number(
        audit['predicted_cost_empirical_cost_spearman'],
    )
    activation_step = warmup['geometry_activation_step']

    row = {
        'seed': seed,
        'status': training['status'],
        'env_steps': int(training['env_steps']),
        'updates': int(training['updates']),
        'segment_count': len(run_dirs),
        'training_seconds': sum(
            number(segment['training_seconds']) for segment in training_segments
        ),
        'estimated_cost_cny': sum(number(receipt['estimated_cost_cny']) for receipt in receipts),
        'warmup_stage': warmup['stage'],
        'geometry_activation_step': (int(activation_step) if activation_step is not None else None),
        'boundary_samples': int(warmup['boundary_samples']),
        'calibration_ema': number(warmup['ema']['calibration_error']),
        'cost_td_ema': number(warmup['ema']['cost_td_error']),
        'valid_gradient_ema': number(warmup['ema']['valid_gradient_rate']),
        'ensemble_disagreement_ema': number(warmup['ema']['ensemble_disagreement']),
        'brier_skill_ema': number(warmup['ema']['brier_skill']),
        'auroc_ema': number(warmup['ema']['auroc']),
        'gate_boundary_mean': boundary_gate,
        'gate_interior_mean': interior_gate,
        'gate_boundary_interior_ratio': gate_ratio,
        'isotropic_vn': number(counterfactual['isotropic_vn']),
        'directional_vn': number(counterfactual['directional_vn']),
        'isotropic_vt': number(counterfactual['isotropic_vt']),
        'directional_vt': number(counterfactual['directional_vt']),
        'vt_preservation_ratio': number(counterfactual['vt_preservation_ratio']),
        'last_w2_dangerous_normal_velocity': w2_danger,
        'final_dangerous_normal_velocity': final_danger,
        'dangerous_normal_velocity_change': final_danger - w2_danger,
        'replay_calibration_ece': number(audit['calibration_ece']),
        'replay_calibration_mae': number(audit['calibration_mae']),
        'replay_cost_spearman': replay_spearman,
        'replay_brier_skill': optional_number(audit['brier_skill']),
        'replay_auroc': optional_number(audit['auroc']),
        'replay_positive_labels': int(audit['positive_labels']),
        'replay_negative_labels': int(audit['negative_labels']),
        'boundary_source_random_fraction': number(
            audit['resident_boundary_source_random_fraction'],
        ),
        'boundary_immediate_violation_fraction': number(
            audit['resident_boundary_immediate_violation_fraction'],
        ),
        'initial_eval_return': number(initial_evaluation['mean_return']),
        'final_eval_return': number(final_evaluation['mean_return']),
        'initial_eval_cost': number(initial_evaluation['mean_cost']),
        'final_eval_cost': number(final_evaluation['mean_cost']),
        'initial_eval_cvar': number(initial_evaluation['cost_cvar_20']),
        'final_eval_cvar': number(final_evaluation['cost_cvar_20']),
        'final_eval_goals_per_episode': number(final_evaluation['goals_per_episode_mean']),
        'final_eval_stall_ratio': number(final_evaluation['stall_ratio']),
        'final_eval_displacement': number(final_evaluation['mean_start_end_displacement']),
        'final_eval_action_norm': number(final_evaluation['mean_action_norm']),
    }
    criteria = {
        'completed_without_crash': all(
            segment['status'] == 'completed' for segment in training_segments
        )
        and int(training['env_steps']) == 300_000,
        'mean_goal_met_positive': row['final_eval_goals_per_episode'] > 0.0,
        'no_stall': row['final_eval_stall_ratio'] <= 1e-12,
        'cost_within_budget': row['final_eval_cost'] <= 10.0,
        'probability_auroc_above_065': row['replay_auroc'] is not None
        and row['replay_auroc'] > 0.65,
        'positive_brier_skill': row['replay_brier_skill'] is not None
        and row['replay_brier_skill'] > 0.0,
        'tangent_variance_preserved': row['vt_preservation_ratio'] >= 0.8,
        'dangerous_normal_velocity_reduced': final_danger < w2_danger,
        'final_warmup_evidence_passed': warmup['stage'] == 'W3'
        and all(bool(value) for value in warmup['conditions'].values()),
    }
    return row, criteria


def summarize(  # pylint: disable=too-many-arguments,too-many-locals
    run_dirs: list[Path],
    audit_dirs: list[Path],
    receipts: list[Path],
    output_dir: Path,
    required_seed_count: int = 3,
    checkpoint_evaluation_csv: Path | None = None,
) -> dict[str, Any]:
    """Generate compact P4 CSV/JSON deliverables and return the verdict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_segments = group_run_segments(run_dirs)
    evaluations = (
        checkpoint_evaluation_rows(checkpoint_evaluation_csv)
        if checkpoint_evaluation_csv is not None
        else evaluation_rows(run_dirs)
    )
    evaluation_keys = [(int(row['seed']), int(row['milestone'])) for row in evaluations]
    if len(evaluation_keys) != len(set(evaluation_keys)):
        raise ValueError('evaluation table contains duplicate seed/milestone rows')
    run_seeds = sorted(run_segments)
    if {int(row['seed']) for row in evaluations} != set(run_seeds):
        raise ValueError('evaluation seeds do not match the run directories')
    audit_payloads = [read_json(path / 'replay_risk_summary.json') for path in audit_dirs]
    audits = {int(payload['seed']): payload for payload in audit_payloads}
    if len(audits) != len(audit_payloads):
        raise ValueError('audit directories contain duplicate seeds')
    receipt_payloads: dict[int, list[dict[str, Any]]] = {}
    for payload in (read_json(path) for path in receipts):
        receipt_payloads.setdefault(int(payload['seed']), []).append(payload)
    if set(audits) != set(run_seeds) or set(receipt_payloads) != set(run_seeds):
        raise ValueError('audit or receipt seeds do not match the run directories')
    for seed in run_seeds:
        if len(receipt_payloads[seed]) != len(run_segments[seed]):
            raise ValueError(f'receipt count does not match run segments for seed {seed}')
        receipt_steps = sorted(
            int(payload['steps']) for payload in receipt_payloads[seed] if 'steps' in payload
        )
        if receipt_steps and len(receipt_steps) == len(run_segments[seed]):
            segment_steps = sorted(
                int(read_json(path / 'artifacts/training_summary.json')['env_steps'])
                for path in run_segments[seed]
            )
            if receipt_steps != segment_steps:
                raise ValueError(f'receipt endpoints do not match run segments for seed {seed}')

    atomic_csv(output_dir / 'pointgoal_evaluation_runs.csv', evaluations)
    atomic_csv(
        output_dir / 'pointgoal_evaluation_summary.csv',
        aggregate_evaluations(evaluations),
    )
    mechanism_rows: list[dict[str, Any]] = []
    criteria_by_seed: dict[str, dict[str, bool]] = {}
    for seed in run_seeds:
        fixed_evaluations = (
            [row for row in evaluations if int(row['seed']) == seed]
            if checkpoint_evaluation_csv is not None
            else None
        )
        row, criteria = mechanism_row(
            run_segments[seed],
            audits[seed],
            receipt_payloads[seed],
            fixed_evaluations,
        )
        mechanism_rows.append(row)
        criteria_by_seed[str(seed)] = criteria
    mechanism_rows.sort(key=lambda row: row['seed'])
    atomic_csv(output_dir / 'pointgoal_mechanism_summary.csv', mechanism_rows)

    seed_pass = {seed: all(criteria.values()) for seed, criteria in criteria_by_seed.items()}
    overall_pass = len(seed_pass) == required_seed_count and all(seed_pass.values())
    verdict = {
        'status': 'passed' if overall_pass else 'failed',
        'required_seed_count': required_seed_count,
        'observed_seed_count': len(seed_pass),
        'seed_pass': seed_pass,
        'criteria_by_seed': criteria_by_seed,
        'total_training_seconds': sum(row['training_seconds'] for row in mechanism_rows),
        'total_estimated_cost_cny': sum(row['estimated_cost_cny'] for row in mechanism_rows),
    }
    atomic_json(output_dir / 'pointgoal_p4_verdict.json', verdict)
    return verdict


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run-dirs', nargs='+', type=Path, required=True)
    result.add_argument('--audit-dirs', nargs='+', type=Path, required=True)
    result.add_argument('--receipts', nargs='+', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--required-seed-count', type=int, default=3)
    result.add_argument('--checkpoint-evaluation-csv', type=Path)
    return result


def main() -> None:
    """Run the P4 aggregation CLI."""
    args = parser().parse_args()
    verdict = summarize(
        args.run_dirs,
        args.audit_dirs,
        args.receipts,
        args.output_dir,
        args.required_seed_count,
        args.checkpoint_evaluation_csv,
    )
    print(json.dumps(verdict, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
