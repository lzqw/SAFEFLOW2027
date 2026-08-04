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
    'spearman',
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


def _changed(values: list[float], tolerance: float = 1e-8) -> bool:
    """Return whether a finite series has a non-degenerate range."""
    return bool(values) and max(values) - min(values) > tolerance


def mechanism_row(  # pylint: disable=too-many-locals
    run_dir: Path,
    audit: dict[str, Any],
    receipt: dict[str, Any],
    fixed_evaluations: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Collect one seed's mechanism evidence and evaluate its P4 conditions."""
    training = read_json(run_dir / 'artifacts/training_summary.json')
    warmup = read_json(run_dir / 'artifacts/warmup_status.json')
    counterfactual = final_counterfactual(run_dir)
    diagnostics = read_csv(run_dir / 'artifacts/diagnostics.csv')
    evaluation_source = (
        fixed_evaluations
        if fixed_evaluations is not None
        else read_csv(run_dir / 'artifacts/evaluation.csv')
    )
    evaluations = sorted(evaluation_source, key=lambda row: int(row['milestone']))
    if not diagnostics or len(evaluations) < 2:
        raise ValueError(f'incomplete P4 artifacts in {run_dir}')
    seed = int(training['seed'])
    if int(audit['seed']) != seed or int(receipt['seed']) != seed:
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
    actor_losses = [number(row['Loss/Actor']) for row in diagnostics]
    cost_losses = [number(row['Loss/CostCritic']) for row in diagnostics]
    reward_losses = [number(row['Loss/RewardCritic']) for row in diagnostics]
    energy_duals = [number(row['Dual/Energy']) for row in diagnostics]
    global_duals = [number(row['Dual/GlobalSafety']) for row in diagnostics]

    row = {
        'seed': seed,
        'status': training['status'],
        'env_steps': int(training['env_steps']),
        'updates': int(training['updates']),
        'training_seconds': number(training['training_seconds']),
        'estimated_cost_cny': number(receipt['estimated_cost_cny']),
        'warmup_stage': warmup['stage'],
        'geometry_activation_step': int(warmup['geometry_activation_step']),
        'boundary_samples': int(warmup['boundary_samples']),
        'calibration_ema': number(warmup['ema']['calibration_error']),
        'cost_td_ema': number(warmup['ema']['cost_td_error']),
        'valid_gradient_ema': number(warmup['ema']['valid_gradient_rate']),
        'ensemble_disagreement_ema': number(warmup['ema']['ensemble_disagreement']),
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
        'replay_cost_spearman': number(
            audit['predicted_cost_empirical_cost_spearman'],
        ),
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
    }
    criteria = {
        'completed_without_crash': training['status'] == 'completed'
        and int(training['env_steps']) == 300_000,
        'return_improved_without_collapse': row['final_eval_return'] > row['initial_eval_return'],
        'evaluation_cost_or_tail_improved': row['final_eval_cost'] < row['initial_eval_cost']
        or row['final_eval_cvar'] < row['initial_eval_cvar'],
        'predicted_empirical_risk_positive': row['replay_cost_spearman'] > 0.0,
        'near_boundary_gate_dominates': boundary_gate > interior_gate,
        'normal_variance_reduced': row['directional_vn'] < 0.9 * row['isotropic_vn'],
        'tangent_variance_preserved': row['vt_preservation_ratio'] >= 0.8,
        'dangerous_normal_velocity_reduced': final_danger < w2_danger,
        'actor_critic_duals_nondegenerate': any(abs(value) > 1e-8 for value in actor_losses)
        and any(value > 0.0 for value in cost_losses)
        and any(value > 0.0 for value in reward_losses)
        and _changed(energy_duals)
        and _changed(global_duals),
        'boundary_not_mainly_random_or_violating': row['boundary_source_random_fraction'] < 0.5
        and row['boundary_immediate_violation_fraction'] < 0.5,
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
    if len(run_dirs) != len(audit_dirs) or len(run_dirs) != len(receipts):
        raise ValueError('run, audit, and receipt counts must match')
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = (
        checkpoint_evaluation_rows(checkpoint_evaluation_csv)
        if checkpoint_evaluation_csv is not None
        else evaluation_rows(run_dirs)
    )
    evaluation_keys = [(int(row['seed']), int(row['milestone'])) for row in evaluations]
    if len(evaluation_keys) != len(set(evaluation_keys)):
        raise ValueError('evaluation table contains duplicate seed/milestone rows')
    run_seeds = [
        int(read_json(run_dir / 'artifacts/training_summary.json')['seed']) for run_dir in run_dirs
    ]
    if len(run_seeds) != len(set(run_seeds)):
        raise ValueError('run directories contain duplicate training seeds')
    if {int(row['seed']) for row in evaluations} != set(run_seeds):
        raise ValueError('evaluation seeds do not match the run directories')
    audits = {
        int(payload['seed']): payload
        for payload in (read_json(path / 'replay_risk_summary.json') for path in audit_dirs)
    }
    receipt_payloads = {
        int(payload['seed']): payload for payload in (read_json(path) for path in receipts)
    }
    if set(audits) != set(run_seeds) or set(receipt_payloads) != set(run_seeds):
        raise ValueError('audit or receipt seeds do not match the run directories')

    atomic_csv(output_dir / 'pointgoal_evaluation_runs.csv', evaluations)
    atomic_csv(
        output_dir / 'pointgoal_evaluation_summary.csv',
        aggregate_evaluations(evaluations),
    )
    mechanism_rows: list[dict[str, Any]] = []
    criteria_by_seed: dict[str, dict[str, bool]] = {}
    for run_dir, seed in zip(run_dirs, run_seeds):
        fixed_evaluations = (
            [row for row in evaluations if int(row['seed']) == seed]
            if checkpoint_evaluation_csv is not None
            else None
        )
        row, criteria = mechanism_row(
            run_dir,
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
