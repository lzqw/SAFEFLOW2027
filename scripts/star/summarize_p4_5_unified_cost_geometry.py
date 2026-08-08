#!/usr/bin/env python3
"""Validate, pool, classify, and report the six preregistered P4.5 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 'p4.5-unified-cost-geometry-v1'
PROTOCOL_SHA256 = '8abebc101021ec71a00fdf6d2c5580b5f97dada710e58de00936a75490e85c11'
ARMS = ('C0', 'C1')
SEEDS = (2027, 2028, 2029)
TARGET_STEPS = 100_000
EVAL_STEPS = (0, 25_000, 50_000, 75_000, 100_000)
CHECKPOINT_STEPS = (0, 10_000, 25_000, 50_000, 75_000, 100_000)
EVAL_SEEDS = tuple(range(12_027, 12_047))
TRAJECTORY_SEEDS = (12_027, 12_036, 12_046)
SAFETY_FIELDS = ('mean_cost', 'cost_cvar_20', 'dangerous_normal_velocity')


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} does not contain a JSON object')
    return payload


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Replace undefined floats recursively for strict JSON output."""
    if isinstance(value, dict):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_text(path: Path, text: str) -> None:
    """Atomically replace one UTF-8 text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict sorted JSON atomically."""
    atomic_text(
        path,
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + '\n',
    )


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a consistently headed CSV atomically."""
    if not rows:
        raise ValueError(f'cannot write empty CSV {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def finite_number(value: Any, location: str) -> float:
    """Return a finite scalar or reject malformed primary evidence."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{location} is not numeric: {value!r}') from error
    if not math.isfinite(number):
        raise ValueError(f'{location} is not finite: {value!r}')
    return number


def _optional_finite_number(value: Any, location: str) -> float | None:
    """Preserve an explicitly undefined diagnostic or validate its finite value."""
    if value is None:
        return None
    return finite_number(value, location)


def _bindings(values: list[str], label: str) -> dict[tuple[str, int], Path]:
    """Parse exactly six ``ARM:SEED=PATH`` bindings."""
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    result: dict[tuple[str, int], Path] = {}
    for value in values:
        identity, separator, raw_path = value.partition('=')
        arm, colon, raw_seed = identity.partition(':')
        if separator != '=' or colon != ':' or not raw_path:
            raise ValueError(f'{label} bindings must use ARM:SEED=PATH')
        key = (arm, int(raw_seed))
        if key not in expected or key in result:
            raise ValueError(f'{label} has unexpected or duplicate identity {key}')
        result[key] = Path(raw_path)
    if set(result) != expected:
        raise ValueError(f'{label} must bind C0/C1 at seeds {SEEDS}')
    return result


def _resolve_run_dir(container: Path) -> Path:
    """Resolve a binding to its unique STAR timestamped run directory."""
    if (container / 'artifacts/training_summary.json').is_file():
        return container
    matches = list(container.glob('**/artifacts/training_summary.json'))
    if len(matches) != 1:
        raise ValueError(f'{container} contains {len(matches)} training summaries, expected one')
    return matches[0].parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV into row mappings."""
    with path.open('r', encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def _tail_mean(values: list[float], fraction: float = 0.2) -> float:
    """Return the upper empirical tail mean."""
    count = max(1, math.ceil(len(values) * fraction))
    return statistics.fmean(sorted(values, reverse=True)[:count])


def _checkpoint_metrics(payload: dict[str, Any], arm: str, seed: int) -> dict[str, Any]:
    """Flatten one built-in formal evaluation artifact."""
    geometry = payload.get('direction_geometry', {})
    required_geometry = (
        'normal_variance',
        'tangent_variance',
        'normal_to_tangent_variance_ratio',
        'dangerous_normal_velocity',
        'outward_velocity_ratio',
        'tangent_energy',
        'normal_energy',
        'crossing_rate',
        'action_clipping_rate',
        'safety_band_sample_ratio',
        'action_gradient_norm',
    )
    row: dict[str, Any] = {
        'arm': arm,
        'training_seed': seed,
        'milestone': int(payload['milestone']),
        'episodes': int(payload['episodes']),
        'mean_return': finite_number(payload['mean_return'], 'mean_return'),
        'success_rate': finite_number(payload['success_rate'], 'success_rate'),
        'mean_cost': finite_number(payload['mean_cost'], 'mean_cost'),
        'cost_cvar_20': finite_number(payload['cost_cvar_20'], 'cost_cvar_20'),
        'violation_ratio': finite_number(payload['violation_ratio'], 'violation_ratio'),
        'training_cumulative_cost': finite_number(
            payload['training_cumulative_cost'],
            'training_cumulative_cost',
        ),
        'training_violation_ratio': finite_number(
            payload['training_violation_ratio'],
            'training_violation_ratio',
        ),
        'stall_ratio': finite_number(payload['stall_ratio'], 'stall_ratio'),
        'mean_displacement': finite_number(payload['mean_displacement'], 'mean_displacement'),
        'mean_path_length': finite_number(payload['mean_path_length'], 'mean_path_length'),
        'mean_action_norm': finite_number(payload['mean_action_norm'], 'mean_action_norm'),
        'cost_per_successful_episode': payload.get('cost_per_successful_episode'),
        'cost_per_unit_path_length': payload.get('cost_per_unit_path_length'),
        'cost_to_go_mae': finite_number(payload['cost_to_go_mae'], 'cost_to_go_mae'),
        'cost_spearman': _optional_finite_number(payload['spearman'], 'spearman'),
        'probability_auroc': payload.get('auroc'),
        'probability_brier_score': payload.get('brier_score'),
        'probability_brier_skill': payload.get('brier_skill'),
        'probability_calibration_error': payload.get('calibration_error'),
    }
    for key in required_geometry:
        row[key] = finite_number(geometry[key], f'direction_geometry.{key}')
    return row


def _validate_audit(path: Path, arm: str, seed: int) -> dict[str, Any]:
    """Validate one final fixed-size three-mode audit."""
    payload = read_json(path)
    if (
        payload.get('stage') != 'P4.5-final-cost-geometry-audit'
        or payload.get('protocol_version') != PROTOCOL_VERSION
        or payload.get('protocol_sha256') != PROTOCOL_SHA256
    ):
        raise ValueError(f'{arm} seed {seed} audit has the wrong protocol seal')
    if payload.get('arm') != arm or int(payload.get('training_seed', -1)) != seed:
        raise ValueError(f'{arm} seed {seed} audit identity mismatch')
    if (
        int(payload.get('env_steps', -1)) != TARGET_STEPS
        or int(payload.get('state_count', -1)) != 64
        or int(payload.get('action_samples_per_state', -1)) != 256
        or int(payload.get('audit_seed', -1)) != 4_500_007
    ):
        raise ValueError(f'{arm} seed {seed} audit changed its locked sample')
    if int(payload.get('valid_fixed_normal_state_count', 0)) <= 0:
        raise ValueError(f'{arm} seed {seed} audit has no valid Q-cost normals')
    expected_modes = {
        'A': (False, True, True),
        'B': (True, False, True),
        'C': (True, True, True),
    }
    modes: dict[str, Any] = {}
    for mode, expected in expected_modes.items():
        section = payload['modes'][mode]
        spec = section['specification']
        actual = (
            bool(spec['learned_drift']),
            bool(spec['wiener_noise']),
            bool(spec['action_clipping']),
        )
        if actual != expected:
            raise ValueError(f'{arm} seed {seed} mode {mode} flags mismatch')
        modes[mode] = {
            key: finite_number(value, f'{arm}.{seed}.audit.{mode}.{key}')
            for key, value in section['metrics'].items()
        }
    return {
        'path': str(path),
        'checkpoint': str(payload['checkpoint']),
        'checkpoint_sha256': str(payload['checkpoint_sha256']),
        'valid_fixed_normal_state_count': int(payload['valid_fixed_normal_state_count']),
        'selection': payload['selection'],
        'modes': modes,
    }


def _validate_run(
    arm: str,
    seed: int,
    container: Path,
    audit_path: Path,
) -> dict[str, Any]:
    """Validate one clean-source 100k run and collect its raw checkpoint evidence."""
    run_dir = _resolve_run_dir(container)
    summary = read_json(run_dir / 'artifacts/training_summary.json')
    manifest = read_json(run_dir / 'artifacts/run_manifest.json')
    receipt_path = container / 'receipt.json'
    if not receipt_path.is_file():
        candidates = list(container.glob('**/receipt.json'))
        if len(candidates) != 1:
            raise ValueError(f'{container} must contain one receipt.json')
        receipt_path = candidates[0]
    receipt = read_json(receipt_path)
    if summary.get('status') != 'completed' or receipt.get('status') != 'passed':
        raise ValueError(f'{arm} seed {seed} did not complete normally')
    if any(int(item.get('seed', -1)) != seed for item in (summary, manifest, receipt)):
        raise ValueError(f'{arm} seed {seed} identity mismatch')
    if int(summary['env_steps']) != TARGET_STEPS or int(receipt['steps']) != TARGET_STEPS:
        raise ValueError(f'{arm} seed {seed} did not finish exactly 100k steps')
    if receipt.get('source_dirty') is not False or not receipt.get('source_commit'):
        raise ValueError(f'{arm} seed {seed} was not trained from clean committed source')
    if bool(summary['directional_loss_enabled']) != (arm == 'C1'):
        raise ValueError(f'{arm} seed {seed} has the wrong direction-loss switch')
    required_flags = {
        'unified_cost_geometry_enabled': True,
        'safety_anisotropy_enabled': True,
        'safety_inward_drift_enabled': False,
        'safety_outward_rectification_enabled': False,
        'candidate_probing_enabled': False,
        'dynamic_geometry_gate': False,
        'engineering_hard_stops_enabled': True,
        'numerical_hard_stops_enabled': False,
    }
    for key, expected in required_flags.items():
        if summary.get(key) != expected:
            raise ValueError(f'{arm} seed {seed} changed method flag {key}')
    p45 = manifest['p4_5_unified_cost_geometry']
    if (
        p45['protocol_version'] != PROTOCOL_VERSION
        or p45['protocol_sha256'] != PROTOCOL_SHA256
        or bool(p45['probability_critic_policy_access'])
    ):
        raise ValueError(f'{arm} seed {seed} manifest violates the P4.5 seal')
    if tuple(int(value) for value in manifest['eval_milestones']) != EVAL_STEPS:
        raise ValueError(f'{arm} seed {seed} has wrong evaluation milestones')
    if tuple(int(value) for value in manifest['checkpoint_milestones']) != CHECKPOINT_STEPS:
        raise ValueError(f'{arm} seed {seed} has wrong checkpoint milestones')

    diagnostics = _read_csv(run_dir / 'artifacts/diagnostics.csv')
    diagnostic_steps = tuple(int(float(row['TotalEnvSteps'])) for row in diagnostics)
    if diagnostic_steps != tuple(range(5_000, TARGET_STEPS + 1, 5_000)):
        raise ValueError(f'{arm} seed {seed} lacks exact 5k diagnostic coverage')
    evaluations: dict[int, dict[str, Any]] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    for milestone in EVAL_STEPS:
        path = run_dir / f'artifacts/evaluation_step_{milestone}.json'
        payload = read_json(path)
        if tuple(int(value) for value in payload['seeds']) != EVAL_SEEDS:
            raise ValueError(f'{arm} seed {seed} step {milestone} used wrong eval seeds')
        if int(payload['episodes']) != 20 or int(payload['milestone']) != milestone:
            raise ValueError(f'{arm} seed {seed} step {milestone} evaluation mismatch')
        if tuple(int(value) for value in payload['trajectory_seeds']) != TRAJECTORY_SEEDS:
            raise ValueError(f'{arm} seed {seed} changed trajectory seeds')
        if len(payload.get('cost_risk_bins', [])) != 5:
            raise ValueError(f'{arm} seed {seed} step {milestone} lacks five risk bins')
        evaluations[milestone] = payload
        checkpoint_rows.append(_checkpoint_metrics(payload, arm, seed))
    trajectory = run_dir / f'artifacts/trajectory_step_{TARGET_STEPS}.csv'
    trajectory_rows = _read_csv(trajectory)
    if {int(row['seed']) for row in trajectory_rows} != set(TRAJECTORY_SEEDS):
        raise ValueError(f'{arm} seed {seed} final trajectory seed coverage mismatch')

    checkpoint_hashes = []
    for milestone in CHECKPOINT_STEPS:
        path = run_dir / f'checkpoints/milestone_{milestone}.pt'
        if not path.is_file():
            raise FileNotFoundError(f'{arm} seed {seed} lacks {path.name}')
        checkpoint_hashes.append(
            {
                'arm': arm,
                'training_seed': seed,
                'kind': 'lightweight',
                'milestone': milestone,
                'path': str(path),
                'bytes': path.stat().st_size,
                'sha256': sha256(path),
            },
        )
    final_full = run_dir / 'checkpoints/final_full.pt'
    if not final_full.is_file():
        raise FileNotFoundError(f'{arm} seed {seed} lacks final_full.pt')
    checkpoint_hashes.append(
        {
            'arm': arm,
            'training_seed': seed,
            'kind': 'full_replay',
            'milestone': TARGET_STEPS,
            'path': str(final_full),
            'bytes': final_full.stat().st_size,
            'sha256': sha256(final_full),
        },
    )
    audit = _validate_audit(audit_path, arm, seed)
    if audit['checkpoint_sha256'] != checkpoint_hashes[-1]['sha256']:
        raise ValueError(f'{arm} seed {seed} audit checkpoint identity mismatch')
    return {
        'arm': arm,
        'training_seed': seed,
        'container': str(container),
        'run_dir': str(run_dir),
        'receipt': str(receipt_path),
        'source_commit': receipt['source_commit'],
        'summary': summary,
        'evaluations': evaluations,
        'checkpoint_rows': checkpoint_rows,
        'diagnostics': diagnostics,
        'trajectory': str(trajectory),
        'checkpoint_hashes': checkpoint_hashes,
        'audit': audit,
    }


def _pooled_final(runs: dict[tuple[str, int], dict[str, Any]], arm: str) -> dict[str, float]:
    """Pool the 60 preregistered final episodes and three seed-level geometry means."""
    payloads = [runs[(arm, seed)]['evaluations'][TARGET_STEPS] for seed in SEEDS]
    returns = [float(value) for payload in payloads for value in payload['returns']]
    costs = [float(value) for payload in payloads for value in payload['costs']]
    goal_counts = [int(value) for payload in payloads for value in payload['goal_counts']]
    return {
        'episodes': float(len(returns)),
        'mean_return': statistics.fmean(returns),
        'success_rate': sum(value > 0 for value in goal_counts) / len(goal_counts),
        'mean_cost': statistics.fmean(costs),
        'cost_cvar_20': _tail_mean(costs),
        'dangerous_normal_velocity': statistics.fmean(
            float(payload['direction_geometry']['dangerous_normal_velocity'])
            for payload in payloads
        ),
        'mean_path_length': statistics.fmean(
            float(payload['mean_path_length']) for payload in payloads
        ),
        'stall_ratio': statistics.fmean(float(payload['stall_ratio']) for payload in payloads),
    }


def _classification(runs: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    """Apply the locked three-label final interpretation without extra gates."""
    final_rows = {
        (arm, seed): runs[(arm, seed)]['checkpoint_rows'][-1] for arm in ARMS for seed in SEEDS
    }
    pooled = {arm: _pooled_final(runs, arm) for arm in ARMS}
    safety_conditions: dict[str, Any] = {}
    for field in SAFETY_FIELDS:
        lower_seeds = [
            seed
            for seed in SEEDS
            if final_rows[('C1', seed)][field] < final_rows[('C0', seed)][field]
        ]
        pooled_lower = pooled['C1'][field] < pooled['C0'][field]
        safety_conditions[field] = {
            'c1_lower_seeds': lower_seeds,
            'paired_seed_count': len(lower_seeds),
            'pooled_c0': pooled['C0'][field],
            'pooled_c1': pooled['C1'][field],
            'pooled_c1_minus_c0': pooled['C1'][field] - pooled['C0'][field],
            'condition_met': len(lower_seeds) >= 2 and pooled_lower,
        }
    safety_majority = all(item['condition_met'] for item in safety_conditions.values())
    return_loss = pooled['C0']['mean_return'] - pooled['C1']['mean_return']
    success_loss = pooled['C0']['success_rate'] - pooled['C1']['success_rate']
    performance_maintained = return_loss <= 0.10 and success_loss <= 0.10
    if safety_majority and performance_maintained:
        label = 'effective'
    elif safety_majority:
        label = 'safety_benefit_with_performance_cost'
    else:
        label = 'no_stable_effect'
    return {
        'label': label,
        'safety_majority': safety_majority,
        'safety_conditions': safety_conditions,
        'performance_maintained': performance_maintained,
        'pooled_return_loss_c0_minus_c1': return_loss,
        'pooled_success_loss_c0_minus_c1': success_loss,
        'performance_loss_limit': 0.10,
        'pooled': pooled,
    }


def _render_report(result: dict[str, Any]) -> str:
    """Render the full six-run evidence and locked three-class conclusion."""
    classification = result['classification']
    lines = [
        '# STAR P4.5 unified cost geometry final report',
        '',
        f"Conclusion: **{classification['label']}**",
        '',
        f'Protocol: `{PROTOCOL_VERSION}` (`{PROTOCOL_SHA256}`)',
        '',
        (
            'All six C0/C1 x seed runs were required to finish 100k unless an engineering '
            'failure occurred. The table below reports every final run; pooled values never '
            'replace the per-seed evidence.'
        ),
        '',
        '| arm | seed | return | success | mean cost | CVaR20 | dangerous v_n+ | V_N/V_T | path | stall |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in result['final_seed_metrics']:
        lines.append(
            f"| {row['arm']} | {row['training_seed']} | {row['mean_return']:.6f} | "
            f"{row['success_rate']:.3f} | {row['mean_cost']:.6f} | "
            f"{row['cost_cvar_20']:.6f} | {row['dangerous_normal_velocity']:.6f} | "
            f"{row['normal_to_tangent_variance_ratio']:.6f} | "
            f"{row['mean_path_length']:.6f} | {row['stall_ratio']:.3f} |",
        )
    lines.extend(
        [
            '',
            '## Pooled final result',
            '',
            '| arm | episodes | return | success | mean cost | CVaR20 | dangerous v_n+ | path | stall |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ],
    )
    for arm in ARMS:
        pooled = classification['pooled'][arm]
        lines.append(
            f"| {arm} | {int(pooled['episodes'])} | {pooled['mean_return']:.6f} | "
            f"{pooled['success_rate']:.3f} | {pooled['mean_cost']:.6f} | "
            f"{pooled['cost_cvar_20']:.6f} | {pooled['dangerous_normal_velocity']:.6f} | "
            f"{pooled['mean_path_length']:.6f} | {pooled['stall_ratio']:.3f} |",
        )
    lines.extend(['', '## Locked interpretation', ''])
    for field, evidence in classification['safety_conditions'].items():
        lines.append(
            f"- `{field}`: C1 lower in {evidence['paired_seed_count']}/3 paired seeds "
            f"({evidence['c1_lower_seeds']}); pooled delta "
            f"{evidence['pooled_c1_minus_c0']:+.6f}; condition "
            f"{'met' if evidence['condition_met'] else 'not met'}.",
        )
    lines.extend(
        [
            f"- Safety-majority condition: {classification['safety_majority']}.",
            (
                '- Pooled C0-C1 return loss: '
                f"{classification['pooled_return_loss_c0_minus_c1']:+.6f}; "
                'allowed maximum 0.10.'
            ),
            (
                '- Pooled C0-C1 success loss: '
                f"{classification['pooled_success_loss_c0_minus_c1']:+.6f}; "
                'allowed maximum 0.10.'
            ),
            f"- Performance maintained: {classification['performance_maintained']}.",
            '',
            '## Audit and critic evidence',
            '',
            (
                'Each final full-replay checkpoint received the sealed 64-state x 256-action '
                'A/B/C common-randomness audit. Cost-critic TD, Q scale, action gradients, '
                'cost-to-go MAE/Spearman, and five risk bins are retained in the raw CSV/JSON '
                'evidence. Probability AUROC, Brier, skill, and calibration remain diagnostic-only.'
            ),
            '',
            '## Figures',
            '',
            '- `p4_5_return_success.png`',
            '- `p4_5_cost_cvar_training_cost.png`',
            '- `p4_5_geometry.png`',
            '- `p4_5_critic_scales.png`',
            '- `p4_5_trajectory_eval_seed_12027.png`',
            '- `p4_5_trajectory_eval_seed_12036.png`',
            '- `p4_5_trajectory_eval_seed_12046.png`',
            '',
            (
                'Large checkpoints and replay stay on the server. '
                '`p4_5_checkpoint_sha256.csv` records their immutable identities; the GitHub '
                'evidence bundle contains compact CSV/JSON, plots, and this report.'
            ),
            '',
        ],
    )
    return '\n'.join(lines)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the entire matrix, write compact evidence, and return the final classification."""
    if sha256(args.protocol) != PROTOCOL_SHA256:
        raise ValueError('P4.5 protocol hash does not match its preregistered seal')
    protocol = read_json(args.protocol)
    if protocol.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('P4.5 protocol version mismatch')
    run_paths = _bindings(args.runs, 'run')
    audit_paths = _bindings(args.audits, 'audit')
    runs = {key: _validate_run(*key, run_paths[key], audit_paths[key]) for key in sorted(run_paths)}
    source_commits = {run['source_commit'] for run in runs.values()}
    if len(source_commits) != 1:
        raise ValueError(f'P4.5 runs used multiple source commits: {sorted(source_commits)}')

    checkpoint_metrics = [row for key in sorted(runs) for row in runs[key]['checkpoint_rows']]
    final_seed_metrics = [
        row for row in checkpoint_metrics if int(row['milestone']) == TARGET_STEPS
    ]
    paired_deltas = []
    for seed in SEEDS:
        c0 = next(
            row for row in final_seed_metrics if row['arm'] == 'C0' and row['training_seed'] == seed
        )
        c1 = next(
            row for row in final_seed_metrics if row['arm'] == 'C1' and row['training_seed'] == seed
        )
        paired_deltas.append(
            {
                'training_seed': seed,
                **{
                    f'c1_minus_c0_{field}': float(c1[field]) - float(c0[field])
                    for field in (
                        'mean_return',
                        'success_rate',
                        'mean_cost',
                        'cost_cvar_20',
                        'dangerous_normal_velocity',
                        'mean_path_length',
                        'stall_ratio',
                    )
                },
            },
        )
    diagnostic_rows = []
    trajectory_rows = []
    checkpoint_hash_rows = []
    audits: dict[str, Any] = {}
    for (arm, seed), run in sorted(runs.items()):
        diagnostic_rows.extend(
            {'arm': arm, 'training_seed': seed, **row} for row in run['diagnostics']
        )
        trajectory_rows.append(
            {
                'arm': arm,
                'training_seed': seed,
                'milestone': TARGET_STEPS,
                'path': run['trajectory'],
            },
        )
        checkpoint_hash_rows.extend(run['checkpoint_hashes'])
        audits[f'{arm}:{seed}'] = run['audit']

    classification = _classification(runs)
    result = {
        'stage': 'P4.5-final-unified-cost-geometry-summary',
        'protocol_version': PROTOCOL_VERSION,
        'protocol_sha256': PROTOCOL_SHA256,
        'source_commit': next(iter(source_commits)),
        'run_count': len(runs),
        'all_runs_completed_100k': True,
        'scientific_early_stops': 0,
        'final_seed_metrics': final_seed_metrics,
        'paired_deltas': paired_deltas,
        'classification': classification,
        'audits': audits,
        'run_paths': {f'{arm}:{seed}': run['run_dir'] for (arm, seed), run in runs.items()},
    }
    output = args.output_dir
    atomic_csv(output / 'p4_5_checkpoint_metrics.csv', checkpoint_metrics)
    atomic_csv(output / 'p4_5_final_seed_metrics.csv', final_seed_metrics)
    atomic_csv(output / 'p4_5_paired_seed_deltas.csv', paired_deltas)
    atomic_csv(output / 'p4_5_diagnostics_combined.csv', diagnostic_rows)
    atomic_csv(output / 'p4_5_checkpoint_sha256.csv', checkpoint_hash_rows)
    atomic_csv(output / 'p4_5_trajectory_manifest.csv', trajectory_rows)
    atomic_json(output / 'p4_5_final_summary.json', result)
    atomic_text(output / 'P4_5_UNIFIED_COST_GEOMETRY_REPORT.md', _render_report(result))
    return result


def parse_args() -> argparse.Namespace:
    """Parse explicit six-run and six-audit bindings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--audits', nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Validate and print the locked final classification."""
    args = parse_args()
    result = summarize(args)
    print(
        json.dumps(
            {
                'classification': result['classification']['label'],
                'output_dir': str(args.output_dir),
            },
        ),
    )


if __name__ == '__main__':
    main()
