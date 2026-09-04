#!/usr/bin/env python3
"""Build the preregistered four-method PointCircle 500k comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

ENVIRONMENT = 'SafetyPointCircle1-v0'
TRAINING_SEEDS = (2027, 2028, 2029)
BASE_SEEDS = (22_027, 22_028, 22_029, 22_030, 22_031)
EPISODES_PER_BASE_SEED = 20
EPISODES_PER_TRAINING_SEED = 100
TRAINING_STEPS = 500_000
METHOD_ORDER = ('C0 / isotropic FLAC', 'SACLag', 'SHAPO-PIDLag', 'P4.7 / Ours')
ARM_METHODS = {'C0': METHOD_ORDER[0], 'P47': METHOD_ORDER[3]}
PRIOR_METHODS = ('SACLag', 'SHAPO-PIDLag')
DEFAULT_ARM_CONFIGS = {
    'C0': Path('configs/star/single_gpu_pointcircle_c0.yaml'),
    'P47': Path('configs/star/single_gpu_pointcircle_p47.yaml'),
}


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one non-empty homogeneous CSV through an atomic rename."""
    if not rows:
        raise ValueError('cannot write an empty comparison table')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    """Write UTF-8 text through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)


def tail_mean(values: list[float], fraction: float = 0.2) -> float:
    """Return empirical upper-tail cost mean."""
    count = max(1, math.ceil(len(values) * fraction))
    return statistics.fmean(sorted(values, reverse=True)[:count])


def sample_std(values: list[float]) -> float:
    """Return sample standard deviation, defined as zero for one value."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def finite(value: Any, label: str) -> float:
    """Require one finite numeric value."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{label} is not finite')
    return number


def require_close(observed: Any, expected: float, label: str) -> None:
    """Require a stored summary to agree with raw episodes."""
    if not math.isclose(float(observed), expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f'{label}: stored {observed} disagrees with raw {expected}')


def expected_schedule() -> list[tuple[int, int]]:
    """Return the frozen 5x20 sequential episode schedule."""
    return [
        (seed, episode_index)
        for seed in BASE_SEEDS
        for episode_index in range(EPISODES_PER_BASE_SEED)
    ]


def raw_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    """Recompute main metrics from raw episode rows."""
    if len(records) != EPISODES_PER_TRAINING_SEED:
        raise ValueError('one training seed must contribute exactly 100 episodes')
    returns = [float(row['return']) for row in records]
    costs = [float(row['cost']) for row in records]
    return {
        'mean_return': statistics.fmean(returns),
        'mean_cost': statistics.fmean(costs),
        'cost_cvar_20': tail_mean(costs),
    }


def validate_schedule(payload: dict[str, Any], label: str, episodes: Any) -> None:
    """Require the frozen PointCircle final schedule."""
    if payload.get('environment') != ENVIRONMENT:
        raise ValueError(f'{label}: wrong environment')
    if [int(seed) for seed in payload.get('base_seeds', [])] != list(BASE_SEEDS):
        raise ValueError(f'{label}: wrong base seeds')
    if int(payload.get('episodes_per_base_seed', -1)) != EPISODES_PER_BASE_SEED:
        raise ValueError(f'{label}: wrong episodes per base seed')
    if int(episodes) != EPISODES_PER_TRAINING_SEED:
        raise ValueError(f'{label}: wrong total episode count')


def load_prior(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one prior-baseline evaluation from raw episode evidence."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    method = str(payload.get('algorithm'))
    seed = int(payload.get('training_seed', -1))
    label = f'{method}:{seed}'
    if payload.get('status') != 'complete' or method not in PRIOR_METHODS:
        raise ValueError(f'{label}: not a completed prior evaluation')
    if payload.get('evaluation_role') != 'primary' or int(payload.get('training_steps', -1)) != TRAINING_STEPS:
        raise ValueError(f'{label}: wrong evaluation role or training budget')
    if seed not in TRAINING_SEEDS:
        raise ValueError(f'{label}: unexpected training seed')
    validate_schedule(payload, label, payload.get('summary', {}).get('episodes'))
    records = [
        {
            'method': method,
            'training_seed': seed,
            'base_seed': int(record['base_seed']),
            'episode_index': int(record['episode_index']),
            'return': float(record['return']),
            'cost': float(record['cost']),
        }
        for record in payload.get('episode_records', [])
    ]
    if [(row['base_seed'], row['episode_index']) for row in records] != expected_schedule():
        raise ValueError(f'{label}: raw episode schedule is incomplete or reordered')
    metrics = raw_summary(records)
    for key, value in metrics.items():
        require_close(payload['summary'][key], value, f'{label}.{key}')
    row = {
        'method': method,
        'training_seed': seed,
        'training_steps': TRAINING_STEPS,
        'episodes': EPISODES_PER_TRAINING_SEED,
        **metrics,
        'training_cumulative_cost': finite(
            payload['training_cumulative_cost'],
            f'{label}.training_cumulative_cost',
        ),
        'high_risk_vn_vt': None,
        'high_risk_dangerous_outward_velocity': None,
        'evaluation_mode': payload['evaluation_mode'],
        'evidence_file': str(path.resolve()),
        'evidence_sha256': sha256_file(path),
    }
    return row, records


def load_star(
    arm: str,
    seed: int,
    evaluation_path: Path,
    receipt_path: Path,
    config_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one C0/P4.7 final evaluation and its training receipt."""
    payload = json.loads(evaluation_path.read_text(encoding='utf-8'))
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    label = f'{arm}:{seed}'
    valid_receipt = (
        receipt.get('status') == 'passed'
        and receipt.get('environment') == ENVIRONMENT
        and int(receipt.get('seed', -1)) == seed
        and int(receipt.get('steps', -1)) == TRAINING_STEPS
        and receipt.get('config_sha256') == config_sha256
        and receipt.get('source_dirty') is False
    )
    if not valid_receipt:
        raise ValueError(f'{label}: invalid training receipt')
    if (
        payload.get('status') != 'complete'
        or payload.get('arm') != arm
        or payload.get('evaluation_mode') != 'fixed-random-flow'
        or payload.get('evaluation_role') != 'primary'
        or int(payload.get('training_seed', -1)) != seed
        or int(payload.get('milestone', -1)) != TRAINING_STEPS
    ):
        raise ValueError(f'{label}: invalid final STAR evaluation')
    validate_schedule(payload, label, payload.get('episodes'))
    checkpoint = Path(payload['checkpoint'])
    if not checkpoint.is_file() or sha256_file(checkpoint) != payload.get('checkpoint_sha256'):
        raise ValueError(f'{label}: checkpoint identity mismatch')
    seeds = [int(value) for value in payload['seeds']]
    indices = [int(value) for value in payload['episode_indices']]
    returns = [float(value) for value in payload['returns']]
    costs = [float(value) for value in payload['costs']]
    if not all(
        len(values) == EPISODES_PER_TRAINING_SEED
        for values in (seeds, indices, returns, costs)
    ):
        raise ValueError(f'{label}: malformed raw arrays')
    records = [
        {
            'method': ARM_METHODS[arm],
            'training_seed': seed,
            'base_seed': base_seed,
            'episode_index': episode_index,
            'return': episode_return,
            'cost': episode_cost,
        }
        for base_seed, episode_index, episode_return, episode_cost in zip(
            seeds,
            indices,
            returns,
            costs,
        )
    ]
    if [(row['base_seed'], row['episode_index']) for row in records] != expected_schedule():
        raise ValueError(f'{label}: raw episode schedule is incomplete or reordered')
    metrics = raw_summary(records)
    for key, value in metrics.items():
        require_close(payload[key], value, f'{label}.{key}')
    geometry = payload.get('direction_geometry_high_risk_top_20_percent', {})
    training_summary = receipt.get('training_summary', {})
    row = {
        'method': ARM_METHODS[arm],
        'training_seed': seed,
        'training_steps': TRAINING_STEPS,
        'episodes': EPISODES_PER_TRAINING_SEED,
        **metrics,
        'training_cumulative_cost': finite(
            training_summary.get('cumulative_training_cost'),
            f'{label}.training_cumulative_cost',
        ),
        'high_risk_vn_vt': finite(
            geometry.get('normal_to_tangent_variance_ratio'),
            f'{label}.high_risk_vn_vt',
        ),
        'high_risk_dangerous_outward_velocity': finite(
            geometry.get('dangerous_outward_velocity'),
            f'{label}.high_risk_outward_velocity',
        ),
        'evaluation_mode': payload['evaluation_mode'],
        'evidence_file': str(evaluation_path.resolve()),
        'evidence_sha256': sha256_file(evaluation_path),
        'receipt_file': str(receipt_path.resolve()),
        'receipt_sha256': sha256_file(receipt_path),
    }
    return row, records


def aggregate_method(
    method: str,
    seed_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate one method while keeping seed-level variation explicit."""
    if sorted(int(row['training_seed']) for row in seed_rows) != list(TRAINING_SEEDS):
        raise ValueError(f'{method}: incomplete training-seed matrix')
    if len(records) != len(TRAINING_SEEDS) * EPISODES_PER_TRAINING_SEED:
        raise ValueError(f'{method}: incomplete raw episode matrix')
    returns = [float(row['return']) for row in records]
    costs = [float(row['cost']) for row in records]
    ratios = [float(row['high_risk_vn_vt']) for row in seed_rows if row['high_risk_vn_vt'] is not None]
    outward = [
        float(row['high_risk_dangerous_outward_velocity'])
        for row in seed_rows
        if row['high_risk_dangerous_outward_velocity'] is not None
    ]
    return {
        'method': method,
        'training_steps_per_seed': TRAINING_STEPS,
        'training_seeds': len(seed_rows),
        'pooled_evaluation_episodes': len(records),
        'mean_return': statistics.fmean(returns),
        'seed_std_mean_return': sample_std([float(row['mean_return']) for row in seed_rows]),
        'mean_cost': statistics.fmean(costs),
        'seed_std_mean_cost': sample_std([float(row['mean_cost']) for row in seed_rows]),
        'cost_cvar_20': tail_mean(costs),
        'seed_std_cost_cvar_20': sample_std([float(row['cost_cvar_20']) for row in seed_rows]),
        'training_cumulative_cost_mean': statistics.fmean(
            float(row['training_cumulative_cost']) for row in seed_rows
        ),
        'seed_std_training_cumulative_cost': sample_std(
            [float(row['training_cumulative_cost']) for row in seed_rows]
        ),
        'high_risk_vn_vt': statistics.fmean(ratios) if ratios else None,
        'high_risk_dangerous_outward_velocity': statistics.fmean(outward) if outward else None,
    }


def optional(value: Any, digits: int = 4) -> str:
    """Format an optional diagnostic."""
    return 'n/a' if value is None else f'{float(value):.{digits}f}'


def markdown_report(
    aggregates: list[dict[str, Any]],
    enhancement_preregistration_sha256: str | None = None,
) -> str:
    """Render the final PointCircle comparison without an every-seed claim."""
    rows = {row['method']: row for row in aggregates}
    ours = rows['P4.7 / Ours']
    c0 = rows['C0 / isotropic FLAC']
    lines = [
        '# PointCircle four-method final comparison',
        '',
        'All methods use three training seeds (`2027–2029`), an equal `500k` budget, and '
        '100 fixed-stream final episodes per checkpoint. The independent statistical unit is '
        'the training seed (`n=3`).',
        '',
    ]
    if enhancement_preregistration_sha256 is not None:
        lines.extend(
            [
                'The P4.7 row uses the single preregistered PointCircle geometry enhancement; '
                f'addendum SHA-256: `{enhancement_preregistration_sha256}`.',
                '',
            ],
        )
    lines.extend(
        [
            '| Method | Return | Mean cost | CVaR-20 | Train cost | High-risk V_N/V_T | Outward velocity |',
            '|---|---:|---:|---:|---:|---:|---:|',
        ],
    )
    for method in METHOD_ORDER:
        row = rows[method]
        lines.append(
            f"| {method} | {row['mean_return']:.4f} | {row['mean_cost']:.3f} | "
            f"{row['cost_cvar_20']:.3f} | {row['training_cumulative_cost_mean']:.1f} | "
            f"{optional(row['high_risk_vn_vt'])} | "
            f"{optional(row['high_risk_dangerous_outward_velocity'])} |"
        )
    lines.extend(
        [
            '',
            '## C0 versus P4.7 aggregate comparison',
            '',
            f"P4.7 changes aggregate return by `{ours['mean_return'] - c0['mean_return']:+.4f}`, "
            f"mean cost by `{ours['mean_cost'] - c0['mean_cost']:+.3f}`, and CVaR-20 by "
            f"`{ours['cost_cvar_20'] - c0['cost_cvar_20']:+.3f}` relative to isotropic FLAC.",
            '',
            'These are aggregate three-seed results. The report does not claim that every seed is '
            'a Pareto improvement. SACLag and SHAPO-PIDLag are prior-method comparisons and were '
            'not used to modify P4.7.',
            '',
        ]
    )
    return '\n'.join(lines)


def git_commit(source: Path) -> str:
    """Require clean finalizer source and return its commit."""
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError('formal finalization requires a clean source')
    return subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_screen_decision(
    decision: dict[str, Any],
    expected_protocol_sha256: str,
    expected_enhancement_sha256: str | None,
) -> None:
    """Require a passing original or explicitly authenticated enhanced screen."""
    protocol_sha = decision.get('protocol_sha256', decision.get('parent_protocol_sha256'))
    valid = (
        decision.get('status') == 'complete'
        and decision.get('environment') == ENVIRONMENT
        and bool(decision.get('decision', {}).get('screen_pass'))
        and protocol_sha == expected_protocol_sha256
    )
    if expected_enhancement_sha256 is not None:
        valid = (
            valid
            and decision.get('enhancement_preregistration_sha256')
            == expected_enhancement_sha256
            and int(decision.get('attempt', -1)) == 1
            and int(decision.get('attempt_limit', -1)) == 1
        )
    if not valid:
        raise ValueError('formal PointCircle matrix lacks an authenticated passing screen')


def parser() -> argparse.ArgumentParser:
    """Build finalization arguments."""
    root = Path(__file__).resolve().parents[2]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--star-training-root', type=Path, required=True)
    result.add_argument('--star-evaluation-root', type=Path, required=True)
    result.add_argument('--prior-evaluations', nargs='+', type=Path, required=True)
    result.add_argument('--screen-decision', type=Path, required=True)
    result.add_argument('--protocol', type=Path, default=root / 'reports/star/single_gpu_paper_core_preregistration.json')
    result.add_argument('--expected-protocol-sha256', required=True)
    result.add_argument('--p47-config', type=Path)
    result.add_argument('--expected-enhancement-preregistration-sha256')
    result.add_argument('--output-dir', type=Path, required=True)
    return result


def main() -> None:  # pylint: disable=too-many-locals
    """Validate and merge the exact four-method by three-seed matrix."""
    args = parser().parse_args()
    source = Path(__file__).resolve().parents[2]
    source_commit = git_commit(source)
    if sha256_file(args.protocol) != args.expected_protocol_sha256:
        raise ValueError('single-GPU protocol SHA-256 mismatch')
    decision = json.loads(args.screen_decision.read_text(encoding='utf-8'))
    validate_screen_decision(
        decision,
        args.expected_protocol_sha256,
        args.expected_enhancement_preregistration_sha256,
    )
    arm_configs = {
        'C0': source / DEFAULT_ARM_CONFIGS['C0'],
        'P47': (
            args.p47_config.resolve()
            if args.p47_config is not None
            else source / DEFAULT_ARM_CONFIGS['P47']
        ),
    }
    config_hashes = {arm: sha256_file(path) for arm, path in arm_configs.items()}
    seed_rows: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    for arm in ARM_METHODS:
        for seed in TRAINING_SEEDS:
            evaluation = args.star_evaluation_root / arm / f'evaluation_seed_{seed}_step_{TRAINING_STEPS}.json'
            receipt = args.star_training_root / f'{arm}-{seed}' / 'receipt.json'
            row, records = load_star(arm, seed, evaluation, receipt, config_hashes[arm])
            seed_rows.append(row)
            episode_records.extend(records)
    for path in args.prior_evaluations:
        row, records = load_prior(path.resolve())
        seed_rows.append(row)
        episode_records.extend(records)
    identities = [(row['method'], int(row['training_seed'])) for row in seed_rows]
    expected = {(method, seed) for method in METHOD_ORDER for seed in TRAINING_SEEDS}
    if set(identities) != expected or len(identities) != len(expected):
        raise ValueError('inputs do not form the exact four-method by three-seed matrix')
    seed_rows.sort(key=lambda row: (METHOD_ORDER.index(row['method']), row['training_seed']))
    episode_records.sort(
        key=lambda row: (
            METHOD_ORDER.index(row['method']),
            row['training_seed'],
            row['base_seed'],
            row['episode_index'],
        )
    )
    aggregates = [
        aggregate_method(
            method,
            [row for row in seed_rows if row['method'] == method],
            [record for record in episode_records if record['method'] == method],
        )
        for method in METHOD_ORDER
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_rows = [
        {key: value for key, value in row.items() if key not in ('evidence_file', 'evidence_sha256', 'receipt_file', 'receipt_sha256')}
        for row in seed_rows
    ]
    atomic_csv(args.output_dir / 'pointcircle_four_method_seed_metrics.csv', public_rows)
    atomic_csv(args.output_dir / 'pointcircle_four_method_summary.csv', aggregates)
    atomic_csv(args.output_dir / 'pointcircle_four_method_episode_records.csv', episode_records)
    atomic_text(
        args.output_dir / 'POINTCIRCLE_FOUR_METHOD_REPORT.md',
        markdown_report(aggregates, args.expected_enhancement_preregistration_sha256),
    )
    evidence: list[dict[str, Any]] = []
    for row in seed_rows:
        evidence.append(
            {
                'method': row['method'],
                'training_seed': row['training_seed'],
                'path': row['evidence_file'],
                'sha256': row['evidence_sha256'],
            }
        )
        if row.get('receipt_file'):
            evidence.append(
                {
                    'method': row['method'],
                    'training_seed': row['training_seed'],
                    'path': row['receipt_file'],
                    'sha256': row['receipt_sha256'],
                }
            )
    manifest = {
        'status': 'complete',
        'environment': ENVIRONMENT,
        'protocol_sha256': args.expected_protocol_sha256,
        'finalizer_source_commit': source_commit,
        'statistical_unit': 'training_seed',
        'training_seeds_per_method': len(TRAINING_SEEDS),
        'training_steps_per_seed': TRAINING_STEPS,
        'evaluation_episodes_per_training_seed': EPISODES_PER_TRAINING_SEED,
        'aggregate_rows': aggregates,
        'screen_decision': str(args.screen_decision.resolve()),
        'screen_decision_sha256': sha256_file(args.screen_decision),
        'arm_configs': {
            arm: {'path': str(path.resolve()), 'sha256': config_hashes[arm]}
            for arm, path in arm_configs.items()
        },
        'enhancement_preregistration_sha256': (
            args.expected_enhancement_preregistration_sha256
        ),
        'evidence': evidence,
    }
    atomic_json(args.output_dir / 'pointcircle_four_method_manifest.json', manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
