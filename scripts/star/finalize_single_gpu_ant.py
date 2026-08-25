#!/usr/bin/env python3
"""Finalize the preregistered three-seed Ant C0/P4.7 comparison."""

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

ENVIRONMENT = 'Ant-v4'
ARMS = ('C0', 'P47')
METHODS = {'C0': 'C0 / isotropic FLAC', 'P47': 'P4.7 / Ours'}
TRAINING_SEEDS = (2027, 2028, 2029)
BASE_SEEDS = (32_027, 32_028, 32_029, 32_030, 32_031)
EPISODES_PER_BASE_SEED = 20
EPISODES_PER_TRAINING_SEED = 100
TRAINING_STEPS = 2_000_000
TRAINING_SOURCE_COMMIT = '62ca7399357c10645a955dcbead7ed2153a82c69'
EXPECTED_PROTOCOL_SHA256 = '5769bd35b603733dd038f94aea1b7dbc45cfd5335314788836e316781bed2692'
CONFIG_RELATIVE = {
    'C0': Path('configs/star/single_gpu_ant_c0.yaml'),
    'P47': Path('configs/star/single_gpu_ant_p47.yaml'),
}
CONFIG_SHA256 = {
    'C0': 'db16e3ed72b0a23a47bd9fd7db0f6aea863e8e1f338a35ada3d25618db4848b9',
    'P47': '4d011ec9c0eaedf1f51835804c3e0476114e4a0427d8ef32a5b9a0336fc97ebf',
}
REUSED_AUDIT_SHA256 = '9e5102da78dae0ef24712df1c081a9f36026214a9e784fc7f2ab925df264aa14'
REUSED_EVALUATION_SHA256 = '54dce88814b8e7072613cc17e50b3f5c38995f7ffdde34f63a55b5de07a76a24'
REUSED_CHECKPOINT_SHA256 = '39a9d7a0717402d5e3fea5694cd0724aebd0982306beb1fb39c8091135fddc35'


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} does not contain a JSON object')
    return payload


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
        raise ValueError('cannot write an empty Ant table')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    """Write text through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)


def finite(value: Any, label: str) -> float:
    """Require one finite numeric value."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{label} is not finite')
    return number


def tail_mean(values: list[float], fraction: float = 0.2) -> float:
    """Return the empirical upper-tail cost mean."""
    count = max(1, math.ceil(len(values) * fraction))
    return statistics.fmean(sorted(values, reverse=True)[:count])


def sample_std(values: list[float]) -> float:
    """Return sample standard deviation."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def require_close(observed: Any, expected: float, label: str) -> None:
    """Require a stored summary to agree with raw episode data."""
    if not math.isclose(float(observed), expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f'{label}: stored {observed} disagrees with raw {expected}')


def expected_schedule() -> list[tuple[int, int]]:
    """Return the frozen five-by-twenty sequential episode schedule."""
    return [
        (seed, episode_index)
        for seed in BASE_SEEDS
        for episode_index in range(EPISODES_PER_BASE_SEED)
    ]


def validate_evaluation(
    path: Path,
    arm: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one Ant evaluation and recompute its primary metrics."""
    payload = read_json(path)
    label = f'{arm}:{seed}'
    if (
        int(payload.get('milestone', -1)) != TRAINING_STEPS
        or int(payload.get('episodes', -1)) != EPISODES_PER_TRAINING_SEED
        or int(payload.get('episodes_per_base_seed', -1)) != EPISODES_PER_BASE_SEED
        or [int(value) for value in payload.get('base_seeds', [])] != list(BASE_SEEDS)
    ):
        raise ValueError(f'{label}: wrong evaluation schedule')
    eval_seeds = [int(value) for value in payload.get('seeds', [])]
    indices = [int(value) for value in payload.get('episode_indices', [])]
    returns = [finite(value, f'{label}.return') for value in payload.get('returns', [])]
    costs = [finite(value, f'{label}.cost') for value in payload.get('costs', [])]
    failures = [bool(value) for value in payload.get('episode_failure_flags', [])]
    arrays = (eval_seeds, indices, returns, costs, failures)
    if not all(len(values) == EPISODES_PER_TRAINING_SEED for values in arrays):
        raise ValueError(f'{label}: malformed raw episode arrays')
    if list(zip(eval_seeds, indices)) != expected_schedule():
        raise ValueError(f'{label}: evaluation stream is incomplete or reordered')
    mean_return = statistics.fmean(returns)
    mean_cost = statistics.fmean(costs)
    cost_cvar = tail_mean(costs)
    failure_rate = statistics.fmean(float(value) for value in failures)
    for key, value in (
        ('mean_return', mean_return),
        ('mean_cost', mean_cost),
        ('cost_cvar_20', cost_cvar),
        ('failure_rate', failure_rate),
    ):
        require_close(payload[key], value, f'{label}.{key}')
    geometry = payload.get('direction_geometry_high_risk_top_20_percent', {})
    row = {
        'method': METHODS[arm],
        'arm': arm,
        'training_seed': seed,
        'training_steps': TRAINING_STEPS,
        'episodes': EPISODES_PER_TRAINING_SEED,
        'mean_return': mean_return,
        'mean_cost': mean_cost,
        'cost_cvar_20': cost_cvar,
        'failure_rate': failure_rate,
        'training_cumulative_cost': finite(
            payload.get('training_cumulative_cost'),
            f'{label}.training_cumulative_cost',
        ),
        'high_risk_vn_vt': finite(
            geometry.get('normal_to_tangent_variance_ratio'),
            f'{label}.high_risk_vn_vt',
        ),
        'high_risk_dangerous_outward_velocity': finite(
            geometry.get('dangerous_outward_velocity'),
            f'{label}.high_risk_dangerous_outward_velocity',
        ),
        'evidence_file': str(path.resolve()),
        'evidence_sha256': sha256_file(path),
    }
    records = [
        {
            'method': METHODS[arm],
            'arm': arm,
            'training_seed': seed,
            'base_seed': base_seed,
            'episode_index': episode_index,
            'return': episode_return,
            'cost': episode_cost,
            'failure': failure,
        }
        for base_seed, episode_index, episode_return, episode_cost, failure in zip(
            eval_seeds,
            indices,
            returns,
            costs,
            failures,
        )
    ]
    return row, records


def unique_evaluation(container: Path) -> Path:
    """Resolve exactly one final 2M evaluation below a training container."""
    matches = sorted(
        container.glob(
            f'training/STAR-{{{ENVIRONMENT}}}/seed-*/artifacts/evaluation_step_{TRAINING_STEPS}.json',
        ),
    )
    if len(matches) != 1:
        raise ValueError(f'expected one Ant 2M evaluation below {container}, found {len(matches)}')
    return matches[0]


def unique_checkpoint(container: Path) -> Path:
    """Resolve exactly one server-only final 2M checkpoint."""
    matches = sorted(
        container.glob(
            f'training/STAR-{{{ENVIRONMENT}}}/seed-*/checkpoints/milestone_{TRAINING_STEPS}.pt',
        ),
    )
    if len(matches) != 1:
        raise ValueError(f'expected one Ant 2M checkpoint below {container}, found {len(matches)}')
    return matches[0]


def validate_receipt(path: Path, arm: str, seed: int) -> dict[str, Any]:
    """Validate one new Ant training receipt."""
    payload = read_json(path)
    valid = (
        payload.get('status') == 'passed'
        and payload.get('algorithm') == 'STAR'
        and payload.get('environment') == ENVIRONMENT
        and int(payload.get('seed', -1)) == seed
        and int(payload.get('steps', -1)) == TRAINING_STEPS
        and payload.get('source_commit') == TRAINING_SOURCE_COMMIT
        and payload.get('source_dirty') is False
        and payload.get('config_sha256') == CONFIG_SHA256[arm]
    )
    if not valid:
        raise ValueError(f'{arm}:{seed}: invalid training receipt')
    return payload


def aggregate(arm: str, rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one arm while retaining training-seed variability."""
    if sorted(int(row['training_seed']) for row in rows) != list(TRAINING_SEEDS):
        raise ValueError(f'{arm}: incomplete training-seed matrix')
    if len(records) != len(TRAINING_SEEDS) * EPISODES_PER_TRAINING_SEED:
        raise ValueError(f'{arm}: incomplete episode matrix')
    returns = [float(row['return']) for row in records]
    costs = [float(row['cost']) for row in records]
    failures = [float(bool(row['failure'])) for row in records]
    return {
        'method': METHODS[arm],
        'arm': arm,
        'training_steps_per_seed': TRAINING_STEPS,
        'training_seeds': len(rows),
        'pooled_evaluation_episodes': len(records),
        'mean_return': statistics.fmean(returns),
        'seed_std_mean_return': sample_std([float(row['mean_return']) for row in rows]),
        'mean_cost': statistics.fmean(costs),
        'seed_std_mean_cost': sample_std([float(row['mean_cost']) for row in rows]),
        'cost_cvar_20': tail_mean(costs),
        'seed_std_cost_cvar_20': sample_std([float(row['cost_cvar_20']) for row in rows]),
        'failure_rate': statistics.fmean(failures),
        'seed_std_failure_rate': sample_std([float(row['failure_rate']) for row in rows]),
        'training_cumulative_cost_mean': statistics.fmean(
            float(row['training_cumulative_cost']) for row in rows
        ),
        'seed_std_training_cumulative_cost': sample_std(
            [float(row['training_cumulative_cost']) for row in rows],
        ),
        'high_risk_vn_vt': statistics.fmean(float(row['high_risk_vn_vt']) for row in rows),
        'high_risk_dangerous_outward_velocity': statistics.fmean(
            float(row['high_risk_dangerous_outward_velocity']) for row in rows
        ),
    }


def relative_change(candidate: float, baseline: float) -> float | None:
    """Return signed relative change, or null for a zero baseline."""
    if abs(baseline) <= 1e-12:
        return None
    return (candidate - baseline) / abs(baseline)


def markdown_report(aggregates: list[dict[str, Any]]) -> str:
    """Render the formal Ant comparison with its claim boundary."""
    rows = {str(row['arm']): row for row in aggregates}
    c0 = rows['C0']
    ours = rows['P47']
    return_change = relative_change(float(ours['mean_return']), float(c0['mean_return']))
    cost_change = relative_change(float(ours['mean_cost']), float(c0['mean_cost']))
    cvar_change = relative_change(float(ours['cost_cvar_20']), float(c0['cost_cvar_20']))
    lines = [
        '# Ant 2M C0/P4.7 formal comparison',
        '',
        (
            'Both methods use training seeds `2027-2029`, an equal `2M` environment-step budget, '
            'and `100` fixed-stream final episodes per checkpoint. The independent statistical '
            'unit is the training seed (`n=3`). P4.7 seed 2027 is reused from the sealed prior run.'
        ),
        '',
        '| Method | Return | Mean cost | CVaR-20 | Failure | Train cost | High-risk V_N/V_T | Outward |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        row = rows[arm]
        lines.append(
            f"| {row['method']} | {row['mean_return']:.3f} | {row['mean_cost']:.3f} | "
            f"{row['cost_cvar_20']:.3f} | {100.0 * row['failure_rate']:.1f}% | "
            f"{row['training_cumulative_cost_mean']:.1f} | {row['high_risk_vn_vt']:.4f} | "
            f"{row['high_risk_dangerous_outward_velocity']:.4f} |",
        )
    lines.extend(
        [
            '',
            '## Aggregate comparison',
            '',
            f'Return relative change: `{return_change:+.1%}`.'
            if return_change is not None
            else 'Return relative change is undefined because the C0 aggregate is zero.',
            f'Mean-cost relative change: `{cost_change:+.1%}`.'
            if cost_change is not None
            else 'Mean-cost relative change is undefined because the C0 aggregate is zero.',
            f'CVaR-20 relative change: `{cvar_change:+.1%}`.'
            if cvar_change is not None
            else 'CVaR-20 relative change is undefined because the C0 aggregate is zero.',
            '',
            (
                'These are aggregate three-training-seed results. They do not establish that '
                'every individual seed is a Pareto improvement.'
            ),
            '',
        ],
    )
    return '\n'.join(lines)


def git_commit(source: Path) -> str:
    """Require a clean finalizer source and return its commit."""
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError('Ant finalization requires a clean controller source')
    return subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parser() -> argparse.ArgumentParser:
    """Build finalization arguments."""
    root = Path(__file__).resolve().parents[2]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run-root', type=Path, required=True)
    result.add_argument('--reused-evaluation', type=Path, required=True)
    result.add_argument('--reused-audit', type=Path, required=True)
    result.add_argument(
        '--protocol',
        type=Path,
        default=root / 'reports/star/single_gpu_ant_preregistration.json',
    )
    result.add_argument('--expected-protocol-sha256', default=EXPECTED_PROTOCOL_SHA256)
    result.add_argument('--output-dir', type=Path, required=True)
    return result


def main() -> None:  # pylint: disable=too-many-locals
    """Validate, aggregate, and seal the complete Ant matrix."""
    args = parser().parse_args()
    source = Path(__file__).resolve().parents[2]
    finalizer_commit = git_commit(source)
    if sha256_file(args.protocol) != args.expected_protocol_sha256:
        raise ValueError('Ant preregistration SHA-256 mismatch')
    if sha256_file(args.reused_audit) != REUSED_AUDIT_SHA256:
        raise ValueError('reused Ant audit SHA-256 mismatch')
    if sha256_file(args.reused_evaluation) != REUSED_EVALUATION_SHA256:
        raise ValueError('reused Ant evaluation SHA-256 mismatch')
    audit = read_json(args.reused_audit)
    if (
        audit.get('status') != 'passed'
        or audit.get('environment') != ENVIRONMENT
        or int(audit.get('training_seed', -1)) != 2027
        or int(audit.get('milestone', -1)) != TRAINING_STEPS
        or audit.get('source_commit') != TRAINING_SOURCE_COMMIT
    ):
        raise ValueError('reused Ant seed-2027 audit is invalid')
    for arm, relative in CONFIG_RELATIVE.items():
        if sha256_file(source / relative) != CONFIG_SHA256[arm]:
            raise ValueError(f'{arm} Ant config SHA-256 mismatch')

    seed_rows: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    reused_row, reused_records = validate_evaluation(args.reused_evaluation, 'P47', 2027)
    reused_checkpoint = Path(str(audit['server_only_checkpoint']['artifact']))
    if (
        not reused_checkpoint.is_file()
        or audit['server_only_checkpoint'].get('sha256') != REUSED_CHECKPOINT_SHA256
        or sha256_file(reused_checkpoint) != REUSED_CHECKPOINT_SHA256
    ):
        raise ValueError('reused Ant seed-2027 checkpoint is missing or changed')
    reused_row['receipt_file'] = None
    reused_row['receipt_sha256'] = None
    reused_row['checkpoint_file'] = str(reused_checkpoint.resolve())
    reused_row['checkpoint_sha256'] = REUSED_CHECKPOINT_SHA256
    reused_row['evidence_role'] = 'reused_sealed_milestone'
    seed_rows.append(reused_row)
    episode_records.extend(reused_records)
    for arm in ARMS:
        for seed in TRAINING_SEEDS:
            if arm == 'P47' and seed == 2027:
                continue
            container = args.run_root / 'runs' / f'{arm}-{seed}'
            receipt_path = container / 'receipt.json'
            receipt = validate_receipt(receipt_path, arm, seed)
            evaluation_path = unique_evaluation(container)
            checkpoint_path = unique_checkpoint(container)
            row, records = validate_evaluation(evaluation_path, arm, seed)
            require_close(
                receipt.get('training_summary', {}).get('cumulative_training_cost'),
                float(row['training_cumulative_cost']),
                f'{arm}:{seed}.receipt_training_cost',
            )
            row['receipt_file'] = str(receipt_path.resolve())
            row['receipt_sha256'] = sha256_file(receipt_path)
            row['checkpoint_file'] = str(checkpoint_path.resolve())
            row['checkpoint_sha256'] = sha256_file(checkpoint_path)
            row['evidence_role'] = 'new_preregistered_training'
            seed_rows.append(row)
            episode_records.extend(records)

    identities = {(row['arm'], int(row['training_seed'])) for row in seed_rows}
    expected = {(arm, seed) for arm in ARMS for seed in TRAINING_SEEDS}
    if identities != expected or len(seed_rows) != len(expected):
        raise ValueError('inputs do not form the exact two-arm by three-seed Ant matrix')
    seed_rows.sort(key=lambda row: (ARMS.index(str(row['arm'])), row['training_seed']))
    episode_records.sort(
        key=lambda row: (
            ARMS.index(str(row['arm'])),
            row['training_seed'],
            row['base_seed'],
            row['episode_index'],
        ),
    )
    aggregates = [
        aggregate(
            arm,
            [row for row in seed_rows if row['arm'] == arm],
            [row for row in episode_records if row['arm'] == arm],
        )
        for arm in ARMS
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            not in (
                'evidence_file',
                'evidence_sha256',
                'receipt_file',
                'receipt_sha256',
                'checkpoint_file',
                'checkpoint_sha256',
            )
        }
        for row in seed_rows
    ]
    atomic_csv(args.output_dir / 'ant_seed_metrics.csv', public_rows)
    atomic_csv(args.output_dir / 'ant_summary.csv', aggregates)
    atomic_csv(args.output_dir / 'ant_episode_records.csv', episode_records)
    atomic_text(args.output_dir / 'ANT_2M_FORMAL_REPORT.md', markdown_report(aggregates))
    evidence = []
    for row in seed_rows:
        evidence.append(
            {
                'arm': row['arm'],
                'training_seed': row['training_seed'],
                'role': row['evidence_role'],
                'path': row['evidence_file'],
                'sha256': row['evidence_sha256'],
            },
        )
        if row.get('receipt_file'):
            evidence.append(
                {
                    'arm': row['arm'],
                    'training_seed': row['training_seed'],
                    'role': 'training_receipt',
                    'path': row['receipt_file'],
                    'sha256': row['receipt_sha256'],
                },
            )
        evidence.append(
            {
                'arm': row['arm'],
                'training_seed': row['training_seed'],
                'role': 'server_only_checkpoint',
                'path': row['checkpoint_file'],
                'sha256': row['checkpoint_sha256'],
            },
        )
    manifest = {
        'status': 'complete',
        'environment': ENVIRONMENT,
        'protocol_sha256': args.expected_protocol_sha256,
        'finalizer_source_commit': finalizer_commit,
        'training_source_commit': TRAINING_SOURCE_COMMIT,
        'statistical_unit': 'training_seed',
        'training_seeds_per_arm': len(TRAINING_SEEDS),
        'training_steps_per_seed': TRAINING_STEPS,
        'evaluation_episodes_per_training_seed': EPISODES_PER_TRAINING_SEED,
        'every_seed_pareto_claim': False,
        'aggregate_rows': aggregates,
        'arm_configs': {
            arm: {
                'path': str((source / relative).resolve()),
                'sha256': CONFIG_SHA256[arm],
            }
            for arm, relative in CONFIG_RELATIVE.items()
        },
        'reused_seed': {
            'arm': 'P47',
            'training_seed': 2027,
            'audit': str(args.reused_audit.resolve()),
            'audit_sha256': REUSED_AUDIT_SHA256,
        },
        'evidence': evidence,
    }
    manifest_path = args.output_dir / 'ant_manifest.json'
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
