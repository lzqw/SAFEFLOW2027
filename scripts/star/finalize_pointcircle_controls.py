#!/usr/bin/env python3
"""Finalize the single-seed PointCircle mechanism and variance-ratio evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from scripts.star.decide_pointcircle_geometry_enhancement import ADDENDUM_SHA256

ENVIRONMENT = 'SafetyPointCircle1-v0'
SEED = 2027
STEPS = 300_000
BASE_SEEDS = (22_027, 22_028, 22_029, 22_030, 22_031)
EPISODES = 100
VARIANT_ORDER = ('Iso-High', 'Iso-Low', 'Random-Aniso', 'Safety-Aniso', 'Safety-Aniso-0.25')
ORIGINAL_CONFIGS = {
    'Iso-High': Path('configs/star/single_gpu_pointcircle_c0.yaml'),
    'Safety-Aniso': Path('configs/star/single_gpu_pointcircle_p47.yaml'),
    'Iso-Low': Path('configs/star/single_gpu_pointcircle_iso_low.yaml'),
    'Random-Aniso': Path('configs/star/single_gpu_pointcircle_random_aniso.yaml'),
    'Safety-Aniso-0.25': Path('configs/star/single_gpu_pointcircle_ratio_025.yaml'),
}
ENHANCED_CONFIGS = {
    'Iso-High': Path('configs/star/single_gpu_pointcircle_c0.yaml'),
    'Safety-Aniso': Path('configs/star/single_gpu_pointcircle_p47_geometry_enhanced.yaml'),
    'Iso-Low': Path('configs/star/single_gpu_pointcircle_iso_low_geometry_enhanced.yaml'),
    'Random-Aniso': Path(
        'configs/star/single_gpu_pointcircle_random_aniso_geometry_enhanced.yaml',
    ),
    'Safety-Aniso-0.25': Path(
        'configs/star/single_gpu_pointcircle_ratio_025_geometry_enhanced.yaml',
    ),
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
    """Write one homogeneous CSV through an atomic rename."""
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    """Write UTF-8 text through an atomic rename."""
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)


def finite(value: Any, label: str) -> float:
    """Require one finite numeric value."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{label} is not finite')
    return number


def tail_mean(values: list[float]) -> float:
    """Return empirical CVaR-20."""
    count = max(1, math.ceil(0.2 * len(values)))
    return sum(sorted(values, reverse=True)[:count]) / count


def load_variant(
    variant: str,
    evaluation: Path,
    receipt: Path,
    config_sha256: str,
) -> dict[str, Any]:
    """Validate one 300k control from raw final episodes and its receipt."""
    result = json.loads(evaluation.read_text(encoding='utf-8'))
    training = json.loads(receipt.read_text(encoding='utf-8'))
    label = variant
    valid_result = (
        result.get('status') == 'complete'
        and result.get('variant') == variant
        and result.get('environment') == ENVIRONMENT
        and int(result.get('training_seed', -1)) == SEED
        and int(result.get('training_steps', -1)) == STEPS
        and result.get('evaluation_mode') == 'fixed-random-flow'
        and result.get('evaluation_role') == 'primary'
        and [int(seed) for seed in result.get('base_seeds', [])] == list(BASE_SEEDS)
        and int(result.get('episodes_per_base_seed', -1)) == 20
        and int(result.get('episodes', -1)) == EPISODES
    )
    if not valid_result:
        raise ValueError(f'{variant}: invalid final evaluation')
    valid_receipt = (
        training.get('status') == 'passed'
        and training.get('environment') == ENVIRONMENT
        and int(training.get('seed', -1)) == SEED
        and int(training.get('steps', -1)) == STEPS
        and training.get('config_sha256') == config_sha256
        and training.get('source_dirty') is False
    )
    if not valid_receipt:
        raise ValueError(f'{variant}: invalid training receipt')
    returns = [finite(value, f'{label}.return') for value in result['returns']]
    costs = [finite(value, f'{label}.cost') for value in result['costs']]
    if len(returns) != EPISODES or len(costs) != EPISODES:
        raise ValueError(f'{variant}: incomplete raw episodes')
    mean_return = sum(returns) / len(returns)
    mean_cost = sum(costs) / len(costs)
    cvar = tail_mean(costs)
    for key, expected in (
        ('mean_return', mean_return),
        ('mean_cost', mean_cost),
        ('cost_cvar_20', cvar),
    ):
        if not math.isclose(float(result[key]), expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f'{variant}.{key} disagrees with raw episodes')
    geometry = result.get('direction_geometry_high_risk_top_20_percent', {})
    checkpoint = Path(result['checkpoint'])
    if not checkpoint.is_file() or sha256_file(checkpoint) != result.get('checkpoint_sha256'):
        raise ValueError(f'{variant}: checkpoint identity mismatch')
    return {
        'variant': variant,
        'training_seed': SEED,
        'training_steps': STEPS,
        'episodes': EPISODES,
        'mean_return': mean_return,
        'mean_cost': mean_cost,
        'cost_cvar_20': cvar,
        'training_cumulative_cost': finite(
            training.get('training_summary', {}).get('cumulative_training_cost'),
            f'{variant}.training_cumulative_cost',
        ),
        'high_risk_normal_variance': finite(
            geometry.get('normal_variance'),
            f'{variant}.normal_variance',
        ),
        'high_risk_tangent_variance': finite(
            geometry.get('tangent_variance'),
            f'{variant}.tangent_variance',
        ),
        'high_risk_vn_vt': finite(
            geometry.get('normal_to_tangent_variance_ratio'),
            f'{variant}.vn_vt',
        ),
        'high_risk_dangerous_outward_velocity': finite(
            geometry.get('dangerous_outward_velocity'),
            f'{variant}.outward_velocity',
        ),
        'evaluation_path': str(evaluation.resolve()),
        'evaluation_sha256': sha256_file(evaluation),
        'receipt_path': str(receipt.resolve()),
        'receipt_sha256': sha256_file(receipt),
    }


def markdown(rows: list[dict[str, Any]], geometry_enhanced: bool = False) -> str:
    """Render mechanism facts without introducing a new selection gate."""
    by_variant = {row['variant']: row for row in rows}
    safety = by_variant['Safety-Aniso']
    low = by_variant['Iso-Low']
    random = by_variant['Random-Aniso']
    lines = [
        '# PointCircle mechanism and normal-variance-ratio evidence',
        '',
        'All rows use training seed `2027`, exactly `300k` steps, and 100 fixed-stream final '
        'episodes. This is a mechanism diagnostic (`n=1`), not a multi-seed main result.',
        '',
    ]
    if geometry_enhanced:
        lines.extend(
            [
                'Safety-Aniso, Random-Aniso, and the 0.25 ratio point share the single locked '
                f'geometry enhancement (`{ADDENDUM_SHA256}`).',
                '',
            ],
        )
    lines.extend(
        [
            '| Variant | Return | Mean cost | CVaR-20 | Train cost | V_N | V_T | V_N/V_T | Outward |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        ],
    )
    for variant in VARIANT_ORDER:
        row = by_variant[variant]
        lines.append(
            f"| {variant} | {row['mean_return']:.4f} | {row['mean_cost']:.3f} | "
            f"{row['cost_cvar_20']:.3f} | {row['training_cumulative_cost']:.1f} | "
            f"{row['high_risk_normal_variance']:.5f} | "
            f"{row['high_risk_tangent_variance']:.5f} | {row['high_risk_vn_vt']:.4f} | "
            f"{row['high_risk_dangerous_outward_velocity']:.5f} |"
        )
    lines.extend(
        [
            '',
            '## Prespecified evidence contrasts',
            '',
            f"- Safety-Aniso minus Iso-Low: V_N `{safety['high_risk_normal_variance'] - low['high_risk_normal_variance']:+.5f}`, "
            f"V_T `{safety['high_risk_tangent_variance'] - low['high_risk_tangent_variance']:+.5f}`, "
            f"return `{safety['mean_return'] - low['mean_return']:+.4f}`, mean cost "
            f"`{safety['mean_cost'] - low['mean_cost']:+.3f}`.",
            f"- Safety-Aniso minus Random-Aniso: return "
            f"`{safety['mean_return'] - random['mean_return']:+.4f}`, mean cost "
            f"`{safety['mean_cost'] - random['mean_cost']:+.3f}`, CVaR-20 "
            f"`{safety['cost_cvar_20'] - random['cost_cvar_20']:+.3f}`.",
            '',
            'The comparisons are reported descriptively; no post-hoc pass/fail threshold or '
            'additional weight search is introduced.',
            '',
        ]
    )
    return '\n'.join(lines)


def git_commit(source: Path) -> str:
    """Require clean finalizer source."""
    status = subprocess.run(
        ['git', '-C', str(source), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError('control finalization requires a clean source')
    return subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parser() -> argparse.ArgumentParser:
    """Build finalization arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--star-screen-root', type=Path, required=True)
    result.add_argument('--safety-aniso-container', type=Path)
    result.add_argument('--control-root', type=Path, required=True)
    result.add_argument('--evaluation-root', type=Path, required=True)
    result.add_argument('--protocol', type=Path, required=True)
    result.add_argument('--expected-protocol-sha256', required=True)
    result.add_argument('--geometry-enhanced', action='store_true')
    result.add_argument('--output-dir', type=Path, required=True)
    return result


def main() -> None:
    """Merge reused and new 300k controls into one authenticated package."""
    args = parser().parse_args()
    source = Path(__file__).resolve().parents[2]
    source_commit = git_commit(source)
    if sha256_file(args.protocol) != args.expected_protocol_sha256:
        raise ValueError('single-GPU protocol SHA-256 mismatch')
    configs = ENHANCED_CONFIGS if args.geometry_enhanced else ORIGINAL_CONFIGS
    safety_aniso_container = (
        args.safety_aniso_container.resolve()
        if args.safety_aniso_container is not None
        else args.star_screen_root / 'P47-2027'
    )
    receipts = {
        'Iso-High': args.star_screen_root / 'C0-2027/receipt.json',
        'Safety-Aniso': safety_aniso_container / 'receipt.json',
        'Iso-Low': args.control_root / 'mechanism/Iso-Low-2027/receipt.json',
        'Random-Aniso': args.control_root / 'mechanism/Random-Aniso-2027/receipt.json',
        'Safety-Aniso-0.25': args.control_root / 'parameter/Safety-Aniso-0.25-2027/receipt.json',
    }
    rows = [
        load_variant(
            variant,
            args.evaluation_root / f'{variant.lower().replace("-", "_")}.json',
            receipts[variant],
            sha256_file(source / configs[variant]),
        )
        for variant in VARIANT_ORDER
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public = [
        {key: value for key, value in row.items() if not key.endswith(('_path', '_sha256'))}
        for row in rows
    ]
    atomic_csv(args.output_dir / 'pointcircle_control_metrics.csv', public)
    atomic_text(
        args.output_dir / 'POINTCIRCLE_CONTROL_REPORT.md',
        markdown(rows, args.geometry_enhanced),
    )
    manifest = {
        'status': 'complete',
        'environment': ENVIRONMENT,
        'training_seed': SEED,
        'training_steps': STEPS,
        'evaluation_episodes_per_variant': EPISODES,
        'protocol_sha256': args.expected_protocol_sha256,
        'geometry_enhanced': args.geometry_enhanced,
        'enhancement_preregistration_sha256': (
            ADDENDUM_SHA256 if args.geometry_enhanced else None
        ),
        'config_sha256': {
            variant: sha256_file(source / configs[variant]) for variant in VARIANT_ORDER
        },
        'finalizer_source_commit': source_commit,
        'rows': public,
        'evidence': [
            {
                'variant': row['variant'],
                'evaluation_path': row['evaluation_path'],
                'evaluation_sha256': row['evaluation_sha256'],
                'receipt_path': row['receipt_path'],
                'receipt_sha256': row['receipt_sha256'],
            }
            for row in rows
        ],
    }
    atomic_json(args.output_dir / 'pointcircle_control_manifest.json', manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
