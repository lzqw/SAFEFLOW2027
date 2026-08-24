#!/usr/bin/env python3
"""Build a sealed, explicitly non-formal SHAPO P4.7 40% snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_COMMIT = '62ca7399357c10645a955dcbead7ed2153a82c69'
PROTOCOL_SHA256 = '007c9a89d6b8f63d2c56516f9611d22e60164b82ccff939aa0af060d1dffa337'
MATRIX_SHA256 = 'f22d644f798e1e036d0ef64359d40ff10fb745d9d97eb2887ec6bb02adc66b0f'
FROZEN_METHOD_SHA256 = (
    'cf447f9ad97053054d1970a462a3089deffc83b332101f896b492621140f5f3d'
)
PRELIMINARY_PROTOCOL_RELATIVE = Path(
    'reports/star/shapo_p47_four_env_preliminary_40pct_preregistration.json',
)
PRELIMINARY_PROTOCOL_SHA256 = (
    '3e48cc7425ebf2900b610f1f62ab17dca19fd4720aa9be0d025e2069e493cf12'
)
TRAINING_SEED = 2027
EVALUATION_EPISODES = 100
SNAPSHOT_FRACTION = 0.4


@dataclass(frozen=True)
class Target:
    """One preregistered 40% milestone used by the preliminary snapshot."""

    environment: str
    slug: str
    milestone: int
    total_steps: int

    @property
    def relative_bundle(self) -> Path:
        """Return the synchronized milestone bundle relative to its root."""
        return Path(f'step_{self.milestone}') / f'{self.slug}-{TRAINING_SEED}'


TARGETS = (
    Target('SafetyPointGoal1-v0', 'point-goal', 4_000_000, 10_000_000),
    Target('SafetyPointButton1-v0', 'point-button', 4_000_000, 10_000_000),
    Target('Ant-v4', 'ant', 2_000_000, 5_000_000),
    Target('Walker2d-v4', 'walker2d', 2_000_000, 5_000_000),
)

PRIMARY_METRICS = (
    'mean_return',
    'mean_cost',
    'cost_cvar_20',
    'training_cumulative_cost',
)
DIAGNOSTIC_METRICS = (
    'success_rate',
    'failure_count',
    'failure_rate',
    'time_limit_count',
    'time_limit_rate',
    'high_risk_vn_vt',
    'high_risk_dangerous_outward_velocity',
)
COPIED_SOURCE_FILES = (
    'milestone_audit.json',
    'artifacts/evaluation.csv',
    'provenance/resolved_formal_config.yaml',
    'provenance/source_commit.txt',
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_preliminary_protocol() -> Path:
    """Require the preliminary scope declaration to retain its frozen identity."""
    path = Path(__file__).resolve().parents[2] / PRELIMINARY_PROTOCOL_RELATIVE
    if not path.is_file() or sha256_file(path) != PRELIMINARY_PROTOCOL_SHA256:
        raise ValueError(f'preliminary protocol identity mismatch: {path}')
    return path


def atomic_text(path: Path, value: str) -> None:
    """Write text atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(value, encoding='utf-8')
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    """Write strict JSON atomically."""
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a stable CSV atomically."""
    if not rows:
        raise ValueError('cannot write an empty preliminary table')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def finite_number(value: Any, label: str) -> float:
    """Require a finite numeric metric."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{label} is not finite')
    return result


def safe_relative_path(value: str) -> Path:
    """Require a normalized path contained by one milestone bundle."""
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or path == Path('.'):
        raise ValueError(f'unsafe checksum path: {value!r}')
    return path


def verify_bundle_checksums(bundle: Path) -> list[dict[str, Any]]:
    """Verify every synchronized compact artifact listed by the sealed audit."""
    checksum_path = bundle / 'checksums.sha256'
    if not checksum_path.is_file():
        raise ValueError(f'missing checksum manifest: {checksum_path}')
    identities: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for line in checksum_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            expected, raw_relative = line.split('  ', 1)
        except ValueError as error:
            raise ValueError(f'invalid checksum row in {checksum_path}') from error
        relative = safe_relative_path(raw_relative)
        if relative in seen:
            raise ValueError(f'duplicate checksum path: {relative}')
        seen.add(relative)
        source = bundle / relative
        if not source.is_file():
            raise ValueError(f'missing sealed artifact: {source}')
        actual = sha256_file(source)
        if actual != expected:
            raise ValueError(f'checksum mismatch for {source}')
        identities.append(
            {
                'artifact': relative.as_posix(),
                'bytes': source.stat().st_size,
                'sha256': actual,
            },
        )
    required = {
        Path('milestone_audit.json'),
        Path('artifacts/evaluation.csv'),
        Path('provenance/resolved_formal_config.yaml'),
        Path('provenance/source_commit.txt'),
    }
    if not required.issubset(seen):
        missing = ', '.join(sorted(path.as_posix() for path in required - seen))
        raise ValueError(f'checksum manifest omits required files: {missing}')
    return identities


def validate_audit(audit: dict[str, Any], target: Target, source: Path) -> dict[str, Any]:
    """Validate one sealed audit and return its preliminary result row."""
    expected = {
        'status': 'passed',
        'environment': target.environment,
        'training_seed': TRAINING_SEED,
        'milestone': target.milestone,
        'source_commit': SOURCE_COMMIT,
        'protocol_sha256': PROTOCOL_SHA256,
        'matrix_sha256': MATRIX_SHA256,
        'frozen_method_canonical_sha256': FROZEN_METHOD_SHA256,
        'training_seed_is_statistical_unit': True,
        'used_for_checkpoint_or_configuration_selection': False,
        'diagnostic_only': True,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise ValueError(f'{source} field {key!r} does not match the frozen snapshot')
    fraction = target.milestone / target.total_steps
    if not math.isclose(fraction, SNAPSHOT_FRACTION):
        raise ValueError(f'{target.environment} target is not exactly a 40% milestone')
    evaluation = audit.get('evaluation')
    if not isinstance(evaluation, dict):
        raise ValueError(f'{source} has no evaluation object')
    if int(evaluation.get('episodes', -1)) != EVALUATION_EPISODES:
        raise ValueError(f'{source} does not contain 100 fixed evaluation episodes')
    row: dict[str, Any] = {
        'environment': target.environment,
        'training_seed': TRAINING_SEED,
        'milestone': target.milestone,
        'total_steps': target.total_steps,
        'budget_fraction': fraction,
        'evaluation_episodes': EVALUATION_EPISODES,
        'formal_cost_budget': finite_number(
            audit.get('formal_cost_budget'),
            f'{target.environment}.formal_cost_budget',
        ),
    }
    for metric in (*PRIMARY_METRICS, *DIAGNOSTIC_METRICS):
        row[metric] = finite_number(
            evaluation.get(metric),
            f'{target.environment}.{metric}',
        )
    return row


def collect_targets(
    milestone_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Collect all available target bundles without mutating source evidence."""
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    missing: list[str] = []
    for target in TARGETS:
        relative_bundle = target.relative_bundle
        bundle = milestone_root / relative_bundle
        seal = bundle / 'milestone_audit_complete'
        if not seal.is_file():
            missing.append(relative_bundle.as_posix())
            continue
        identities = verify_bundle_checksums(bundle)
        audit_path = bundle / 'milestone_audit.json'
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        rows.append(validate_audit(audit, target, audit_path))
        sources.append(
            {
                'environment': target.environment,
                'training_seed': TRAINING_SEED,
                'milestone': target.milestone,
                'source_bundle': relative_bundle.as_posix(),
                'artifacts': identities,
            },
        )
    return rows, sources, missing


def copy_source_subset(milestone_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Copy the compact evidence needed to reproduce the preliminary table."""
    copied: list[dict[str, Any]] = []
    for target in TARGETS:
        bundle = milestone_root / target.relative_bundle
        destination_root = output_dir / 'sources' / target.slug
        evaluation_json = f'artifacts/evaluation_step_{target.milestone}.json'
        for relative in (*COPIED_SOURCE_FILES, evaluation_json):
            source = bundle / relative
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
            copied.append(
                {
                    'environment': target.environment,
                    'artifact': (Path('sources') / target.slug / relative).as_posix(),
                    'bytes': destination.stat().st_size,
                    'sha256': sha256_file(destination),
                },
            )
    return copied


def display(value: float, digits: int = 4) -> str:
    """Format one scalar without implying across-seed precision."""
    return f'{value:.{digits}f}'


def render_report(rows: list[dict[str, Any]], generated_at: str) -> str:
    """Render the explicit one-seed preliminary report."""
    lines = [
        '# SHAPO P4.7 Four-Environment 40% Snapshot — PRELIMINARY ONLY',
        '',
        '> **Not a formal result.** This snapshot contains one training seed (`n=1`) and '
        'must not be used for uncertainty, significance, or cross-seed generalization claims.',
        '',
        f'- Generated at: `{generated_at}`',
        f'- Frozen formal source commit: `{SOURCE_COMMIT}`',
        f'- Preliminary protocol SHA-256: `{PRELIMINARY_PROTOCOL_SHA256}`',
        '- Training seed: `2027`',
        '- Snapshot: exactly `40%` of each environment-specific training budget',
        '- Evaluation: 5 fixed base seeds × 20 episodes = 100 episodes per policy',
        '- Scientific early stopping: disabled in the continuing formal runs',
        '- Checkpoint/configuration selection from these results: no',
        '',
        '## Primary snapshot',
        '',
        '| Environment | Step | Budget | Return | Mean cost | CVaR-20 | Training cost |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['environment']} | {row['milestone']:,} | 40% | "
            f"{display(row['mean_return'])} | {display(row['mean_cost'])} | "
            f"{display(row['cost_cvar_20'])} | "
            f"{display(row['training_cumulative_cost'], 0)} |",
        )
    lines.extend(
        [
            '',
            'Metrics are absolute within-task values; their scales are not comparable across '
            'different environments.',
            '',
            '## Task diagnostics',
            '',
            '| Environment | Success rate | Failure rate | Time-limit rate |',
            '|---|---:|---:|---:|',
        ],
    )
    for row in rows:
        lines.append(
            f"| {row['environment']} | {display(row['success_rate'])} | "
            f"{display(row['failure_rate'])} | {display(row['time_limit_rate'])} |",
        )
    lines.extend(
        [
            '',
            '## High-risk direction diagnostics',
            '',
            '| Environment | Top-20% V_N/V_T | Dangerous outward velocity |',
            '|---|---:|---:|',
        ],
    )
    for row in rows:
        lines.append(
            f"| {row['environment']} | {display(row['high_risk_vn_vt'])} | "
            f"{display(row['high_risk_dangerous_outward_velocity'])} |",
        )
    lines.extend(
        [
            '',
            '## Interpretation boundary',
            '',
            '- This is an early frozen-P4.7 snapshot, not a method comparison.',
            '- The independent statistical unit is the training seed; there is only one here.',
            '- The 100 evaluation episodes do not convert this into `n=100` training repeats.',
            '- The complete 4-environment × 5-seed formal study continues unchanged.',
            '- Only the completed formal study may replace this preliminary snapshot.',
            '',
        ],
    )
    return '\n'.join(lines)


def verify_existing_completion(output_dir: Path) -> dict[str, Any] | None:
    """Return an existing immutable completion receipt when one is present."""
    seal = output_dir / 'preliminary_complete'
    summary_path = output_dir / 'preliminary_summary.json'
    if not seal.is_file():
        return None
    if not summary_path.is_file():
        raise ValueError('preliminary completion seal exists without its summary')
    expected = seal.read_text(encoding='utf-8').strip()
    actual = sha256_file(summary_path)
    if expected != actual:
        raise ValueError('sealed preliminary summary identity changed')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    return {
        'state': 'complete',
        'generated': False,
        'observed_at_utc': utc_now(),
        'summary_sha256': actual,
        'output_dir': str(output_dir),
        'rows': len(summary['results']),
    }


def build_once(milestone_root: Path, output_dir: Path) -> dict[str, Any]:
    """Build the snapshot once all four sealed 40% milestones are available."""
    preliminary_protocol = verify_preliminary_protocol()
    existing = verify_existing_completion(output_dir)
    if existing is not None:
        return existing
    observed_at = utc_now()
    missing = [
        target.relative_bundle.as_posix()
        for target in TARGETS
        if not (
            milestone_root / target.relative_bundle / 'milestone_audit_complete'
        ).is_file()
    ]
    if missing:
        return {
            'state': 'waiting',
            'observed_at_utc': observed_at,
            'available_targets': len(TARGETS) - len(missing),
            'total_targets': len(TARGETS),
            'missing_targets': missing,
        }
    rows, sources, unexpected_missing = collect_targets(milestone_root)
    if unexpected_missing:
        raise ValueError('a sealed target disappeared during final collection')
    if len(rows) != len(TARGETS):
        raise ValueError('complete target set has an unexpected row count')
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = copy_source_subset(milestone_root, output_dir)
    protocol_copy = output_dir / 'protocol' / preliminary_protocol.name
    protocol_copy.parent.mkdir(parents=True, exist_ok=True)
    protocol_temporary = protocol_copy.with_name(f'.{protocol_copy.name}.{os.getpid()}.tmp')
    shutil.copyfile(preliminary_protocol, protocol_temporary)
    os.replace(protocol_temporary, protocol_copy)
    summary = {
        'artifact_role': 'preliminary_non_formal_snapshot',
        'generated_at_utc': observed_at,
        'source_commit': SOURCE_COMMIT,
        'protocol_sha256': PROTOCOL_SHA256,
        'matrix_sha256': MATRIX_SHA256,
        'frozen_method_canonical_sha256': FROZEN_METHOD_SHA256,
        'preliminary_protocol_sha256': PRELIMINARY_PROTOCOL_SHA256,
        'training_seed': TRAINING_SEED,
        'n_training_seeds': 1,
        'training_seed_is_statistical_unit': True,
        'evaluation_episodes_per_policy': EVALUATION_EPISODES,
        'snapshot_budget_fraction': SNAPSHOT_FRACTION,
        'formal_study_continues_unchanged': True,
        'scientific_protocol_changed': False,
        'training_configuration_changed': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'formal_result': False,
        'results': rows,
    }
    atomic_csv(output_dir / 'preliminary_results.csv', rows)
    atomic_json(
        output_dir / 'source_manifest.json',
        {
            'source_milestone_bundles': sources,
            'copied_artifacts': copied,
            'preliminary_protocol': {
                'artifact': protocol_copy.relative_to(output_dir).as_posix(),
                'bytes': protocol_copy.stat().st_size,
                'sha256': sha256_file(protocol_copy),
            },
        },
    )
    atomic_text(output_dir / 'PRELIMINARY_REPORT.md', render_report(rows, observed_at))
    atomic_text(
        output_dir / 'PRELIMINARY_ONLY',
        'One training seed at 40% budget; not a formal result.\n',
    )
    atomic_json(output_dir / 'preliminary_summary.json', summary)
    checksummed = []
    for path in sorted(output_dir.rglob('*')):
        if not path.is_file() or path.name in {'checksums.sha256', 'preliminary_complete'}:
            continue
        checksummed.append(f'{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}')
    atomic_text(output_dir / 'checksums.sha256', '\n'.join(checksummed) + '\n')
    summary_sha256 = sha256_file(output_dir / 'preliminary_summary.json')
    atomic_text(output_dir / 'preliminary_complete', summary_sha256 + '\n')
    return {
        'state': 'complete',
        'generated': True,
        'observed_at_utc': observed_at,
        'summary_sha256': summary_sha256,
        'output_dir': str(output_dir),
        'rows': len(rows),
    }


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--milestone-root', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--state-dir', type=Path, required=True)
    result.add_argument('--poll-seconds', type=float, default=300.0)
    result.add_argument('--once', action='store_true')
    return result


def main() -> None:
    """Wait for the four sealed targets and build the preliminary bundle once."""
    args = parser().parse_args()
    if args.poll_seconds <= 0:
        raise ValueError('--poll-seconds must be positive')
    args.state_dir.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            snapshot = build_once(args.milestone_root, args.output_dir)
        except Exception as error:
            snapshot = {
                'state': 'error',
                'observed_at_utc': utc_now(),
                'error': f'{type(error).__name__}: {error}',
            }
            atomic_json(args.state_dir / 'preliminary_watch_latest.json', snapshot)
            raise
        atomic_json(args.state_dir / 'preliminary_watch_latest.json', snapshot)
        print(json.dumps(snapshot, sort_keys=True), flush=True)
        if snapshot['state'] == 'complete' or args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
