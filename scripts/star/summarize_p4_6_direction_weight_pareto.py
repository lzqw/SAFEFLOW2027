#!/usr/bin/env python3
"""Validate P4.6 candidates and report the sealed four-weight Pareto comparison."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scripts.star import summarize_p4_5_unified_cost_geometry as p45


PROTOCOL_VERSION = 'p4.6-direction-weight-pareto-v1'
PROTOCOL_SHA256 = 'c473da2ed99b31cde673a1cd038ec0477053b23ac63a130376ebccc4271ad93e'
P45_PROTOCOL_SHA256 = '8abebc101021ec71a00fdf6d2c5580b5f97dada710e58de00936a75490e85c11'
P45_SUMMARY_SHA256 = '4be7910f27350afa83c3d375728b425af1b1f10db4e5a8460bd32514d5f33236'
P45_FINAL_ROWS_SHA256 = '6e782b5449667da72399d59c9882fbe698e6c66e9a8ab50bf398986b64daf7b3'
ARMS = ('C0', 'C025', 'C050', 'C1')
CANDIDATES = ('C025', 'C050')
ARM_WEIGHTS = {'C0': 0.0, 'C025': 0.25, 'C050': 0.50, 'C1': 1.0}
SEEDS = (2027, 2028, 2029)
TARGET_STEPS = 100_000
EVAL_STEPS = (0, 25_000, 50_000, 75_000, 100_000)
CHECKPOINT_STEPS = (0, 10_000, 25_000, 50_000, 75_000, 100_000)
EVAL_SEEDS = tuple(range(12_027, 12_047))
TRAJECTORY_SEEDS = (12_027, 12_036, 12_046)


def _bindings(values: list[str], label: str) -> dict[tuple[str, int], Path]:
    """Parse exactly six candidate ``ARM:SEED=PATH`` bindings."""
    expected = {(arm, seed) for arm in CANDIDATES for seed in SEEDS}
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
        raise ValueError(f'{label} must bind C025/C050 at seeds {SEEDS}')
    return result


def _validate_checksum_manifest(root: Path) -> None:
    """Verify every sealed P4.5 compact-evidence object before reuse."""
    manifest = root / 'checksums.sha256'
    rows = manifest.read_text(encoding='utf-8').splitlines()
    if len(rows) != 196:
        raise ValueError('P4.5 checksum manifest no longer has 196 entries')
    for row in rows:
        expected, separator, relative = row.partition('  ./')
        path = root / relative
        if separator != '  ./' or not path.is_file() or p45.sha256(path) != expected:
            raise ValueError(f'P4.5 compact evidence failed its seal: {relative}')


def _unique_artifact(container: Path, name: str) -> Path:
    """Resolve one compact run artifact without relying on timestamp names."""
    matches = list(container.glob(f'**/artifacts/{name}'))
    if len(matches) != 1:
        raise ValueError(f'{container} contains {len(matches)} copies of {name}, expected one')
    return matches[0]


def _load_baseline(root: Path) -> dict[str, Any]:
    """Load immutable C0/C1 evidence after verifying its complete transfer seal."""
    _validate_checksum_manifest(root)
    summary_dir = root / 'summary'
    summary_path = summary_dir / 'p4_5_final_summary.json'
    final_rows_path = summary_dir / 'p4_5_final_seed_metrics.csv'
    if p45.sha256(summary_path) != P45_SUMMARY_SHA256:
        raise ValueError('P4.5 final summary identity changed')
    if p45.sha256(final_rows_path) != P45_FINAL_ROWS_SHA256:
        raise ValueError('P4.5 final seed table identity changed')
    summary = p45.read_json(summary_path)
    if (
        summary.get('protocol_sha256') != P45_PROTOCOL_SHA256
        or summary.get('run_count') != 6
        or summary.get('all_runs_completed_100k') is not True
    ):
        raise ValueError('P4.5 baseline summary violates its seal')

    evaluations: dict[tuple[str, int], dict[str, Any]] = {}
    trajectories: dict[tuple[str, int], str] = {}
    for arm in ('C0', 'C1'):
        for seed in SEEDS:
            container = root / 'runs' / f'{arm}-{seed}'
            receipt = p45.read_json(container / 'receipt.json')
            if (
                receipt.get('status') != 'passed'
                or int(receipt.get('steps', -1)) != TARGET_STEPS
                or receipt.get('source_dirty') is not False
            ):
                raise ValueError(f'sealed baseline {arm}-{seed} receipt is invalid')
            evaluation = p45.read_json(_unique_artifact(container, 'evaluation_step_100000.json'))
            if tuple(int(value) for value in evaluation['seeds']) != EVAL_SEEDS:
                raise ValueError(f'sealed baseline {arm}-{seed} changed evaluation seeds')
            evaluations[(arm, seed)] = evaluation
            trajectory = _unique_artifact(container, 'trajectory_step_100000.csv')
            trajectories[(arm, seed)] = str(trajectory)

    return {
        'summary': summary,
        'checkpoint_rows': p45._read_csv(  # pylint: disable=protected-access
            summary_dir / 'p4_5_checkpoint_metrics.csv',
        ),
        'final_rows': p45._read_csv(final_rows_path),  # pylint: disable=protected-access
        'diagnostics': p45._read_csv(  # pylint: disable=protected-access
            summary_dir / 'p4_5_diagnostics_combined.csv',
        ),
        'checkpoint_hashes': p45._read_csv(  # pylint: disable=protected-access
            summary_dir / 'p4_5_checkpoint_sha256.csv',
        ),
        'evaluations': evaluations,
        'trajectories': trajectories,
    }


def _validate_audit(path: Path, arm: str, seed: int) -> dict[str, Any]:
    """Validate one P4.6 fixed-size three-mode audit."""
    payload = p45.read_json(path)
    if (
        payload.get('stage') != 'P4.6-final-cost-geometry-audit'
        or payload.get('protocol_version') != PROTOCOL_VERSION
        or payload.get('protocol_sha256') != PROTOCOL_SHA256
    ):
        raise ValueError(f'{arm} seed {seed} audit has the wrong protocol seal')
    if payload.get('arm') != arm or int(payload.get('training_seed', -1)) != seed:
        raise ValueError(f'{arm} seed {seed} audit identity mismatch')
    if not math.isclose(float(payload.get('directional_loss_weight', -1.0)), ARM_WEIGHTS[arm]):
        raise ValueError(f'{arm} seed {seed} audit has the wrong weight')
    if (
        int(payload.get('env_steps', -1)) != TARGET_STEPS
        or int(payload.get('state_count', -1)) != 64
        or int(payload.get('action_samples_per_state', -1)) != 256
        or int(payload.get('audit_seed', -1)) != 4_500_007
    ):
        raise ValueError(f'{arm} seed {seed} audit changed its locked sample')
    if int(payload.get('valid_fixed_normal_state_count', 0)) <= 0:
        raise ValueError(f'{arm} seed {seed} audit has no valid Q-cost normals')
    if set(payload.get('modes', {})) != {'A', 'B', 'C'}:
        raise ValueError(f'{arm} seed {seed} audit mode set changed')
    for mode in ('A', 'B', 'C'):
        for key, value in payload['modes'][mode]['metrics'].items():
            p45.finite_number(value, f'{arm}.{seed}.audit.{mode}.{key}')
    return payload


def _validate_candidate_run(
    arm: str,
    seed: int,
    container: Path,
    audit_path: Path,
) -> dict[str, Any]:
    """Validate one clean-source 100k P4.6 run and collect its evidence."""
    run_dir = p45._resolve_run_dir(container)  # pylint: disable=protected-access
    summary = p45.read_json(run_dir / 'artifacts/training_summary.json')
    manifest = p45.read_json(run_dir / 'artifacts/run_manifest.json')
    receipt = p45.read_json(container / 'receipt.json')
    if summary.get('status') != 'completed' or receipt.get('status') != 'passed':
        raise ValueError(f'{arm} seed {seed} did not complete normally')
    if any(int(item.get('seed', -1)) != seed for item in (summary, manifest, receipt)):
        raise ValueError(f'{arm} seed {seed} identity mismatch')
    if int(summary['env_steps']) != TARGET_STEPS or int(receipt['steps']) != TARGET_STEPS:
        raise ValueError(f'{arm} seed {seed} did not finish exactly 100k steps')
    if receipt.get('source_dirty') is not False or not receipt.get('source_commit'):
        raise ValueError(f'{arm} seed {seed} was not trained from clean committed source')
    if not bool(summary.get('directional_loss_enabled')) or not math.isclose(
        float(summary.get('directional_loss_weight', -1.0)),
        ARM_WEIGHTS[arm],
    ):
        raise ValueError(f'{arm} seed {seed} has the wrong direction-loss weight')
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
    protocol = manifest['p4_5_unified_cost_geometry']
    if (
        protocol['protocol_version'] != PROTOCOL_VERSION
        or protocol['protocol_sha256'] != PROTOCOL_SHA256
        or bool(protocol['probability_critic_policy_access'])
        or not math.isclose(float(protocol['directional_loss_weight']), ARM_WEIGHTS[arm])
    ):
        raise ValueError(f'{arm} seed {seed} manifest violates the P4.6 seal')
    if tuple(int(value) for value in manifest['eval_milestones']) != EVAL_STEPS:
        raise ValueError(f'{arm} seed {seed} has wrong evaluation milestones')
    if tuple(int(value) for value in manifest['checkpoint_milestones']) != CHECKPOINT_STEPS:
        raise ValueError(f'{arm} seed {seed} has wrong checkpoint milestones')

    diagnostics = p45._read_csv(  # pylint: disable=protected-access
        run_dir / 'artifacts/diagnostics.csv',
    )
    diagnostic_steps = tuple(int(float(row['TotalEnvSteps'])) for row in diagnostics)
    if diagnostic_steps != tuple(range(5_000, TARGET_STEPS + 1, 5_000)):
        raise ValueError(f'{arm} seed {seed} lacks exact 5k diagnostic coverage')
    evaluations: dict[int, dict[str, Any]] = {}
    checkpoint_rows = []
    for milestone in EVAL_STEPS:
        payload = p45.read_json(run_dir / f'artifacts/evaluation_step_{milestone}.json')
        if tuple(int(value) for value in payload['seeds']) != EVAL_SEEDS:
            raise ValueError(f'{arm} seed {seed} step {milestone} used wrong eval seeds')
        if int(payload['episodes']) != 20 or int(payload['milestone']) != milestone:
            raise ValueError(f'{arm} seed {seed} step {milestone} evaluation mismatch')
        if tuple(int(value) for value in payload['trajectory_seeds']) != TRAJECTORY_SEEDS:
            raise ValueError(f'{arm} seed {seed} changed trajectory seeds')
        if len(payload.get('cost_risk_bins', [])) != 5:
            raise ValueError(f'{arm} seed {seed} step {milestone} lacks five risk bins')
        evaluations[milestone] = payload
        checkpoint_rows.append(
            p45._checkpoint_metrics(payload, arm, seed),  # pylint: disable=protected-access
        )
    trajectory = run_dir / f'artifacts/trajectory_step_{TARGET_STEPS}.csv'
    trajectory_rows = p45._read_csv(trajectory)  # pylint: disable=protected-access
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
                'sha256': p45.sha256(path),
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
            'sha256': p45.sha256(final_full),
        },
    )
    audit = _validate_audit(audit_path, arm, seed)
    if audit['checkpoint_sha256'] != checkpoint_hashes[-1]['sha256']:
        raise ValueError(f'{arm} seed {seed} audit checkpoint identity mismatch')
    return {
        'arm': arm,
        'training_seed': seed,
        'run_dir': str(run_dir),
        'receipt': str(container / 'receipt.json'),
        'source_commit': receipt['source_commit'],
        'evaluations': evaluations,
        'checkpoint_rows': checkpoint_rows,
        'diagnostics': diagnostics,
        'trajectory': str(trajectory),
        'checkpoint_hashes': checkpoint_hashes,
        'audit': audit,
    }


def _pooled(evaluations: dict[tuple[str, int], dict[str, Any]], arm: str) -> dict[str, float]:
    """Pool the 60 fixed final episodes for one weight."""
    payloads = [evaluations[(arm, seed)] for seed in SEEDS]
    returns = [float(value) for payload in payloads for value in payload['returns']]
    costs = [float(value) for payload in payloads for value in payload['costs']]
    goals = [int(value) for payload in payloads for value in payload['goal_counts']]
    return {
        'episodes': float(len(returns)),
        'mean_return': statistics.fmean(returns),
        'success_rate': sum(value > 0 for value in goals) / len(goals),
        'mean_cost': statistics.fmean(costs),
        'cost_cvar_20': p45._tail_mean(costs),  # pylint: disable=protected-access
        'dangerous_normal_velocity': statistics.fmean(
            float(payload['direction_geometry']['dangerous_normal_velocity'])
            for payload in payloads
        ),
        'mean_path_length': statistics.fmean(
            float(payload['mean_path_length']) for payload in payloads
        ),
        'stall_ratio': statistics.fmean(float(payload['stall_ratio']) for payload in payloads),
    }


def _value(rows: dict[tuple[str, int], dict[str, Any]], arm: str, seed: int, key: str) -> float:
    """Read one finite final seed metric from mixed CSV/native rows."""
    return p45.finite_number(rows[(arm, seed)][key], f'{arm}.{seed}.{key}')


def classify_tradeoff(
    final_rows: dict[tuple[str, int], dict[str, Any]],
    pooled: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Apply the locked P4.6 candidate gates and deterministic tie break."""
    gates: dict[str, Any] = {}
    qualified = []
    safety_passers = []
    for arm in CANDIDATES:
        safety_metrics = {}
        for field in ('mean_cost', 'cost_cvar_20'):
            lower_seeds = [
                seed
                for seed in SEEDS
                if _value(final_rows, arm, seed, field) < _value(final_rows, 'C0', seed, field)
            ]
            reduction = 1.0 - pooled[arm][field] / pooled['C0'][field]
            safety_metrics[field] = {
                'lower_seeds': lower_seeds,
                'paired_seed_count': len(lower_seeds),
                'pooled_reduction_fraction': reduction,
                'passed': len(lower_seeds) >= 2 and reduction >= 0.10,
            }
        safety = all(item['passed'] for item in safety_metrics.values())
        return_loss = pooled['C0']['mean_return'] - pooled[arm]['mean_return']
        success_loss = pooled['C0']['success_rate'] - pooled[arm]['success_rate']
        performance = return_loss <= 0.10 and success_loss <= 0.10
        path_ratio_2028 = _value(final_rows, arm, 2028, 'mean_path_length') / _value(
            final_rows,
            'C0',
            2028,
            'mean_path_length',
        )
        stall_increase_2029 = _value(final_rows, arm, 2029, 'stall_ratio') - _value(
            final_rows,
            'C0',
            2029,
            'stall_ratio',
        )
        robustness = path_ratio_2028 <= 1.5 and stall_increase_2029 <= 0.10
        candidate_qualified = safety and performance and robustness
        gates[arm] = {
            'safety': safety,
            'safety_metrics': safety_metrics,
            'return_loss_c0_minus_candidate': return_loss,
            'success_loss_c0_minus_candidate': success_loss,
            'performance': performance,
            'seed_2028_path_ratio_to_c0': path_ratio_2028,
            'seed_2029_stall_increase_over_c0': stall_increase_2029,
            'seed_robustness': robustness,
            'qualified': candidate_qualified,
        }
        if safety:
            safety_passers.append(arm)
        if candidate_qualified:
            qualified.append(arm)

    selected = None
    tie_break = 'not_applicable'
    if not safety_passers:
        label = 'no_stable_tradeoff'
    elif not qualified:
        label = 'partial_tradeoff_only'
    elif len(qualified) == 1:
        selected = qualified[0]
        label = f'select_{selected}'
        tie_break = 'only_qualified_candidate'
    else:
        c050_extra_safety = (
            pooled['C050']['mean_cost'] <= 0.90 * pooled['C025']['mean_cost']
            and pooled['C050']['cost_cvar_20'] <= 0.90 * pooled['C025']['cost_cvar_20']
        )
        selected = 'C050' if c050_extra_safety else 'C025'
        label = f'select_{selected}'
        tie_break = 'c050_extra_safety' if c050_extra_safety else 'lower_intervention_c025'
    return {
        'label': label,
        'selected_arm': selected,
        'selected_weight': None if selected is None else ARM_WEIGHTS[selected],
        'candidate_gates': gates,
        'qualified_candidates': qualified,
        'safety_passing_candidates': safety_passers,
        'tie_break': tie_break,
        'pooled': pooled,
    }


def _render_report(result: dict[str, Any]) -> str:
    """Render the complete four-weight result and locked selection decision."""
    decision = result['decision']
    lines = [
        '# STAR P4.6 direction-loss weight Pareto final report',
        '',
        f"Conclusion: **{decision['label']}**",
        '',
        f'Protocol: `{PROTOCOL_VERSION}` (`{PROTOCOL_SHA256}`)',
        '',
        (
            'C0 and C1 are checksum-verified P4.5 evidence reused without retraining. C025 and '
            'C050 are the six new 100k runs. Every final seed is shown below.'
        ),
        '',
        '| arm | lambda | seed | return | success | mean cost | CVaR20 | dangerous v_n+ | path | stall | clip |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in result['final_seed_metrics']:
        lines.append(
            f"| {row['arm']} | {ARM_WEIGHTS[row['arm']]:.2f} | {row['training_seed']} | "
            f"{float(row['mean_return']):.6f} | {float(row['success_rate']):.3f} | "
            f"{float(row['mean_cost']):.6f} | {float(row['cost_cvar_20']):.6f} | "
            f"{float(row['dangerous_normal_velocity']):.6f} | "
            f"{float(row['mean_path_length']):.6f} | {float(row['stall_ratio']):.3f} | "
            f"{float(row['action_clipping_rate']):.6f} |",
        )
    lines.extend(
        [
            '',
            '## Pooled 60-episode Pareto points',
            '',
            '| arm | lambda | return | success | mean cost | CVaR20 | dangerous v_n+ | path | stall | training cost |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ],
    )
    for arm in ARMS:
        row = decision['pooled'][arm]
        lines.append(
            f"| {arm} | {ARM_WEIGHTS[arm]:.2f} | {row['mean_return']:.6f} | "
            f"{row['success_rate']:.3f} | {row['mean_cost']:.6f} | "
            f"{row['cost_cvar_20']:.6f} | {row['dangerous_normal_velocity']:.6f} | "
            f"{row['mean_path_length']:.6f} | {row['stall_ratio']:.3f} | "
            f"{row['training_cumulative_cost']:.0f} |",
        )
    lines.extend(['', '## Locked gates', ''])
    for arm in CANDIDATES:
        gate = decision['candidate_gates'][arm]
        lines.extend(
            [
                f'### {arm} (`lambda={ARM_WEIGHTS[arm]:.2f}`)',
                '',
                f"- Safety gate: {gate['safety']}.",
                (
                    f"- Performance gate: {gate['performance']} (return loss "
                    f"{gate['return_loss_c0_minus_candidate']:+.6f}, success loss "
                    f"{gate['success_loss_c0_minus_candidate']:+.6f})."
                ),
                (
                    f"- Seed robustness gate: {gate['seed_robustness']} (2028 path/C0 "
                    f"{gate['seed_2028_path_ratio_to_c0']:.3f}, 2029 stall increase "
                    f"{gate['seed_2029_stall_increase_over_c0']:+.3f})."
                ),
                f"- Fully qualified: {gate['qualified']}.",
                '',
            ],
        )
    lines.extend(
        [
            (
                f"Selected arm: `{decision['selected_arm']}`; selected weight: "
                f"`{decision['selected_weight']}`; tie break: `{decision['tie_break']}`."
            ),
            '',
            '## Evidence and figures',
            '',
            (
                'All new final full-replay checkpoints received the unchanged 64 x 256 A/B/C '
                'audit. The combined CSVs retain all checkpoints, 5k diagnostics, fixed-seed '
                'evaluations, checkpoint hashes, and trajectory paths.'
            ),
            '',
            '- `p4_6_return_success.png`',
            '- `p4_6_cost_cvar_training_cost.png`',
            '- `p4_6_geometry.png`',
            '- `p4_6_critic_scales.png`',
            '- `p4_6_final_pareto.png`',
            '- `p4_6_trajectory_eval_seed_12027.png`',
            '- `p4_6_trajectory_eval_seed_12036.png`',
            '- `p4_6_trajectory_eval_seed_12046.png`',
            '',
            (
                'Large checkpoints and replay remain server-only. '
                '`p4_6_checkpoint_sha256.csv` records their immutable identities.'
            ),
            '',
        ],
    )
    return '\n'.join(lines)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    """Validate all reused/new evidence and write the four-weight summary."""
    if p45.sha256(args.protocol) != PROTOCOL_SHA256:
        raise ValueError('P4.6 protocol hash does not match its preregistered seal')
    protocol = p45.read_json(args.protocol)
    if protocol.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('P4.6 protocol version mismatch')
    baseline = _load_baseline(args.baseline_evidence_root)
    run_paths = _bindings(args.runs, 'run')
    audit_paths = _bindings(args.audits, 'audit')
    candidates = {
        key: _validate_candidate_run(*key, run_paths[key], audit_paths[key])
        for key in sorted(run_paths)
    }
    source_commits = {run['source_commit'] for run in candidates.values()}
    if len(source_commits) != 1:
        raise ValueError(f'P4.6 candidates used multiple source commits: {sorted(source_commits)}')

    checkpoint_rows: list[dict[str, Any]] = list(baseline['checkpoint_rows'])
    diagnostic_rows: list[dict[str, Any]] = list(baseline['diagnostics'])
    checkpoint_hashes: list[dict[str, Any]] = list(baseline['checkpoint_hashes'])
    evaluations: dict[tuple[str, int], dict[str, Any]] = dict(baseline['evaluations'])
    trajectories = dict(baseline['trajectories'])
    audits = dict(baseline['summary']['audits'])
    for (arm, seed), run in sorted(candidates.items()):
        checkpoint_rows.extend(run['checkpoint_rows'])
        diagnostic_rows.extend(
            {'arm': arm, 'training_seed': seed, **row} for row in run['diagnostics']
        )
        checkpoint_hashes.extend(run['checkpoint_hashes'])
        evaluations[(arm, seed)] = run['evaluations'][TARGET_STEPS]
        trajectories[(arm, seed)] = run['trajectory']
        audits[f'{arm}:{seed}'] = run['audit']

    checkpoint_rows.sort(
        key=lambda row: (
            ARMS.index(str(row['arm'])),
            int(row['training_seed']),
            int(row['milestone']),
        ),
    )
    final_rows_list = [row for row in checkpoint_rows if int(row['milestone']) == TARGET_STEPS]
    final_rows = {(str(row['arm']), int(row['training_seed'])): row for row in final_rows_list}
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(final_rows) != expected:
        raise ValueError('four-weight final seed table is incomplete')
    pooled = {arm: _pooled(evaluations, arm) for arm in ARMS}
    for arm in ARMS:
        pooled[arm]['training_cumulative_cost'] = sum(
            _value(final_rows, arm, seed, 'training_cumulative_cost') for seed in SEEDS
        )
    decision = classify_tradeoff(final_rows, pooled)

    deltas = []
    for arm in ('C025', 'C050', 'C1'):
        for seed in SEEDS:
            deltas.append(
                {
                    'arm': arm,
                    'directional_loss_weight': ARM_WEIGHTS[arm],
                    'training_seed': seed,
                    **{
                        f'arm_minus_c0_{field}': _value(final_rows, arm, seed, field)
                        - _value(final_rows, 'C0', seed, field)
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
    trajectory_manifest = [
        {
            'arm': arm,
            'directional_loss_weight': ARM_WEIGHTS[arm],
            'training_seed': seed,
            'milestone': TARGET_STEPS,
            'path': trajectories[(arm, seed)],
        }
        for arm in ARMS
        for seed in SEEDS
    ]
    result = {
        'stage': 'P4.6-final-direction-weight-pareto-summary',
        'protocol_version': PROTOCOL_VERSION,
        'protocol_sha256': PROTOCOL_SHA256,
        'base_commit': protocol['base_commit'],
        'p4_5_source_commit': baseline['summary']['source_commit'],
        'p4_6_source_commit': next(iter(source_commits)),
        'reused_run_count': 6,
        'new_run_count': 6,
        'all_new_runs_completed_100k': True,
        'scientific_early_stops': 0,
        'arm_weights': ARM_WEIGHTS,
        'final_seed_metrics': final_rows_list,
        'paired_deltas_from_c0': deltas,
        'decision': decision,
        'audits': audits,
        'candidate_run_paths': {
            f'{arm}:{seed}': run['run_dir'] for (arm, seed), run in candidates.items()
        },
    }
    output = args.output_dir
    p45.atomic_csv(output / 'p4_6_checkpoint_metrics.csv', checkpoint_rows)
    p45.atomic_csv(output / 'p4_6_final_seed_metrics.csv', final_rows_list)
    p45.atomic_csv(output / 'p4_6_paired_deltas_from_c0.csv', deltas)
    p45.atomic_csv(output / 'p4_6_diagnostics_combined.csv', diagnostic_rows)
    p45.atomic_csv(output / 'p4_6_checkpoint_sha256.csv', checkpoint_hashes)
    p45.atomic_csv(output / 'p4_6_trajectory_manifest.csv', trajectory_manifest)
    p45.atomic_json(output / 'p4_6_final_summary.json', result)
    p45.atomic_text(output / 'P4_6_DIRECTION_WEIGHT_PARETO_REPORT.md', _render_report(result))
    return result


def parse_args() -> argparse.Namespace:
    """Parse baseline and explicit six-run/six-audit bindings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--baseline-evidence-root', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--audits', nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Validate the sweep and print its locked final decision."""
    args = parse_args()
    result = summarize(args)
    print(
        json.dumps(
            {
                'decision': result['decision']['label'],
                'selected_weight': result['decision']['selected_weight'],
                'output_dir': str(args.output_dir),
            },
        ),
    )


if __name__ == '__main__':
    main()
