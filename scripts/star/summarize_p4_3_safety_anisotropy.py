#!/usr/bin/env python3
"""Apply the locked P4.3 S0--S2 stochastic safety-reference decision."""

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


ARMS = ('S0', 'S1', 'S2')
SEEDS = (2027, 2028, 2029)
TARGET_STEPS = 50_000
PROTOCOL_VERSION = 'p4.3-safety-anisotropy-v1'
PROTOCOL_SHA256 = '271b65a6c1869de9d15cfc0037c5b0e61f7bf3864d022769dcc9ae256046e052'
EVAL_SEEDS = tuple(range(12_027, 12_047))
EXPECTED_FLAGS = {
    'S0': (True, False, False, False),
    'S1': (True, True, False, False),
    'S2': (True, True, True, False),
}


def read_json(path: Path) -> dict[str, Any]:
    """Read one strict JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a JSON object')
    return payload


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    """Write one report atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def finite_number(value: Any, location: str) -> float:
    """Return a finite float or reject the evidence artifact."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{location} is not numeric: {value!r}') from error
    if not math.isfinite(result):
        raise ValueError(f'{location} is non-finite: {value!r}')
    return result


def bindings(values: list[str], label: str) -> dict[tuple[str, int], Path]:
    """Parse exactly nine ``ARM:SEED=PATH`` bindings."""
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    result: dict[tuple[str, int], Path] = {}
    for value in values:
        identity, separator, raw_path = value.partition('=')
        arm, colon, raw_seed = identity.partition(':')
        if separator != '=' or colon != ':' or not raw_path:
            raise ValueError(f'{label} bindings must use ARM:SEED=PATH')
        key = (arm, int(raw_seed))
        if key not in expected or key in result:
            raise ValueError(f'{label} has unexpected/duplicate identity {key}')
        result[key] = Path(raw_path)
    if set(result) != expected:
        raise ValueError(f'{label} must bind S0--S2 at seeds {SEEDS}')
    return result


def read_diagnostics(path: Path, arm: str, seed: int) -> dict[str, Any]:
    """Validate 5k coverage, finite core training data, trace, and reward-TD trend."""
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10:
        raise ValueError(f'{arm} seed {seed} must have ten diagnostic rows')
    steps = [int(finite_number(row['TotalEnvSteps'], 'TotalEnvSteps')) for row in rows]
    if tuple(steps) != tuple(range(5_000, 50_001, 5_000)):
        raise ValueError(f'{arm} seed {seed} has incorrect diagnostic coverage')
    core = (
        'Critic/CostTD',
        'Critic/RewardTD',
        'Loss/CostCritic',
        'Loss/RewardCritic',
        'Dual/GlobalSafety',
        'Time/FPS',
    )
    for row_index, row in enumerate(rows):
        for key in core:
            finite_number(row[key], f'{arm}.{seed}.diagnostics[{row_index}].{key}')
    actor_rows = [row for row in rows if int(float(row['TotalEnvSteps'])) >= 10_000]
    trace_errors = [
        finite_number(row['Reference/TraceRelativeError'], 'Reference/TraceRelativeError')
        for row in actor_rows
    ]
    reward_td = [finite_number(row['Critic/RewardTD'], 'Critic/RewardTD') for row in rows]
    historical_minimum = min(reward_td)
    final_ratio = reward_td[-1] / historical_minimum
    rising_tail = all(after > before for before, after in zip(reward_td[-4:], reward_td[-3:]))
    return {
        'rows': len(rows),
        'maximum_reference_trace_relative_error': max(trace_errors),
        'reward_td': {
            'minimum': historical_minimum,
            'final': reward_td[-1],
            'final_to_minimum_ratio': final_ratio,
            'last_four': reward_td[-4:],
            'strictly_rising_last_three_intervals': rising_tail,
            'monotonic_divergence': rising_tail and final_ratio > 1.5,
        },
    }


def validate_run(
    arm: str,
    seed: int,
    run_dir: Path,
    receipt_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Validate one immutable 50k training run and return compact evidence."""
    summary = read_json(run_dir / 'artifacts/training_summary.json')
    manifest = read_json(run_dir / 'artifacts/run_manifest.json')
    receipt = read_json(receipt_path)
    if summary.get('status') != 'completed' or receipt.get('status') != 'passed':
        raise ValueError(f'{arm} seed {seed} did not complete')
    if (
        int(summary['seed']) != seed
        or int(receipt['seed']) != seed
        or int(manifest['seed']) != seed
    ):
        raise ValueError(f'{arm} seed {seed} identity mismatch')
    if int(summary['env_steps']) != TARGET_STEPS or int(receipt['steps']) != TARGET_STEPS:
        raise ValueError(f'{arm} seed {seed} did not stop at 50k')
    if int(summary['updates']) != 48_000 or int(receipt['updates']) != 48_000:
        raise ValueError(f'{arm} seed {seed} has the wrong update count')
    if receipt.get('source_dirty') is not False or not receipt.get('source_commit'):
        raise ValueError(f'{arm} seed {seed} was not trained from clean committed source')
    stochastic, anisotropy, drift, probing = EXPECTED_FLAGS[arm]
    actual_flags = (
        bool(summary['stochastic_flow_enabled']),
        bool(summary['safety_anisotropy_enabled']),
        bool(summary['safety_inward_drift_enabled']),
        bool(summary['candidate_probing_enabled']),
    )
    if actual_flags != EXPECTED_FLAGS[arm]:
        raise ValueError(f'{arm} seed {seed} has flags {actual_flags}')
    reference = manifest['p4_3_stochastic_safety_reference']
    if (
        reference['protocol_version'] != PROTOCOL_VERSION
        or reference['protocol_sha256'] != PROTOCOL_SHA256
    ):
        raise ValueError(f'{arm} seed {seed} has the wrong protocol seal')
    if (
        bool(reference['stochastic_flow_enabled']) != stochastic
        or bool(reference['safety_anisotropy_enabled']) != anisotropy
        or bool(reference['safety_inward_drift_enabled']) != drift
        or bool(reference['candidate_probing_enabled']) != probing
    ):
        raise ValueError(f'{arm} seed {seed} manifest flags disagree with summary')
    backbone = protocol['backbone']
    summary_names = {
        'enable_global_safety_constraint': 'global_safety_constraint_enabled',
        'use_target_flow_for_reward_backup': 'reward_target_actor_enabled',
    }
    for key, expected in backbone.items():
        actual_key = summary_names.get(key, key)
        actual = summary.get(actual_key)
        if actual is None:
            resolved = receipt['training_summary'].get(actual_key)
            actual = resolved
        if isinstance(expected, float):
            if not math.isclose(finite_number(actual, key), expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f'{arm} seed {seed} changed backbone setting {key}')
        elif actual != expected:
            raise ValueError(f'{arm} seed {seed} changed backbone setting {key}')
    checkpoint = run_dir / 'checkpoints/final_full.pt'
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f'{arm} seed {seed} lacks final_full.pt')
    diagnostics = read_diagnostics(run_dir / 'artifacts/diagnostics.csv', arm, seed)
    return {
        'source_commit': receipt['source_commit'],
        'receipt': str(receipt_path),
        'run_dir': str(run_dir),
        'checkpoint': str(checkpoint),
        'checkpoint_bytes': checkpoint.stat().st_size,
        'diagnostics': diagnostics,
        'final_calibration_reliability': finite_number(
            summary['calibration_reliability'],
            'calibration_reliability',
        ),
    }


def validate_evaluation(path: Path, arm: str, seed: int) -> dict[str, float]:
    """Validate one formal external stochastic-policy evaluation."""
    payload = read_json(path)
    if int(payload['training_seed']) != seed or int(payload['milestone']) != TARGET_STEPS:
        raise ValueError(f'{arm} seed {seed} evaluation identity mismatch')
    if tuple(int(value) for value in payload['seeds']) != EVAL_SEEDS:
        raise ValueError(f'{arm} seed {seed} used the wrong external seeds')
    if payload['evaluation_mode'] != 'fixed-random-flow':
        raise ValueError(f'{arm} seed {seed} evaluation is not the primary policy')
    if 'standard normal' not in payload.get('wiener_increment_policy', ''):
        raise ValueError(f'{arm} seed {seed} evaluation omitted stochastic-flow noise')
    return {
        key: finite_number(payload[key], f'{arm}.{seed}.evaluation.{key}')
        for key in (
            'mean_return',
            'mean_cost',
            'cost_cvar_20',
            'goal_met_episode_rate',
            'stall_ratio',
        )
    }


def validate_audit(path: Path, arm: str, seed: int) -> dict[str, Any]:
    """Validate the fixed-state M=256 held-out directional audit."""
    payload = read_json(path)
    if (
        payload.get('stage') != 'P4.3-safety-anisotropy-heldout-audit'
        or payload.get('protocol_version') != PROTOCOL_VERSION
        or payload.get('protocol_sha256') != PROTOCOL_SHA256
    ):
        raise ValueError(f'{arm} seed {seed} audit has the wrong protocol')
    if int(payload['training_seed']) != seed or int(payload['env_steps']) != TARGET_STEPS:
        raise ValueError(f'{arm} seed {seed} audit identity mismatch')
    if int(payload['state_count']) != 64 or int(payload['action_samples_per_state']) != 256:
        raise ValueError(f'{arm} seed {seed} changed the fixed-state audit size')
    flags = payload['flags']
    actual_flags = (
        bool(flags['stochastic_flow_enabled']),
        bool(flags['safety_anisotropy_enabled']),
        bool(flags['safety_inward_drift_enabled']),
        bool(flags['candidate_probing_enabled']),
    )
    if actual_flags != EXPECTED_FLAGS[arm]:
        raise ValueError(f'{arm} seed {seed} audit flags mismatch')
    if int(payload['valid_normal_state_count']) <= 0:
        raise ValueError(f'{arm} seed {seed} audit has no valid normals')
    trace_error = finite_number(
        payload['maximum_reference_trace_relative_error'],
        'maximum_reference_trace_relative_error',
    )
    compact: dict[str, Any] = {
        'path': str(path),
        'in_band_state_count': int(payload['selection']['in_band_state_count']),
        'valid_normal_state_count': int(payload['valid_normal_state_count']),
        'maximum_reference_trace_relative_error': trace_error,
        'heldout_auroc': payload['heldout_evidence'].get('Heldout/AUROC'),
        'heldout_brier_skill': payload['heldout_evidence'].get('Heldout/BrierSkill'),
    }
    for section in ('local_reference', 'full_policy', 'conditional_wiener'):
        compact[section] = {
            key: finite_number(value, f'{arm}.{seed}.{section}.{key}')
            for key, value in payload[section].items()
        }
    return compact


def pooled(runs: dict[int, dict[str, Any]], section: str) -> dict[str, float]:
    """Average one same-named numeric section across three training seeds."""
    keys = runs[SEEDS[0]][section]
    return {
        key: statistics.fmean(float(runs[seed][section][key]) for seed in SEEDS) for key in keys
    }


def render_report(result: dict[str, Any]) -> str:
    """Render a compact human-readable causal decision."""
    lines = [
        '# STAR P4.3 safety-conditioned stochastic-flow decision',
        '',
        f"Decision: **{result['decision']}**",
        '',
        (
            'All nine preregistered SafetyPointGoal1-v0 runs stopped at 50k. S0, S1, and S2 '
            'use one policy and the same Bellman probability critic; candidate probing is absent.'
        ),
        '',
        '| arm | return | cost | CVaR-20 | success | stall | V_N (conditional) | V_T (conditional) | crossing |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for arm in ARMS:
        task = result['arms'][arm]['pooled_external']
        audit = result['arms'][arm]['pooled_conditional_wiener']
        lines.append(
            f"| {arm} | {task['mean_return']:.3f} | {task['mean_cost']:.3f} | "
            f"{task['cost_cvar_20']:.3f} | {task['goal_met_episode_rate']:.3f} | "
            f"{task['stall_ratio']:.3f} | {audit['normal_variance']:.6f} | "
            f"{audit['tangent_variance']:.6f} | {audit['crossing_rate']:.3f} |",
        )
    lines.extend(
        [
            '',
            '## Frozen gates',
            '',
            *[
                f"- {'PASS' if value else 'FAIL'} — {name}"
                for name, value in result['criteria'].items()
            ],
            '',
            (
                'This is a three-seed directional mechanism screen, not a global safety '
                'guarantee, a significance claim, or automatic authorization to enter P5.'
            ),
            '',
        ],
    )
    return '\n'.join(lines)


def summarize(args: argparse.Namespace) -> dict[str, Any]:  # pylint: disable=too-many-locals
    """Validate all evidence and apply the immutable causal gates."""
    if sha256(args.protocol) != PROTOCOL_SHA256:
        raise ValueError('P4.3 protocol hash does not match its pre-training seal')
    protocol = read_json(args.protocol)
    if protocol.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('P4.3 protocol version mismatch')
    run_paths = bindings(args.runs, 'run')
    receipts = bindings(args.receipts, 'receipt')
    evaluations = bindings(args.evaluations, 'evaluation')
    audits = bindings(args.audits, 'audit')
    arm_results: dict[str, Any] = {}
    source_commits: set[str] = set()
    reward_divergence = False
    trace_limit = float(protocol['decision_thresholds']['reference_trace_relative_error_maximum'])
    for arm in ARMS:
        seed_results: dict[int, dict[str, Any]] = {}
        for seed in SEEDS:
            identity = (arm, seed)
            run = validate_run(
                arm,
                seed,
                run_paths[identity],
                receipts[identity],
                protocol,
            )
            evaluation = validate_evaluation(evaluations[identity], arm, seed)
            audit = validate_audit(audits[identity], arm, seed)
            source_commits.add(run['source_commit'])
            if run['diagnostics']['reward_td']['monotonic_divergence']:
                reward_divergence = True
            if (
                run['diagnostics']['maximum_reference_trace_relative_error'] > trace_limit
                or audit['maximum_reference_trace_relative_error'] > trace_limit
            ):
                raise ValueError(f'{arm} seed {seed} violated the fixed trace budget')
            seed_results[seed] = {**run, 'external': evaluation, 'audit': audit}
        pooled_external = pooled(seed_results, 'external')
        audit_by_seed = {seed: seed_results[seed]['audit'] for seed in SEEDS}
        pooled_conditional = pooled(audit_by_seed, 'conditional_wiener')
        pooled_full = pooled(audit_by_seed, 'full_policy')
        pooled_reference = pooled(audit_by_seed, 'local_reference')
        arm_results[arm] = {
            'runs': seed_results,
            'pooled_external': pooled_external,
            'pooled_conditional_wiener': pooled_conditional,
            'pooled_full_policy': pooled_full,
            'pooled_local_reference': pooled_reference,
        }
    if len(source_commits) != 1:
        raise ValueError(f'P4.3 runs used multiple source commits: {sorted(source_commits)}')
    s0 = arm_results['S0']
    s1 = arm_results['S1']
    s2 = arm_results['S2']
    threshold = protocol['decision_thresholds']
    s0_cond = s0['pooled_conditional_wiener']
    s1_cond = s1['pooled_conditional_wiener']
    s2_cond = s2['pooled_conditional_wiener']
    s0_task = s0['pooled_external']
    s1_task = s1['pooled_external']
    s2_task = s2['pooled_external']

    def critic_good(arm: str) -> int:
        count = 0
        for seed in SEEDS:
            audit = arm_results[arm]['runs'][seed]['audit']
            try:
                auroc = finite_number(audit['heldout_auroc'], 'heldout_auroc')
                brier_skill = finite_number(
                    audit['heldout_brier_skill'],
                    'heldout_brier_skill',
                )
            except ValueError:
                continue
            count += int(
                auroc >= float(threshold['critic_heldout_auroc_minimum']) and brier_skill > 0.0,
            )
        return count

    boundary_evidence = all(
        arm_results[arm]['runs'][seed]['audit']['in_band_state_count']
        >= int(threshold['boundary_states_in_band_per_seed_minimum'])
        for arm in ('S1', 'S2')
        for seed in SEEDS
    )
    crossing_reduced = (
        s1_cond['crossing_rate']
        <= s0_cond['crossing_rate'] - float(threshold['s1_crossing_absolute_reduction_minimum'])
        or s1_cond['crossing_rate']
        <= float(threshold['s1_crossing_ratio_maximum']) * s0_cond['crossing_rate']
    )
    criteria = {
        'no_reward_td_divergence': not reward_divergence,
        'boundary_evidence_every_s1_s2_seed': boundary_evidence,
        'critic_evidence_s1': critic_good('S1')
        >= int(threshold['critic_good_seeds_per_arm_minimum']),
        'critic_evidence_s2': critic_good('S2')
        >= int(threshold['critic_good_seeds_per_arm_minimum']),
        's1_normal_variance_reduced': s1_cond['normal_variance']
        <= float(threshold['s1_normal_variance_ratio_maximum']) * s0_cond['normal_variance'],
        's1_tangent_variance_preserved': s1_cond['tangent_variance']
        >= float(threshold['s1_tangent_variance_ratio_minimum']) * s0_cond['tangent_variance'],
        's1_tangent_entropy_increased': s1_cond['tangent_entropy'] > s0_cond['tangent_entropy'],
        's1_crossing_reduced': crossing_reduced,
        'anisotropic_entropy_exceeds_matched_isotropic_low': s1['pooled_local_reference'][
            'entropy_advantage_vs_matched_normal_isotropic_low'
        ]
        > 0.0,
        's1_functional_success': s1_task['goal_met_episode_rate']
        >= s0_task['goal_met_episode_rate']
        - float(threshold['goal_met_episode_rate_loss_maximum']),
        's1_functional_stall': s1_task['stall_ratio']
        <= s0_task['stall_ratio'] + float(threshold['stall_ratio_increase_maximum']),
        's2_functional_success': s2_task['goal_met_episode_rate']
        >= s0_task['goal_met_episode_rate']
        - float(threshold['goal_met_episode_rate_loss_maximum']),
        's2_functional_stall': s2_task['stall_ratio']
        <= s0_task['stall_ratio'] + float(threshold['stall_ratio_increase_maximum']),
        's2_more_inward_than_s1': s2_cond['mean_normal_displacement']
        < s1_cond['mean_normal_displacement'],
        's2_crossing_no_higher_than_s1': s2_cond['crossing_rate'] <= s1_cond['crossing_rate'],
        's2_mean_cost_no_higher_than_s1': s2_task['mean_cost'] <= s1_task['mean_cost'],
        's2_cvar_no_higher_than_s1': s2_task['cost_cvar_20'] <= s1_task['cost_cvar_20'],
        's2_success_within_0_10_of_s1': s2_task['goal_met_episode_rate']
        >= s1_task['goal_met_episode_rate']
        - float(threshold['goal_met_episode_rate_loss_maximum']),
        's2_stall_within_0_10_of_s1': s2_task['stall_ratio']
        <= s1_task['stall_ratio'] + float(threshold['stall_ratio_increase_maximum']),
    }
    covariance_keys = (
        'boundary_evidence_every_s1_s2_seed',
        'critic_evidence_s1',
        'critic_evidence_s2',
        's1_normal_variance_reduced',
        's1_tangent_variance_preserved',
        's1_tangent_entropy_increased',
        's1_crossing_reduced',
        'anisotropic_entropy_exceeds_matched_isotropic_low',
    )
    function_keys = (
        's1_functional_success',
        's1_functional_stall',
        's2_functional_success',
        's2_functional_stall',
    )
    drift_keys = (
        's2_more_inward_than_s1',
        's2_crossing_no_higher_than_s1',
        's2_mean_cost_no_higher_than_s1',
        's2_cvar_no_higher_than_s1',
        's2_success_within_0_10_of_s1',
        's2_stall_within_0_10_of_s1',
    )
    if not criteria['no_reward_td_divergence']:
        decision = 'hard_stop_reward_td_divergence'
    elif not all(criteria[key] for key in covariance_keys):
        decision = 'reject_safety_anisotropy_mechanism'
    elif not all(criteria[key] for key in function_keys):
        decision = 'hard_stop_task_functionality'
    elif not all(criteria[key] for key in drift_keys):
        decision = 'retain_s1_covariance_reject_s2_drift'
    else:
        decision = 's0_s1_s2_mechanism_chain_passed'
    result = {
        'stage': 'P4.3-safety-conditioned-stochastic-flow-screen',
        'protocol_version': PROTOCOL_VERSION,
        'protocol_sha256': PROTOCOL_SHA256,
        'training_source_commit': next(iter(source_commits)),
        'seeds': list(SEEDS),
        'target_steps': TARGET_STEPS,
        'external_evaluation_seeds': list(EVAL_SEEDS),
        'arms': arm_results,
        'criteria': criteria,
        'decision': decision,
        'final_p5_verdict': None,
        'scope': 'directional three-seed mechanism screen; no significance/global-safety claim',
    }
    atomic_json(args.output, result)
    atomic_text(args.report, render_report(result))
    return result


def parse_args() -> argparse.Namespace:
    """Parse all explicit evidence bindings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--receipts', nargs='+', required=True)
    parser.add_argument('--evaluations', nargs='+', required=True)
    parser.add_argument('--audits', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Apply and print the P4.3 decision."""
    result = summarize(parse_args())
    print(json.dumps({'decision': result['decision'], 'output': result['stage']}))


if __name__ == '__main__':
    main()
