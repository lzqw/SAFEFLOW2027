#!/usr/bin/env python3
"""Validate the two seed-2028 P4.4 pilot arms and authorize or stop expansion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 'p4.4-outward-rectification-v1'
PROTOCOL_SHA256 = '005c42348a22f664448c688b3cecc16eba7cbbea55d7727537caeef5d3b69457'
ARMS = ('R0', 'R1')
SEED = 2028
TARGET_STEPS = 30_000
EVAL_SEEDS = tuple(range(12_027, 12_047))
EXPECTED_RECTIFICATION = {'R0': False, 'R1': True}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    with path.open('r', encoding='utf-8') as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f'{path} does not contain a JSON object')
    return payload


def atomic_text(path: Path, value: str) -> None:
    """Atomically write a UTF-8 text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write strict, sorted JSON."""
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def finite(value: Any, location: str) -> float:
    """Return one finite scalar or reject the evidence."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{location} is not numeric') from error
    if not math.isfinite(number):
        raise ValueError(f'{location} is not finite')
    return number


def bindings(values: list[str], label: str) -> dict[str, Path]:
    """Parse exact ``ARM=PATH`` bindings for R0 and R1."""
    result: dict[str, Path] = {}
    for value in values:
        arm, separator, raw_path = value.partition('=')
        if not separator or arm not in ARMS:
            raise ValueError(f'invalid {label} binding {value!r}')
        if arm in result:
            raise ValueError(f'duplicate {label} binding for {arm}')
        result[arm] = Path(raw_path)
    if set(result) != set(ARMS):
        raise ValueError(f'{label} bindings must contain exactly R0 and R1')
    return result


def ols_slope(values: list[float]) -> float:
    """Return the equally spaced ordinary-least-squares slope."""
    center = 0.5 * (len(values) - 1)
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    return (
        sum((index - center) * value for index, value in enumerate(values)) / denominator
        if denominator
        else 0.0
    )


def validate_diagnostics(path: Path, arm: str, protocol: dict[str, Any]) -> dict[str, Any]:
    """Validate six exact 5k rows and independently reapply every numerical stop."""
    with path.open('r', encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 6:
        raise ValueError(f'{arm} pilot must have six 5k diagnostic rows')
    steps = [int(finite(row['TotalEnvSteps'], f'{arm}.TotalEnvSteps')) for row in rows]
    if steps != list(range(5_000, TARGET_STEPS + 1, 5_000)):
        raise ValueError(f'{arm} pilot diagnostic coverage is not 5k--30k')
    required = (
        'Critic/CostTD',
        'Critic/RewardTD',
        'Critic/RewardQAbsMax',
        'Critic/RewardTargetQAbsMax',
        'Numerics/CostGradientNorm',
        'Numerics/ViolationGradientNorm',
        'Numerics/RewardGradientNorm',
        'Numerics/ActorGradientNorm',
        'Reference/RawDangerousNormalVelocity',
        'Reference/DangerousNormalVelocity',
        'Reference/PreClippingVariance',
        'Reference/PostClippingVariance',
    )
    for row_index, row in enumerate(rows):
        for key in required:
            finite(row[key], f'{arm}.diagnostics[{row_index}].{key}')
    reward_td = [finite(row['Critic/RewardTD'], f'{arm}.RewardTD') for row in rows]
    reward_q = [finite(row['Critic/RewardQAbsMax'], f'{arm}.RewardQAbsMax') for row in rows]
    reward_target = [
        finite(row['Critic/RewardTargetQAbsMax'], f'{arm}.RewardTargetQAbsMax') for row in rows
    ]
    thresholds = protocol['reward_numerical_hard_stops']
    minimum = min(reward_td)
    ratio = reward_td[-1] / max(minimum, 1e-12)
    slope = ols_slope(reward_td[-4:])
    hard_stop_reasons = []
    if max(abs(value) for value in reward_td) > float(thresholds['reward_td_absolute_maximum']):
        hard_stop_reasons.append('reward_td_absolute_threshold_exceeded')
    if max(reward_q) > float(thresholds['reward_q_absolute_maximum']):
        hard_stop_reasons.append('online_reward_q_absolute_threshold_exceeded')
    if max(reward_target) > float(thresholds['reward_target_q_absolute_maximum']):
        hard_stop_reasons.append('reward_target_q_absolute_threshold_exceeded')
    if ratio > float(thresholds['final_to_historical_minimum_ratio']) and slope > 0.0:
        hard_stop_reasons.append('reward_td_ratio_and_positive_last_four_slope')
    return {
        'path': str(path),
        'rows': len(rows),
        'reward_td': reward_td,
        'historical_minimum': minimum,
        'final_to_historical_minimum_ratio': ratio,
        'last_four_ols_slope': slope,
        'maximum_online_reward_q_abs': max(reward_q),
        'maximum_reward_target_q_abs': max(reward_target),
        'hard_stop_reasons': hard_stop_reasons,
        'passed': not hard_stop_reasons,
    }


def validate_run(
    arm: str,
    run_dir: Path,
    receipt_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Validate one clean committed seed-2028 30k training arm."""
    summary = read_json(run_dir / 'artifacts/training_summary.json')
    manifest = read_json(run_dir / 'artifacts/run_manifest.json')
    receipt = read_json(receipt_path)
    if summary.get('status') != 'completed' or receipt.get('status') != 'passed':
        raise ValueError(f'{arm} pilot did not complete')
    if any(int(item['seed']) != SEED for item in (summary, manifest, receipt)):
        raise ValueError(f'{arm} pilot seed identity mismatch')
    if int(summary['env_steps']) != TARGET_STEPS or int(receipt['steps']) != TARGET_STEPS:
        raise ValueError(f'{arm} pilot did not stop at exactly 30k')
    if int(summary['run_env_steps']) != TARGET_STEPS or int(receipt['run_steps']) != TARGET_STEPS:
        raise ValueError(f'{arm} pilot was not trained from zero')
    if int(summary['updates']) != 28_000 or int(receipt['updates']) != 28_000:
        raise ValueError(f'{arm} pilot has the wrong UTD update count')
    if receipt.get('source_dirty') is not False or not receipt.get('source_commit'):
        raise ValueError(f'{arm} pilot source was not clean and committed')
    expected_rectification = EXPECTED_RECTIFICATION[arm]
    flags = (
        bool(summary['stochastic_flow_enabled']),
        bool(summary['safety_anisotropy_enabled']),
        bool(summary['safety_inward_drift_enabled']),
        bool(summary['safety_outward_rectification_enabled']),
        bool(summary['complete_heldout_evidence']),
        bool(summary['candidate_probing_enabled']),
    )
    expected = (True, True, False, expected_rectification, True, False)
    if flags != expected:
        raise ValueError(f'{arm} pilot flags are {flags}, expected {expected}')
    p4_4 = manifest['p4_4_outward_rectification']
    if (
        p4_4['protocol_version'] != PROTOCOL_VERSION
        or p4_4['protocol_sha256'] != PROTOCOL_SHA256
        or bool(p4_4['outward_rectification_enabled']) != expected_rectification
    ):
        raise ValueError(f'{arm} pilot manifest has the wrong P4.4 seal or flag')
    if summary.get('numerical_hard_stop_reason') is not None:
        raise ValueError(f'{arm} pilot summary records a numerical hard stop')
    latest = read_json(run_dir / 'artifacts/heldout_probability_latest.json')
    positives = int(finite(latest['PositiveLabels'], f'{arm}.heldout.positive'))
    negatives = int(finite(latest['NegativeLabels'], f'{arm}.heldout.negative'))
    diagnostics = validate_diagnostics(run_dir / 'artifacts/diagnostics.csv', arm, protocol)
    checkpoint = run_dir / 'checkpoints/final_full.pt'
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f'{arm} pilot has no full 30k checkpoint')
    return {
        'source_commit': receipt['source_commit'],
        'run_dir': str(run_dir),
        'receipt': str(receipt_path),
        'checkpoint': str(checkpoint),
        'checkpoint_bytes': checkpoint.stat().st_size,
        'heldout_evidence_status': str(latest['evidence_status']),
        'heldout_positive_labels': positives,
        'heldout_negative_labels': negatives,
        'heldout_has_both_classes': positives > 0 and negatives > 0,
        'diagnostics': diagnostics,
    }


def validate_evaluation(path: Path, arm: str) -> dict[str, float]:
    """Validate the fixed 20-seed external stochastic-flow evaluation."""
    payload = read_json(path)
    if int(payload['training_seed']) != SEED or int(payload['milestone']) != TARGET_STEPS:
        raise ValueError(f'{arm} external evaluation identity mismatch')
    if tuple(int(value) for value in payload['seeds']) != EVAL_SEEDS:
        raise ValueError(f'{arm} external evaluation seeds changed')
    if payload['evaluation_mode'] != 'fixed-random-flow':
        raise ValueError(f'{arm} external evaluation is not the primary stochastic policy')
    if 'uniform[-1,1]' not in payload['base_action_policy']:
        raise ValueError(f'{arm} external evaluation omitted the uniform base draw')
    if 'standard normal' not in payload['wiener_increment_policy']:
        raise ValueError(f'{arm} external evaluation omitted Wiener increments')
    return {
        key: finite(payload[key], f'{arm}.evaluation.{key}')
        for key in (
            'mean_return',
            'mean_cost',
            'cost_cvar_20',
            'goal_met_episode_rate',
            'stall_ratio',
        )
    }


def validate_audit(path: Path, arm: str) -> dict[str, Any]:
    """Validate the final 64-state, 256-sample P4.4 four-mode audit."""
    payload = read_json(path)
    if (
        payload.get('stage') != 'P4.4-checkpoint-outward-decomposition'
        or payload.get('protocol_version') != PROTOCOL_VERSION
        or payload.get('protocol_sha256') != PROTOCOL_SHA256
    ):
        raise ValueError(f'{arm} pilot audit has the wrong P4.4 seal')
    if (
        payload.get('parent_arm') != arm
        or int(payload['training_seed']) != SEED
        or int(payload['env_steps']) != TARGET_STEPS
    ):
        raise ValueError(f'{arm} pilot audit identity mismatch')
    if int(payload['state_count']) != 64 or int(payload['action_samples_per_state']) != 256:
        raise ValueError(f'{arm} pilot audit changed the fixed sample')
    if bool(payload['flags']['outward_rectification_enabled']) != EXPECTED_RECTIFICATION[arm]:
        raise ValueError(f'{arm} pilot audit rectification flag mismatch')
    expected_specs = {
        'A': (False, True, False),
        'B': (False, True, True),
        'C': (True, False, True),
        'D': (True, True, True),
    }
    if set(payload['modes']) != set(expected_specs):
        raise ValueError(f'{arm} pilot audit does not contain exactly modes A--D')
    for mode, expected in expected_specs.items():
        spec = payload['modes'][mode]['specification']
        actual = (
            bool(spec['learned_drift']),
            bool(spec['wiener_noise']),
            bool(spec['action_clipping']),
        )
        if actual != expected:
            raise ValueError(f'{arm} pilot mode {mode} flags changed')
    heldout = payload.get('heldout_evidence', {})
    mode_d = {
        key: finite(value, f'{arm}.audit.D.{key}')
        for key, value in payload['modes']['D']['metrics'].items()
    }
    return {
        'path': str(path),
        'mode_D': mode_d,
        'heldout_auroc': heldout.get('Heldout/AUROC'),
        'heldout_brier_skill': heldout.get('Heldout/BrierSkill'),
        'heldout_evidence_sufficient': bool(
            heldout.get('Heldout/EvidenceSufficient', 0.0),
        ),
    }


def render_report(result: dict[str, Any]) -> str:
    """Render a compact pilot decision report."""
    lines = [
        '# STAR P4.4 seed-2028 pilot decision',
        '',
        f"Decision: **{result['decision']}**",
        '',
        '| arm | crossing D | executed v_out D | V_T D | success | stall | cost | CVaR-20 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for arm in ARMS:
        audit = result['arms'][arm]['audit']['mode_D']
        task = result['arms'][arm]['external']
        lines.append(
            f"| {arm} | {audit['crossing_rate']:.4f} | {audit['outward_speed']:.6f} | "
            f"{audit['tangent_variance']:.6f} | {task['goal_met_episode_rate']:.3f} | "
            f"{task['stall_ratio']:.3f} | {task['mean_cost']:.3f} | "
            f"{task['cost_cvar_20']:.3f} |",
        )
    lines.extend(
        [
            '',
            '## Frozen pilot gates',
            '',
            *[
                f"- {'PASS' if passed else 'FAIL'} — {name}"
                for name, passed in result['criteria'].items()
            ],
            '',
            f"Expansion authorized: **{result['expansion_authorized']}**",
            '',
        ],
    )
    return '\n'.join(lines)


def summarize(args: argparse.Namespace) -> dict[str, Any]:  # pylint: disable=too-many-locals
    """Validate all pilot evidence and apply the immutable R1-versus-R0 gates."""
    if sha256(args.protocol) != PROTOCOL_SHA256:
        raise ValueError('P4.4 protocol hash does not match its pretraining seal')
    protocol = read_json(args.protocol)
    if protocol.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('P4.4 protocol version mismatch')
    run_paths = bindings(args.runs, 'run')
    receipt_paths = bindings(args.receipts, 'receipt')
    evaluation_paths = bindings(args.evaluations, 'evaluation')
    audit_paths = bindings(args.audits, 'audit')
    arms = {}
    source_commits = set()
    for arm in ARMS:
        run = validate_run(arm, run_paths[arm], receipt_paths[arm], protocol)
        source_commits.add(run['source_commit'])
        arms[arm] = {
            'run': run,
            'external': validate_evaluation(evaluation_paths[arm], arm),
            'audit': validate_audit(audit_paths[arm], arm),
        }
    if len(source_commits) != 1:
        raise ValueError('R0 and R1 pilot arms did not use one source commit')
    r0_task = arms['R0']['external']
    r1_task = arms['R1']['external']
    r0_mode = arms['R0']['audit']['mode_D']
    r1_mode = arms['R1']['audit']['mode_D']
    threshold = protocol['pilot']['thresholds']
    criteria = {
        'r1_outward_speed_reduced_by_at_least_half': r1_mode['outward_speed']
        <= float(threshold['r1_outward_speed_ratio_maximum']) * r0_mode['outward_speed'],
        'r1_crossing_reduced_by_at_least_ten_percent': r1_mode['crossing_rate']
        <= float(threshold['r1_crossing_ratio_maximum']) * r0_mode['crossing_rate'],
        'r1_tangent_variance_preserved': r1_mode['tangent_variance']
        >= float(threshold['r1_tangent_variance_ratio_minimum']) * r0_mode['tangent_variance'],
        'r1_success_within_0_10_of_r0': r1_task['goal_met_episode_rate']
        >= r0_task['goal_met_episode_rate'] - float(threshold['r1_success_loss_maximum']),
        'r1_stall_within_0_10_of_r0': r1_task['stall_ratio']
        <= r0_task['stall_ratio'] + float(threshold['r1_stall_increase_maximum']),
        'r1_mean_cost_no_higher_than_r0': r1_task['mean_cost']
        <= float(threshold['r1_mean_cost_ratio_maximum']) * r0_task['mean_cost'],
        'r1_cvar_within_ratio': r1_task['cost_cvar_20']
        <= float(threshold['r1_cost_cvar_ratio_maximum']) * r0_task['cost_cvar_20'],
        'r0_numerically_stable': arms['R0']['run']['diagnostics']['passed'],
        'r1_numerically_stable': arms['R1']['run']['diagnostics']['passed'],
        'r0_heldout_has_both_classes': arms['R0']['run']['heldout_has_both_classes'],
        'r1_heldout_has_both_classes': arms['R1']['run']['heldout_has_both_classes'],
    }
    expansion_authorized = all(criteria.values())
    decision = (
        'continue_seed_2028_and_expand_to_three_seeds'
        if expansion_authorized
        else 'stop_after_seed_2028_pilot'
    )
    result = {
        'stage': 'P4.4-seed-2028-pilot-screen',
        'protocol_version': PROTOCOL_VERSION,
        'protocol_sha256': PROTOCOL_SHA256,
        'training_source_commit': next(iter(source_commits)),
        'seed': SEED,
        'target_steps': TARGET_STEPS,
        'external_evaluation_seeds': list(EVAL_SEEDS),
        'arms': arms,
        'criteria': criteria,
        'decision': decision,
        'expansion_authorized': expansion_authorized,
    }
    atomic_json(args.output, result)
    atomic_text(args.report, render_report(result))
    return result


def parse_args() -> argparse.Namespace:
    """Parse explicit R0/R1 pilot evidence bindings."""
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
    """Apply and print the immutable pilot decision."""
    result = summarize(parse_args())
    print(json.dumps({'decision': result['decision'], 'output': str(result['stage'])}))


if __name__ == '__main__':
    main()
