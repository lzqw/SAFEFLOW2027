#!/usr/bin/env python3
"""Validate all nine P4.3 decompositions and apply the P4.4 critic-alignment gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 'p4.4-outward-rectification-v1'
PROTOCOL_SHA256 = '005c42348a22f664448c688b3cecc16eba7cbbea55d7727537caeef5d3b69457'
ARMS = ('S0', 'S1', 'S2')
SEEDS = (2027, 2028, 2029)
MODES = ('A', 'B', 'C', 'D')
EXPECTED_FLAGS = {
    'S0': (True, False, False, False),
    'S1': (True, True, False, False),
    'S2': (True, True, True, False),
}


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


def atomic_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write strict, sorted JSON."""
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
    )


def finite_number(value: Any, location: str) -> float:
    """Return one finite scalar or reject malformed evidence."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{location} is not numeric') from error
    if not math.isfinite(number):
        raise ValueError(f'{location} is not finite')
    return number


def bindings(values: list[str]) -> dict[tuple[str, int], Path]:
    """Parse explicit ``ARM:SEED=PATH`` evidence bindings."""
    result: dict[tuple[str, int], Path] = {}
    for value in values:
        identity, separator, raw_path = value.partition('=')
        if not separator:
            raise ValueError(f'invalid audit binding {value!r}')
        arm, colon, raw_seed = identity.partition(':')
        if not colon or arm not in ARMS:
            raise ValueError(f'invalid audit identity {identity!r}')
        key = (arm, int(raw_seed))
        if key in result:
            raise ValueError(f'duplicate audit binding {identity!r}')
        result[key] = Path(raw_path)
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise ValueError(f'audit bindings mismatch; missing={missing}, extra={extra}')
    return result


def validate_audit(path: Path, arm: str, seed: int) -> dict[str, Any]:
    """Validate one fixed-size, immutable-checkpoint P4.4 decomposition."""
    payload = read_json(path)
    if (
        payload.get('stage') != 'P4.4-pretraining-outward-decomposition'
        or payload.get('protocol_version') != PROTOCOL_VERSION
        or payload.get('protocol_sha256') != PROTOCOL_SHA256
    ):
        raise ValueError(f'{arm} seed {seed} audit has the wrong P4.4 protocol seal')
    if payload.get('parent_arm') != arm or int(payload.get('training_seed', -1)) != seed:
        raise ValueError(f'{arm} seed {seed} audit identity mismatch')
    if int(payload.get('env_steps', -1)) != 50_000:
        raise ValueError(f'{arm} seed {seed} audit is not a final 50k checkpoint')
    if (
        int(payload.get('state_count', -1)) != 64
        or int(payload.get('action_samples_per_state', -1)) != 256
        or int(payload.get('audit_seed', -1)) != 4_000_007
    ):
        raise ValueError(f'{arm} seed {seed} changed the locked audit sample')
    flags = payload['flags']
    actual_flags = (
        bool(flags['stochastic_flow_enabled']),
        bool(flags['safety_anisotropy_enabled']),
        bool(flags['checkpoint_inward_drift_enabled']),
        bool(flags['candidate_probing_enabled']),
    )
    if actual_flags != EXPECTED_FLAGS[arm] or bool(flags['decomposition_inward_drift_enabled']):
        raise ValueError(f'{arm} seed {seed} decomposition flags mismatch')
    if int(payload['valid_fixed_normal_state_count']) <= 0:
        raise ValueError(f'{arm} seed {seed} has no valid probability normal')
    compact_modes: dict[str, dict[str, float]] = {}
    for mode in MODES:
        section = payload['modes'][mode]
        expected_spec = {
            'A': (False, True, False),
            'B': (False, True, True),
            'C': (True, False, True),
            'D': (True, True, True),
        }[mode]
        spec = section['specification']
        actual_spec = (
            bool(spec['learned_drift']),
            bool(spec['wiener_noise']),
            bool(spec['action_clipping']),
        )
        if actual_spec != expected_spec:
            raise ValueError(f'{arm} seed {seed} mode {mode} flags mismatch')
        compact_modes[mode] = {
            key: finite_number(value, f'{arm}.{seed}.{mode}.{key}')
            for key, value in section['metrics'].items()
        }
    alignment = payload['critic_alignment']
    valid_count = int(alignment['valid_state_count'])
    negative_count = int(alignment['negative_count'])
    if not 0 <= negative_count <= valid_count <= 64:
        raise ValueError(f'{arm} seed {seed} critic-alignment counts are invalid')
    if int(alignment['invalid_state_count']) != 64 - valid_count:
        raise ValueError(f'{arm} seed {seed} invalid alignment count disagrees')
    if valid_count:
        negative_fraction = finite_number(
            alignment['negative_fraction'],
            f'{arm}.{seed}.negative_fraction',
        )
        if not math.isclose(
            negative_fraction,
            negative_count / valid_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f'{arm} seed {seed} alignment fraction disagrees with counts')
        mean_cosine = finite_number(alignment['mean_cosine'], f'{arm}.{seed}.mean_cosine')
        median_cosine = finite_number(
            alignment['median_cosine'],
            f'{arm}.{seed}.median_cosine',
        )
    else:
        if any(alignment.get(key) is not None for key in ('negative_fraction', 'mean_cosine')):
            raise ValueError(f'{arm} seed {seed} empty alignment has numeric summaries')
        negative_fraction = None
        mean_cosine = None
        median_cosine = None
    return {
        'path': str(path),
        'checkpoint': str(payload['checkpoint']),
        'checkpoint_sha256': str(payload['checkpoint_sha256']),
        'selection': payload['selection'],
        'heldout_evidence': payload.get('heldout_evidence', {}),
        'valid_fixed_normal_state_count': int(payload['valid_fixed_normal_state_count']),
        'modes': compact_modes,
        'critic_alignment': {
            'valid_state_count': valid_count,
            'negative_count': negative_count,
            'negative_fraction': negative_fraction,
            'mean_cosine': mean_cosine,
            'median_cosine': median_cosine,
        },
    }


def _pooled_mode(runs: dict[int, dict[str, Any]], mode: str) -> dict[str, float]:
    """Average one mode across the three fixed training seeds."""
    keys = runs[SEEDS[0]]['modes'][mode]
    return {
        key: statistics.fmean(runs[seed]['modes'][mode][key] for seed in SEEDS)
        for key in keys
    }


def render_report(result: dict[str, Any]) -> str:
    """Render the pretraining critic gate and decomposition root-cause table."""
    gate = result['critic_alignment_gate']
    lines = [
        '# STAR P4.4 pretraining outward-drift decomposition',
        '',
        f"Decision: **{result['decision']}**",
        '',
        (
            'The nine immutable P4.3 final checkpoints were evaluated on 64 held-out '
            'near-boundary states with 256 common-random-number action samples per state.'
        ),
        '',
        '| arm | mode | V_N | V_T | trace | crossing | normal displacement | r_out | v_out | clip |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for arm in ARMS:
        for mode in MODES:
            metric = result['arms'][arm]['pooled_modes'][mode]
            lines.append(
                f"| {arm} | {mode} | {metric['normal_variance']:.6f} | "
                f"{metric['tangent_variance']:.6f} | "
                f"{metric['total_empirical_variance']:.6f} | "
                f"{metric['crossing_rate']:.4f} | "
                f"{metric['mean_normal_displacement']:.6f} | "
                f"{metric['outward_velocity_ratio']:.4f} | "
                f"{metric['outward_speed']:.6f} | {metric['clipping_rate']:.4f} |",
            )
    lines.extend(
        [
            '',
            '## Critic-normal agreement gate',
            '',
            f"- Valid cosine states: {gate['pooled_valid_state_count']} / 576",
            f"- Negative cosines: {gate['pooled_negative_count']}",
            f"- Pooled negative fraction: {gate['pooled_negative_fraction']:.6f}",
            (
                '- Per-checkpoint minimum valid states: '
                f"{gate['minimum_valid_states_across_checkpoints']}"
            ),
            f"- Training authorized: {gate['training_authorized']}",
            '',
            (
                'A blocked decision requires critic repair before any R0/R1 training; '
                'decomposition differences do not override that gate.'
            ),
            '',
        ],
    )
    return '\n'.join(lines)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    """Validate all nine audits and make the locked pretraining decision."""
    if sha256(args.protocol) != PROTOCOL_SHA256:
        raise ValueError('P4.4 protocol hash does not match its pretraining seal')
    protocol = read_json(args.protocol)
    if protocol.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('P4.4 protocol version mismatch')
    paths = bindings(args.audits)
    arms: dict[str, Any] = {}
    all_runs: list[dict[str, Any]] = []
    for arm in ARMS:
        runs = {seed: validate_audit(paths[(arm, seed)], arm, seed) for seed in SEEDS}
        all_runs.extend(runs.values())
        arms[arm] = {
            'runs': runs,
            'pooled_modes': {mode: _pooled_mode(runs, mode) for mode in MODES},
        }
    alignment_cfg = protocol['decomposition_audit']['critic_alignment']
    minimum_valid = int(alignment_cfg['valid_states_per_checkpoint_minimum'])
    negative_threshold = float(alignment_cfg['negative_fraction_block_threshold'])
    pooled_valid = sum(run['critic_alignment']['valid_state_count'] for run in all_runs)
    pooled_negative = sum(run['critic_alignment']['negative_count'] for run in all_runs)
    pooled_negative_fraction = 1.0 if pooled_valid <= 0 else pooled_negative / pooled_valid
    minimum_observed = min(
        run['critic_alignment']['valid_state_count'] for run in all_runs
    )
    per_checkpoint_gate = minimum_observed >= minimum_valid
    pooled_orientation_gate = pooled_negative_fraction < negative_threshold
    training_authorized = per_checkpoint_gate and pooled_orientation_gate
    decision = (
        'p4_4_training_authorized'
        if training_authorized
        else 'blocked_for_critic_repair'
    )
    result = {
        'stage': 'P4.4-pretraining-critic-alignment-gate',
        'protocol_version': PROTOCOL_VERSION,
        'protocol_sha256': PROTOCOL_SHA256,
        'parent_checkpoint_count': len(all_runs),
        'arms': arms,
        'critic_alignment_gate': {
            'valid_states_per_checkpoint_minimum': minimum_valid,
            'pooled_negative_fraction_block_threshold': negative_threshold,
            'minimum_valid_states_across_checkpoints': minimum_observed,
            'pooled_valid_state_count': pooled_valid,
            'pooled_negative_count': pooled_negative,
            'pooled_negative_fraction': pooled_negative_fraction,
            'per_checkpoint_validity_pass': per_checkpoint_gate,
            'pooled_orientation_pass': pooled_orientation_gate,
            'training_authorized': training_authorized,
        },
        'decision': decision,
        'training_started_before_decision': False,
    }
    atomic_json(args.output, result)
    atomic_text(args.report, render_report(result))
    return result


def parse_args() -> argparse.Namespace:
    """Parse the explicit protocol and nine audit bindings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--audits', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Apply and print the immutable pretraining decision."""
    result = summarize(parse_args())
    print(json.dumps({'decision': result['decision'], 'output': str(result['stage'])}))


if __name__ == '__main__':
    main()
