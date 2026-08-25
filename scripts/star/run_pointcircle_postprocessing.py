#!/usr/bin/env python3
"""Evaluate and finalize the conditional PointCircle four-method matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ENVIRONMENT = 'SafetyPointCircle1-v0'
ARMS = ('C0', 'P47')
METHOD_SLUGS = ('saclag', 'shapo_pidlag')
SEEDS = (2027, 2028, 2029)
FINAL_STEPS = 500_000
EXPECTED_PROTOCOL_SHA256 = '0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8'
EXPECTED_SHAPO_COMMIT = '1bb770437bbbe40a89938661343744243e164bf9'
PROTOCOL_RELATIVE = Path('reports/star/single_gpu_paper_core_preregistration.json')


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


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


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} does not contain a JSON object')
    return payload


def git_output(source: Path, *arguments: str) -> str:
    """Run one read-only Git identity command."""
    return subprocess.run(
        ['git', '-C', str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_sources(source: Path, official_source: Path) -> dict[str, str]:
    """Require clean fixed evaluator sources and the frozen protocol."""
    if git_output(source, 'status', '--porcelain'):
        raise ValueError('PointCircle postprocessing source must be clean')
    if git_output(official_source, 'status', '--porcelain'):
        raise ValueError('official SHAPO evaluation source must be clean')
    official_commit = git_output(official_source, 'rev-parse', 'HEAD')
    if official_commit != EXPECTED_SHAPO_COMMIT:
        raise ValueError(f'expected SHAPO {EXPECTED_SHAPO_COMMIT}, observed {official_commit}')
    protocol_sha = sha256_file(source / PROTOCOL_RELATIVE)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise ValueError('PointCircle postprocessor has the wrong protocol identity')
    return {
        'source_commit': git_output(source, 'rev-parse', 'HEAD'),
        'official_source_commit': official_commit,
        'protocol_sha256': protocol_sha,
    }


def wait_state(path: Path, poll_seconds: int) -> str:
    """Wait for a complete or preregistered screen-not-passed terminal state."""
    while True:
        if path.is_file():
            payload = read_json(path)
            if payload.get('protocol_sha256') != EXPECTED_PROTOCOL_SHA256:
                raise ValueError(f'{path} has the wrong protocol identity')
            state = str(payload.get('state'))
            if state in ('complete', 'screen_not_passed'):
                return state
        time.sleep(poll_seconds)


def unique_checkpoint(star_root: Path, arm: str, seed: int) -> Path:
    """Resolve exactly one lightweight final STAR checkpoint."""
    matches = sorted(
        (star_root / 'formal' / f'{arm}-{seed}').glob(
            f'training/STAR-{{{ENVIRONMENT}}}/seed-*/checkpoints/milestone_{FINAL_STEPS}.pt'
        )
    )
    if len(matches) != 1:
        raise ValueError(f'expected one {arm}:{seed} final checkpoint, found {len(matches)}')
    return matches[0]


def prior_receipts(prior_root: Path, method_slug: str) -> list[Path]:
    """Return the exact ordered three-seed prior receipt list."""
    return [prior_root / 'formal' / method_slug / f'seed-{seed}' / 'receipt.json' for seed in SEEDS]


def start(command: list[str], log: Path, source: Path) -> tuple[subprocess.Popen[str], Any]:
    """Start one postprocessing command with bytecode disabled."""
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open('a', encoding='utf-8')
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        command,
        cwd=source,
        env={
            **os.environ,
            'CUDA_VISIBLE_DEVICES': '0',
            'MKL_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONPATH': str(source),
        },
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, stream


def run_concurrent(commands: list[tuple[str, list[str], Path]], source: Path) -> None:
    """Run at most two paired postprocessing commands and report exact failures."""
    if len(commands) > 2:
        raise ValueError('postprocessing concurrency is capped at two')
    running = [(label, *start(command, log, source), log) for label, command, log in commands]
    failures: list[str] = []
    for label, process, stream, log in running:
        return_code = process.wait()
        stream.close()
        if return_code:
            failures.append(f'{label} exit={return_code} log={log}')
    if failures:
        raise RuntimeError('; '.join(failures))


def evaluate_star(args: argparse.Namespace) -> None:
    """Evaluate C0/P4.7 concurrently on the same fixed stream."""
    commands: list[tuple[str, list[str], Path]] = []
    for arm in ARMS:
        output = args.output_root / 'evaluations' / 'star' / arm
        checkpoints = [unique_checkpoint(args.star_root, arm, seed) for seed in SEEDS]
        command = [
            str(args.python_bin),
            str(args.source / 'scripts/star/evaluate_pointcircle_checkpoints.py'),
            '--arm',
            arm,
            '--checkpoints',
            *[str(path) for path in checkpoints],
            '--output-dir',
            str(output),
            '--work-dir',
            str(args.output_root / 'work' / 'star' / arm),
            '--device',
            'cuda:0',
        ]
        commands.append((arm, command, args.output_root / 'logs' / f'evaluate-star-{arm}.log'))
    run_concurrent(commands, args.source)
    for arm in ARMS:
        manifest = read_json(args.output_root / 'evaluations' / 'star' / arm / 'evaluation_manifest.json')
        if manifest.get('status') != 'complete' or manifest.get('arm') != arm:
            raise ValueError(f'{arm} STAR evaluation manifest is incomplete')


def evaluate_priors(args: argparse.Namespace, protocol_sha: str) -> None:
    """Evaluate both prior methods concurrently, three checkpoints per process."""
    commands: list[tuple[str, list[str], Path]] = []
    for slug in METHOD_SLUGS:
        output = args.output_root / 'evaluations' / 'prior' / slug
        command = [
            str(args.python_bin),
            str(args.source / 'scripts/star/evaluate_pointgoal_prior_baselines.py'),
            '--receipts',
            *[str(path) for path in prior_receipts(args.prior_root, slug)],
            '--official-source',
            str(args.official_shapo_source),
            '--output-dir',
            str(output),
            '--environment',
            ENVIRONMENT,
            '--expected-steps',
            str(FINAL_STEPS),
            '--expected-protocol-sha256',
            protocol_sha,
        ]
        commands.append((slug, command, args.output_root / 'logs' / f'evaluate-prior-{slug}.log'))
    run_concurrent(commands, args.source)
    for slug in METHOD_SLUGS:
        manifest = read_json(args.output_root / 'evaluations' / 'prior' / slug / 'evaluation_manifest.json')
        if (
            manifest.get('status') != 'complete'
            or manifest.get('environment') != ENVIRONMENT
            or int(manifest.get('training_steps', -1)) != FINAL_STEPS
        ):
            raise ValueError(f'{slug} prior evaluation manifest is incomplete')


def finalize(args: argparse.Namespace, protocol_sha: str) -> Path:
    """Build the final four-method table and return its manifest path."""
    prior_results = [
        args.output_root
        / 'evaluations'
        / 'prior'
        / slug
        / f'{slug}_seed_{seed}_step_{FINAL_STEPS}.json'
        for slug in METHOD_SLUGS
        for seed in SEEDS
    ]
    output = args.output_root / 'final'
    command = [
        str(args.python_bin),
        str(args.source / 'scripts/star/finalize_pointcircle_four_method_comparison.py'),
        '--star-training-root',
        str(args.star_root / 'formal'),
        '--star-evaluation-root',
        str(args.output_root / 'evaluations' / 'star'),
        '--prior-evaluations',
        *[str(path) for path in prior_results],
        '--screen-decision',
        str(args.screen_decision),
        '--expected-protocol-sha256',
        protocol_sha,
        '--p47-config',
        str(args.p47_config),
        '--output-dir',
        str(output),
    ]
    if args.expected_enhancement_preregistration_sha256 is not None:
        command.extend(
            [
                '--expected-enhancement-preregistration-sha256',
                args.expected_enhancement_preregistration_sha256,
            ],
        )
    log = args.output_root / 'logs' / 'finalize.log'
    process, stream = start(command, log, args.source)
    return_code = process.wait()
    stream.close()
    if return_code:
        raise RuntimeError(f'PointCircle finalization failed with exit={return_code}; see {log}')
    manifest = output / 'pointcircle_four_method_manifest.json'
    if read_json(manifest).get('status') != 'complete':
        raise ValueError('PointCircle final manifest is incomplete')
    return manifest


def parser() -> argparse.ArgumentParser:
    """Build postprocessing arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source', type=Path, required=True)
    result.add_argument('--official-shapo-source', type=Path, required=True)
    result.add_argument('--star-root', type=Path, required=True)
    result.add_argument('--prior-root', type=Path, required=True)
    result.add_argument('--output-root', type=Path, required=True)
    result.add_argument('--python-bin', type=Path, required=True)
    result.add_argument('--screen-decision', type=Path)
    result.add_argument('--p47-config', type=Path)
    result.add_argument('--expected-enhancement-preregistration-sha256')
    result.add_argument('--poll-seconds', type=int, default=30)
    return result


def main() -> None:
    """Wait for both matrices, evaluate them, and emit the final manifest."""
    args = parser().parse_args()
    for field in ('source', 'official_shapo_source', 'star_root', 'prior_root', 'output_root', 'python_bin'):
        setattr(args, field, getattr(args, field).resolve())
    args.screen_decision = (
        args.screen_decision.resolve()
        if args.screen_decision is not None
        else args.star_root / 'screen_decision.json'
    )
    args.p47_config = (
        args.p47_config.resolve()
        if args.p47_config is not None
        else args.source / 'configs/star/single_gpu_pointcircle_p47.yaml'
    )
    if args.poll_seconds <= 0:
        raise ValueError('poll interval must be positive')
    identities = validate_sources(args.source, args.official_shapo_source)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output_root / 'postprocessing_state.json',
        {
            'state': 'waiting_for_training',
            'started_at_utc': utc_now(),
            **identities,
        },
    )
    star_state = wait_state(args.star_root / 'queue_state.json', args.poll_seconds)
    prior_state = wait_state(args.prior_root / 'queue_state.json', args.poll_seconds)
    if star_state == 'screen_not_passed' or prior_state == 'screen_not_passed':
        atomic_json(
            args.output_root / 'postprocessing_state.json',
            {
                'state': 'screen_not_passed',
                'finished_at_utc': utc_now(),
                **identities,
            },
        )
        return
    atomic_json(
        args.output_root / 'postprocessing_state.json',
        {'state': 'evaluating', 'updated_at_utc': utc_now(), **identities},
    )
    evaluate_star(args)
    evaluate_priors(args, identities['protocol_sha256'])
    manifest = finalize(args, identities['protocol_sha256'])
    atomic_json(
        args.output_root / 'postprocessing_state.json',
        {
            'state': 'complete',
            'finished_at_utc': utc_now(),
            'final_manifest': str(manifest),
            'final_manifest_sha256': sha256_file(manifest),
            **identities,
        },
    )


if __name__ == '__main__':
    main()
