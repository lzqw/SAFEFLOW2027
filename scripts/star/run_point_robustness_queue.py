#!/usr/bin/env python3
"""Run the frozen checkpoint-only PointGoal/PointCircle robustness matrix."""

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

from scripts.star.decide_pointcircle_geometry_enhancement import ADDENDUM_SHA256

ENVIRONMENTS = ('SafetyPointGoal1-v0', 'SafetyPointCircle1-v0')
ARMS = ('C0', 'P47')
SEEDS = (2027, 2028, 2029)
EXPECTED_GLOBAL_PROTOCOL_SHA256 = '0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8'
GLOBAL_PROTOCOL_RELATIVE = Path('reports/star/single_gpu_paper_core_preregistration.json')
ROBUSTNESS_PROTOCOL_RELATIVE = Path('reports/star/single_gpu_robustness_protocol.json')


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


def validate_source(source: Path) -> dict[str, str]:
    """Require clean source and both locked protocol identities."""
    if git_output(source, 'status', '--porcelain'):
        raise ValueError('robustness source must be clean')
    global_sha = sha256_file(source / GLOBAL_PROTOCOL_RELATIVE)
    if global_sha != EXPECTED_GLOBAL_PROTOCOL_SHA256:
        raise ValueError('robustness source has the wrong global protocol')
    robustness_path = source / ROBUSTNESS_PROTOCOL_RELATIVE
    robustness = read_json(robustness_path)
    if (
        robustness.get('version') != 'star-single-gpu-robustness-v1'
        or robustness.get('status') != 'locked_before_any_new_robustness_evaluation'
        or bool(robustness.get('reporting', {}).get('retraining'))
    ):
        raise ValueError('robustness supplemental protocol is not locked checkpoint-only')
    return {
        'source_commit': git_output(source, 'rev-parse', 'HEAD'),
        'global_protocol_sha256': global_sha,
        'robustness_protocol_sha256': sha256_file(robustness_path),
    }


def wait_for_prerequisite(
    path: Path,
    poll_seconds: int,
    expected_enhancement_sha256: str | None,
) -> bool:
    """Wait for completed controls or propagate the preregistered screen failure."""
    while True:
        if path.is_file():
            payload = read_json(path)
            if payload.get('protocol_sha256') != EXPECTED_GLOBAL_PROTOCOL_SHA256:
                raise ValueError('robustness prerequisite has the wrong global protocol')
            state = payload.get('state')
            if state == 'complete':
                manifest = Path(payload['final_manifest'])
                manifest_payload = read_json(manifest)
                if manifest_payload.get('status') != 'complete':
                    raise ValueError('robustness prerequisite final manifest is invalid')
                if (
                    expected_enhancement_sha256 is not None
                    and manifest_payload.get('enhancement_preregistration_sha256')
                    != expected_enhancement_sha256
                ):
                    raise ValueError('robustness prerequisite does not use the enhanced method')
                return True
            if state == 'screen_not_passed':
                return False
        time.sleep(poll_seconds)


def wait_for_resource_release(path: Path, poll_seconds: int) -> str:
    """Wait until another frozen queue releases the single GPU."""
    while True:
        if path.is_file():
            payload = read_json(path)
            if payload.get('protocol_sha256') != EXPECTED_GLOBAL_PROTOCOL_SHA256:
                raise ValueError('resource prerequisite has the wrong global protocol')
            state = str(payload.get('state'))
            if state in ('complete', 'failed'):
                return state
        time.sleep(poll_seconds)


def validate_environment_scope(args: argparse.Namespace) -> tuple[str, ...]:
    """Validate and normalize the requested robustness task subset."""
    environments = tuple(dict.fromkeys(str(item) for item in args.environments))
    if not environments or any(item not in ENVIRONMENTS for item in environments):
        raise ValueError('robustness environment scope is invalid')
    if args.run_pointgoal_after_screen_failure and environments != ('SafetyPointGoal1-v0',):
        raise ValueError(
            '--run-pointgoal-after-screen-failure requires PointGoal as the only environment'
        )
    return environments


def unique_checkpoint(args: argparse.Namespace, environment: str, arm: str, seed: int) -> Path:
    """Resolve one exact lightweight final checkpoint."""
    if environment == 'SafetyPointGoal1-v0':
        container = args.pointgoal_root / f'{arm}-{seed}'
        milestone = 200_000
    else:
        container = args.pointcircle_root / 'formal' / f'{arm}-{seed}'
        milestone = 500_000
    matches = sorted(
        container.glob(
            f'training/STAR-{{{environment}}}/seed-*/checkpoints/milestone_{milestone}.pt'
        )
    )
    if len(matches) != 1:
        raise ValueError(f'expected one {environment}:{arm}:{seed} checkpoint, found {len(matches)}')
    return matches[0]


def result_path(root: Path, environment: str, arm: str, seed: int) -> Path:
    """Return one stable robustness result path."""
    task = 'pointgoal' if environment == 'SafetyPointGoal1-v0' else 'pointcircle'
    return root / 'evaluations' / task / f'{arm}-{seed}.json'


def reusable_result(
    path: Path,
    environment: str,
    arm: str,
    seed: int,
    checkpoint: Path,
    protocol_sha: str,
) -> bool:
    """Return whether an existing result matches the exact requested identity."""
    if not path.is_file():
        return False
    payload = read_json(path)
    exact = (
        payload.get('status') == 'complete'
        and payload.get('environment') == environment
        and payload.get('arm') == arm
        and int(payload.get('training_seed', -1)) == seed
        and payload.get('checkpoint_sha256') == sha256_file(checkpoint)
        and payload.get('robustness_protocol_sha256') == protocol_sha
        and bool(payload.get('retraining')) is False
    )
    if not exact:
        raise FileExistsError(f'refusing to overwrite incompatible robustness result {path}')
    return True


def command_for(
    args: argparse.Namespace,
    identities: dict[str, str],
    environment: str,
    arm: str,
    seed: int,
) -> tuple[list[str], Path, Path]:
    """Build one exact checkpoint-only robustness command."""
    checkpoint = unique_checkpoint(args, environment, arm, seed)
    output = result_path(args.run_root, environment, arm, seed)
    log = output.with_suffix('.log')
    command = [
        str(args.python_bin),
        str(args.source / 'scripts/star/evaluate_point_robustness.py'),
        '--environment',
        environment,
        '--arm',
        arm,
        '--checkpoint',
        str(checkpoint),
        '--protocol',
        str(args.source / ROBUSTNESS_PROTOCOL_RELATIVE),
        '--expected-protocol-sha256',
        identities['robustness_protocol_sha256'],
        '--output',
        str(output),
        '--work-dir',
        str(args.run_root / 'work' / environment / f'{arm}-{seed}'),
        '--device',
        'cuda:0',
    ]
    return command, output, log


def start(command: list[str], log: Path, source: Path) -> tuple[subprocess.Popen[str], Any]:
    """Start one evaluator with isolated single-process resource settings."""
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


def run_pair(
    args: argparse.Namespace,
    identities: dict[str, str],
    environment: str,
    seed: int,
) -> None:
    """Evaluate the paired C0/P47 checkpoints concurrently."""
    running: list[tuple[str, subprocess.Popen[str], Any, Path]] = []
    for arm in ARMS:
        command, output, log = command_for(args, identities, environment, arm, seed)
        checkpoint = Path(command[command.index('--checkpoint') + 1])
        if reusable_result(
            output,
            environment,
            arm,
            seed,
            checkpoint,
            identities['robustness_protocol_sha256'],
        ):
            continue
        process, stream = start(command, log, args.source)
        running.append((arm, process, stream, log))
    failures: list[str] = []
    for arm, process, stream, log in running:
        return_code = process.wait()
        stream.close()
        if return_code:
            failures.append(f'{environment}:{arm}:{seed} exit={return_code} log={log}')
    if failures:
        raise RuntimeError('; '.join(failures))


def finalize(args: argparse.Namespace, identities: dict[str, str]) -> Path:
    """Build the final robustness report and return its manifest."""
    results = [
        result_path(args.run_root, environment, arm, seed)
        for environment in args.environments
        for arm in ARMS
        for seed in SEEDS
    ]
    output = args.run_root / 'final'
    command = [
        str(args.python_bin),
        str(args.source / 'scripts/star/finalize_point_robustness.py'),
        '--results',
        *[str(path) for path in results],
        '--protocol',
        str(args.source / ROBUSTNESS_PROTOCOL_RELATIVE),
        '--expected-protocol-sha256',
        identities['robustness_protocol_sha256'],
        '--environments',
        *args.environments,
        '--output-dir',
        str(output),
    ]
    log = args.run_root / 'finalize.log'
    process, stream = start(command, log, args.source)
    return_code = process.wait()
    stream.close()
    if return_code:
        raise RuntimeError(f'robustness finalization exit={return_code}; see {log}')
    manifest = output / 'point_robustness_manifest.json'
    if read_json(manifest).get('status') != 'complete':
        raise ValueError('robustness final manifest is incomplete')
    return manifest


def parser() -> argparse.ArgumentParser:
    """Build queue arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source', type=Path, required=True)
    result.add_argument('--pointgoal-root', type=Path, required=True)
    result.add_argument('--pointcircle-root', type=Path, required=True)
    result.add_argument('--run-root', type=Path, required=True)
    result.add_argument('--python-bin', type=Path, required=True)
    result.add_argument('--prerequisite-state', type=Path, required=True)
    result.add_argument(
        '--environments',
        nargs='+',
        choices=ENVIRONMENTS,
        default=list(ENVIRONMENTS),
    )
    result.add_argument('--run-pointgoal-after-screen-failure', action='store_true')
    result.add_argument('--resource-prerequisite-state', type=Path)
    result.add_argument('--geometry-enhanced', action='store_true')
    result.add_argument('--poll-seconds', type=int, default=30)
    return result


def main() -> None:
    """Wait for core training, run checkpoint-only evaluations, and finalize."""
    args = parser().parse_args()
    for field in (
        'source',
        'pointgoal_root',
        'pointcircle_root',
        'run_root',
        'python_bin',
        'prerequisite_state',
    ):
        setattr(args, field, getattr(args, field).resolve())
    if args.resource_prerequisite_state is not None:
        args.resource_prerequisite_state = args.resource_prerequisite_state.resolve()
    if args.poll_seconds <= 0:
        raise ValueError('poll interval must be positive')
    args.environments = validate_environment_scope(args)
    identities = validate_source(args.source)
    args.run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_root / 'queue_identity.json',
        {
            **identities,
            'environments': list(args.environments),
            'arms': list(ARMS),
            'training_seeds': list(SEEDS),
            'retraining': False,
            'maximum_concurrent_evaluators': 2,
            'geometry_enhanced': args.geometry_enhanced,
            'enhancement_preregistration_sha256': (
                ADDENDUM_SHA256 if args.geometry_enhanced else None
            ),
        },
    )
    screen_passed = wait_for_prerequisite(
        args.prerequisite_state,
        args.poll_seconds,
        ADDENDUM_SHA256 if args.geometry_enhanced else None,
    )
    if not screen_passed and not args.run_pointgoal_after_screen_failure:
        atomic_json(
            args.run_root / 'queue_state.json',
            {
                'state': 'screen_not_passed',
                'finished_at_utc': utc_now(),
                'retraining': False,
                'protocol_sha256': identities['global_protocol_sha256'],
            },
        )
        return
    resource_terminal_state = None
    if args.resource_prerequisite_state is not None:
        atomic_json(
            args.run_root / 'queue_state.json',
            {
                'state': 'waiting_for_resource',
                'started_at_utc': utc_now(),
                'retraining': False,
                'protocol_sha256': identities['global_protocol_sha256'],
                'environments': list(args.environments),
                'pointcircle_screen_passed': screen_passed,
                'scientific_training_launched': False,
                'resource_prerequisite_state': str(args.resource_prerequisite_state),
            },
        )
        resource_terminal_state = wait_for_resource_release(
            args.resource_prerequisite_state,
            args.poll_seconds,
        )
    atomic_json(
        args.run_root / 'queue_state.json',
        {
            'state': 'evaluating',
            'started_at_utc': utc_now(),
            'retraining': False,
            'protocol_sha256': identities['global_protocol_sha256'],
            'environments': list(args.environments),
            'pointcircle_screen_passed': screen_passed,
            'scientific_training_launched': False,
            'checkpoint_only_evaluation_launched': True,
            'resource_terminal_state': resource_terminal_state,
        },
    )
    for environment in args.environments:
        for seed in SEEDS:
            run_pair(args, identities, environment, seed)
    manifest = finalize(args, identities)
    atomic_json(
        args.run_root / 'queue_state.json',
        {
            'state': 'complete',
            'finished_at_utc': utc_now(),
            'retraining': False,
            'protocol_sha256': identities['global_protocol_sha256'],
            'robustness_protocol_sha256': identities['robustness_protocol_sha256'],
            'final_manifest': str(manifest),
            'final_manifest_sha256': sha256_file(manifest),
            'geometry_enhanced': args.geometry_enhanced,
            'enhancement_preregistration_sha256': (
                ADDENDUM_SHA256 if args.geometry_enhanced else None
            ),
            'environments': list(args.environments),
            'pointcircle_screen_passed': screen_passed,
            'checkpoint_only_evaluation_launched': True,
            'resource_terminal_state': resource_terminal_state,
        },
    )


if __name__ == '__main__':
    main()
