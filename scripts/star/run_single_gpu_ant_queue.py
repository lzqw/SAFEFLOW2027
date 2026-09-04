#!/usr/bin/env python3
"""Run the Ant 2M C0/P4.7 matrix only after the Point core is terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ENVIRONMENT = 'Ant-v4'
ARMS = ('C0', 'P47')
SEEDS = (2027, 2028, 2029)
TRAINING_STEPS = 2_000_000
REPLAY_SIZE = 400_000
TRAINING_SOURCE_COMMIT = '62ca7399357c10645a955dcbead7ed2153a82c69'
PARENT_PROTOCOL_SHA256 = '0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8'
ANT_PROTOCOL_SHA256 = '5769bd35b603733dd038f94aea1b7dbc45cfd5335314788836e316781bed2692'
PARENT_PROTOCOL_RELATIVE = Path('reports/star/single_gpu_paper_core_preregistration.json')
ANT_PROTOCOL_RELATIVE = Path('reports/star/single_gpu_ant_preregistration.json')
CONFIG_RELATIVE = {
    'C0': Path('configs/star/single_gpu_ant_c0.yaml'),
    'P47': Path('configs/star/single_gpu_ant_p47.yaml'),
}
CONFIG_SHA256 = {
    'C0': 'db16e3ed72b0a23a47bd9fd7db0f6aea863e8e1f338a35ada3d25618db4848b9',
    'P47': '4d011ec9c0eaedf1f51835804c3e0476114e4a0427d8ef32a5b9a0336fc97ebf',
}
REUSED_AUDIT_RELATIVE = Path(
    'reports/star/shapo_p47_four_env_artifacts/milestones/'
    'step_2000000/ant-2027/milestone_audit.json',
)
REUSED_EVALUATION_RELATIVE = Path(
    'reports/star/shapo_p47_four_env_artifacts/milestones/'
    'step_2000000/ant-2027/artifacts/evaluation_step_2000000.json',
)
REUSED_AUDIT_SHA256 = '9e5102da78dae0ef24712df1c081a9f36026214a9e784fc7f2ab925df264aa14'
REUSED_EVALUATION_SHA256 = '54dce88814b8e7072613cc17e50b3f5c38995f7ffdde34f63a55b5de07a76a24'
REUSED_CHECKPOINT_SHA256 = '39a9d7a0717402d5e3fea5694cd0724aebd0982306beb1fb39c8091135fddc35'
WAVES = (
    (('C0', 2028), ('P47', 2028)),
    (('C0', 2029), ('P47', 2029)),
    (('C0', 2027),),
)


def utc_now() -> str:
    """Return one UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


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


def git_output(source: Path, *arguments: str) -> str:
    """Run one read-only Git identity command."""
    return subprocess.run(
        ['git', '-C', str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping."""
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} does not contain a YAML mapping')
    return payload


def validate_config_pair(source: Path) -> None:
    """Require the C0/P4.7 configs to differ only in the two frozen method switches."""
    c0 = load_yaml(source / CONFIG_RELATIVE['C0'])
    p47 = load_yaml(source / CONFIG_RELATIVE['P47'])
    if c0.get('env_id') != ENVIRONMENT or p47.get('env_id') != ENVIRONMENT:
        raise ValueError('Ant config has the wrong environment')
    normalized = deepcopy(p47['custom_cfgs'])
    algo = normalized['algo_cfgs']
    if (
        algo.get('directional_loss_enabled') is not True
        or algo.get('safety_aware_cost_geometry_enabled') is not True
    ):
        raise ValueError('P47 Ant config does not enable the frozen method')
    algo['directional_loss_enabled'] = False
    algo['safety_aware_cost_geometry_enabled'] = False
    if normalized != c0['custom_cfgs']:
        raise ValueError('C0/P47 Ant configs differ outside the two preregistered switches')


def validate_sources(
    controller_source: Path,
    training_source: Path,
    expected_controller_commit: str,
) -> dict[str, Any]:
    """Authenticate clean controller and exact historical training code."""
    if git_output(controller_source, 'status', '--porcelain'):
        raise ValueError('Ant controller source must be clean')
    controller_commit = git_output(controller_source, 'rev-parse', 'HEAD')
    if controller_commit != expected_controller_commit:
        raise ValueError(
            f'controller commit mismatch: expected {expected_controller_commit}, '
            f'found {controller_commit}',
        )
    if git_output(training_source, 'status', '--porcelain'):
        raise ValueError('Ant training source must be clean')
    training_commit = git_output(training_source, 'rev-parse', 'HEAD')
    if training_commit != TRAINING_SOURCE_COMMIT:
        raise ValueError(
            f'training commit mismatch: expected {TRAINING_SOURCE_COMMIT}, found {training_commit}',
        )
    parent_sha = sha256_file(controller_source / PARENT_PROTOCOL_RELATIVE)
    ant_sha = sha256_file(controller_source / ANT_PROTOCOL_RELATIVE)
    config_hashes = {
        arm: sha256_file(controller_source / relative)
        for arm, relative in CONFIG_RELATIVE.items()
    }
    if parent_sha != PARENT_PROTOCOL_SHA256 or ant_sha != ANT_PROTOCOL_SHA256:
        raise ValueError('Ant queue does not match its frozen protocol identities')
    if config_hashes != CONFIG_SHA256:
        raise ValueError('Ant queue does not match its frozen configs')
    validate_config_pair(controller_source)
    protocol = read_json(controller_source / ANT_PROTOCOL_RELATIVE)
    if (
        protocol.get('version') != 'star-single-gpu-ant-v1'
        or protocol.get('status') != 'locked_before_any_new_single_gpu_ant_training'
        or protocol.get('parent_protocol_sha256') != PARENT_PROTOCOL_SHA256
    ):
        raise ValueError('Ant supplemental protocol is not locked')
    return {
        'controller_source_commit': controller_commit,
        'training_source_commit': training_commit,
        'parent_protocol_sha256': parent_sha,
        'ant_protocol_sha256': ant_sha,
        'config_sha256': config_hashes,
    }


def wait_for_point_core(path: Path, poll_seconds: int) -> str:
    """Wait until Point core reaches a successful or screened-out terminal state."""
    while True:
        if path.is_file():
            payload = read_json(path)
            state = str(payload.get('state'))
            if state == 'failed':
                raise RuntimeError(f'Point-core prerequisite failed: {payload.get("error")}')
            if (
                state in ('complete', 'screen_not_passed')
                and payload.get('protocol_sha256') == PARENT_PROTOCOL_SHA256
                and payload.get('retraining') is False
            ):
                if state == 'complete':
                    manifest = Path(str(payload['final_manifest']))
                    if read_json(manifest).get('status') != 'complete':
                        raise ValueError('Point-core robustness manifest is incomplete')
                return state
        time.sleep(poll_seconds)


def validate_reused_seed(controller_source: Path, checkpoint: Path) -> dict[str, str]:
    """Authenticate the sealed P4.7 seed-2027 Ant checkpoint and portable evidence."""
    audit_path = controller_source / REUSED_AUDIT_RELATIVE
    evaluation_path = controller_source / REUSED_EVALUATION_RELATIVE
    if sha256_file(audit_path) != REUSED_AUDIT_SHA256:
        raise ValueError('reused Ant audit changed')
    if sha256_file(evaluation_path) != REUSED_EVALUATION_SHA256:
        raise ValueError('reused Ant evaluation changed')
    audit = read_json(audit_path)
    checkpoint_row = audit.get('server_only_checkpoint', {})
    valid = (
        audit.get('status') == 'passed'
        and audit.get('environment') == ENVIRONMENT
        and int(audit.get('training_seed', -1)) == 2027
        and int(audit.get('milestone', -1)) == TRAINING_STEPS
        and audit.get('source_commit') == TRAINING_SOURCE_COMMIT
        and checkpoint_row.get('sha256') == REUSED_CHECKPOINT_SHA256
        and int(checkpoint_row.get('env_steps', -1)) == TRAINING_STEPS
    )
    if not valid:
        raise ValueError('reused Ant P47 seed-2027 audit is invalid')
    if not checkpoint.is_file() or sha256_file(checkpoint) != REUSED_CHECKPOINT_SHA256:
        raise ValueError('reused Ant P47 seed-2027 checkpoint is missing or changed')
    return {
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': REUSED_CHECKPOINT_SHA256,
        'audit': str(audit_path),
        'audit_sha256': REUSED_AUDIT_SHA256,
        'evaluation': str(evaluation_path),
        'evaluation_sha256': REUSED_EVALUATION_SHA256,
    }


def run_container(root: Path, arm: str, seed: int) -> Path:
    """Return one stable Ant run container."""
    return root / 'runs' / f'{arm}-{seed}'


def receipt_complete(
    path: Path,
    arm: str,
    seed: int,
    identities: dict[str, Any],
) -> bool:
    """Return whether a receipt proves exact run completion."""
    if not path.is_file():
        return False
    payload = read_json(path)
    valid = (
        payload.get('status') == 'passed'
        and payload.get('algorithm') == 'STAR'
        and payload.get('environment') == ENVIRONMENT
        and int(payload.get('seed', -1)) == seed
        and int(payload.get('steps', -1)) == TRAINING_STEPS
        and payload.get('source_commit') == TRAINING_SOURCE_COMMIT
        and payload.get('source_dirty') is False
        and payload.get('config_sha256') == identities['config_sha256'][arm]
    )
    if not valid:
        raise FileExistsError(f'refusing to overwrite incompatible Ant receipt {path}')
    return True


def newest_recovery_checkpoint(container: Path) -> Path | None:
    """Return the newest replay-bearing checkpoint after an engineering interruption."""
    matches = list(container.glob(f'training/STAR-{{{ENVIRONMENT}}}/seed-*/checkpoints/latest_full.pt'))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def training_command(
    args: argparse.Namespace,
    identities: dict[str, Any],
    arm: str,
    seed: int,
) -> tuple[list[str], Path, Path, Path | None]:
    """Build one exact Ant learner command."""
    container = run_container(args.run_root, arm, seed)
    receipt = container / 'receipt.json'
    log = container / 'train.log'
    config = args.controller_source / CONFIG_RELATIVE[arm]
    if sha256_file(config) != identities['config_sha256'][arm]:
        raise ValueError(f'{arm} Ant config changed after queue validation')
    resume = newest_recovery_checkpoint(container)
    command = [
        str(args.python_bin),
        str(args.training_source / 'scripts/star/train_star.py'),
        '--config',
        str(config),
        '--seed',
        str(seed),
        '--steps',
        str(TRAINING_STEPS),
        '--steps-per-epoch',
        '5000',
        '--device',
        'cuda:0',
        '--vector-envs',
        '1',
        '--batch-size',
        '256',
        '--replay-size',
        str(REPLAY_SIZE),
        '--checkpoint-replay',
        '--full-checkpoint-freq',
        '250000',
        '--log-dir',
        str(container / 'training'),
        '--result',
        str(receipt),
        '--no-tensorboard',
    ]
    if resume is not None:
        command.extend(['--resume', str(resume)])
    return command, receipt, log, resume


def start_process(
    command: list[str],
    log: Path,
    training_source: Path,
) -> tuple[subprocess.Popen[str], Any]:
    """Start one learner with isolated single-environment resources."""
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open('a', encoding='utf-8')
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        command,
        cwd=training_source,
        env={
            **os.environ,
            'CUDA_VISIBLE_DEVICES': '0',
            'MKL_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONPATH': str(training_source),
        },
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, stream


def run_wave(
    args: argparse.Namespace,
    identities: dict[str, Any],
    wave_index: int,
    jobs: tuple[tuple[str, int], ...],
) -> dict[str, Any]:
    """Run at most two Ant learners and stop the counterpart on engineering failure."""
    if not 1 <= len(jobs) <= 2:
        raise ValueError('one Ant wave must contain one or two learners')
    pending = []
    rows: list[dict[str, Any]] = []
    for arm, seed in jobs:
        command, receipt, log, resume = training_command(args, identities, arm, seed)
        if receipt.is_file() and receipt_complete(receipt, arm, seed, identities):
            rows.append({'arm': arm, 'seed': seed, 'state': 'already_complete'})
        else:
            pending.append((arm, seed, command, receipt, log, resume))
    started_at = utc_now()
    started = time.monotonic()
    running: dict[tuple[str, int], tuple[subprocess.Popen[str], Any, Path, Path]] = {}
    for arm, seed, command, receipt, log, resume in pending:
        process, stream = start_process(command, log, args.training_source)
        running[(arm, seed)] = (process, stream, receipt, log)
        rows.append(
            {
                'arm': arm,
                'seed': seed,
                'state': 'started',
                'resume': str(resume) if resume is not None else None,
            },
        )
    failures = []
    while running:
        for (arm, seed), (process, stream, receipt, log) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            stream.close()
            del running[(arm, seed)]
            if return_code or not receipt_complete(receipt, arm, seed, identities):
                failures.append(f'{arm}:{seed} exit={return_code} log={log}')
            else:
                rows.append({'arm': arm, 'seed': seed, 'state': 'completed'})
        if failures and running:
            for process, _, _, _ in running.values():
                process.terminate()
            for process, stream, _, _ in running.values():
                process.wait(timeout=30)
                stream.close()
            running.clear()
        elif running:
            time.sleep(5)
    receipt = {
        'wave': wave_index,
        'jobs': [{'arm': arm, 'seed': seed} for arm, seed in jobs],
        'maximum_concurrent_learners': 2,
        'started_at_utc': started_at,
        'finished_at_utc': utc_now(),
        'elapsed_seconds': time.monotonic() - started,
        'rows': rows,
        'ant_protocol_sha256': identities['ant_protocol_sha256'],
    }
    atomic_json(args.run_root / f'wave-{wave_index}.json', receipt)
    if failures:
        raise RuntimeError('; '.join(failures))
    return receipt


def finalize(args: argparse.Namespace, identities: dict[str, Any]) -> Path:
    """Run the strict Ant aggregator and return its manifest."""
    output = args.run_root / 'final'
    log = args.run_root / 'finalize.log'
    command = [
        str(args.python_bin),
        str(args.controller_source / 'scripts/star/finalize_single_gpu_ant.py'),
        '--run-root',
        str(args.run_root),
        '--reused-evaluation',
        str(args.controller_source / REUSED_EVALUATION_RELATIVE),
        '--reused-audit',
        str(args.controller_source / REUSED_AUDIT_RELATIVE),
        '--protocol',
        str(args.controller_source / ANT_PROTOCOL_RELATIVE),
        '--expected-protocol-sha256',
        identities['ant_protocol_sha256'],
        '--output-dir',
        str(output),
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        subprocess.run(
            command,
            cwd=args.controller_source,
            env={
                **os.environ,
                'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONPATH': str(args.controller_source),
            },
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    manifest = output / 'ant_manifest.json'
    if read_json(manifest).get('status') != 'complete':
        raise ValueError('Ant final manifest is incomplete')
    return manifest


def parser() -> argparse.ArgumentParser:
    """Build queue arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--controller-source', type=Path, required=True)
    result.add_argument('--training-source', type=Path, required=True)
    result.add_argument('--run-root', type=Path, required=True)
    result.add_argument('--python-bin', type=Path, required=True)
    result.add_argument('--point-core-state', type=Path, required=True)
    result.add_argument('--reused-p47-2027-checkpoint', type=Path, required=True)
    result.add_argument('--expected-controller-commit', required=True)
    result.add_argument('--poll-seconds', type=int, default=30)
    return result


def main() -> None:
    """Wait for Point core, execute five new runs, and finalize the Ant table."""
    args = parser().parse_args()
    for field in (
        'controller_source',
        'training_source',
        'run_root',
        'python_bin',
        'point_core_state',
        'reused_p47_2027_checkpoint',
    ):
        setattr(args, field, getattr(args, field).resolve())
    if args.poll_seconds <= 0:
        raise ValueError('poll interval must be positive')
    identities = validate_sources(
        args.controller_source,
        args.training_source,
        args.expected_controller_commit,
    )
    args.run_root.mkdir(parents=True, exist_ok=True)
    existing_state = args.run_root / 'queue_state.json'
    if existing_state.is_file():
        existing = read_json(existing_state)
        if existing.get('state') == 'complete':
            manifest = Path(str(existing['final_manifest']))
            if read_json(manifest).get('status') == 'complete':
                return
    reused = validate_reused_seed(
        args.controller_source,
        args.reused_p47_2027_checkpoint,
    )
    atomic_json(
        args.run_root / 'queue_identity.json',
        {
            **identities,
            'environment': ENVIRONMENT,
            'arms': list(ARMS),
            'seeds': list(SEEDS),
            'steps_per_run': TRAINING_STEPS,
            'new_training_steps': 10_000_000,
            'maximum_concurrent_learners': 2,
            'vector_env_nums': 1,
            'utd_ratio': 1.0,
            'reused_P47_2027': reused,
            'waves': [[f'{arm}-{seed}' for arm, seed in wave] for wave in WAVES],
        },
    )
    atomic_json(
        args.run_root / 'queue_state.json',
        {
            'state': 'waiting_for_point_core',
            'started_at_utc': utc_now(),
            'protocol_sha256': PARENT_PROTOCOL_SHA256,
            'ant_protocol_sha256': ANT_PROTOCOL_SHA256,
            'scientific_training_launched': False,
        },
    )
    try:
        point_core_state = wait_for_point_core(args.point_core_state, args.poll_seconds)
        atomic_json(
            args.run_root / 'queue_state.json',
            {
                'state': 'training',
                'started_at_utc': utc_now(),
                'protocol_sha256': PARENT_PROTOCOL_SHA256,
                'ant_protocol_sha256': ANT_PROTOCOL_SHA256,
                'point_core_terminal_state': point_core_state,
                'scientific_training_launched': True,
            },
        )
        for wave_index, jobs in enumerate(WAVES, 1):
            run_wave(args, identities, wave_index, jobs)
        manifest = finalize(args, identities)
        atomic_json(
            args.run_root / 'queue_state.json',
            {
                'state': 'complete',
                'finished_at_utc': utc_now(),
                'protocol_sha256': PARENT_PROTOCOL_SHA256,
                'ant_protocol_sha256': ANT_PROTOCOL_SHA256,
                'point_core_terminal_state': point_core_state,
                'scientific_training_launched': True,
                'final_manifest': str(manifest),
                'final_manifest_sha256': sha256_file(manifest),
            },
        )
    except Exception as error:
        atomic_json(
            args.run_root / 'queue_state.json',
            {
                'state': 'failed',
                'failed_at_utc': utc_now(),
                'protocol_sha256': PARENT_PROTOCOL_SHA256,
                'ant_protocol_sha256': ANT_PROTOCOL_SHA256,
                'engineering_failure': True,
                'error': str(error),
            },
        )
        raise


if __name__ == '__main__':
    main()
