#!/usr/bin/env python3
"""Fail-closed remote process recovery for the frozen SHAPO P4.7 matrix.

This module is sent over SSH and executed by the host-side recovery watcher.  It
never chooses a checkpoint or scientific configuration: the sealed formal
launcher remains the only component that resolves ``latest_full.pt`` and starts
training.  The agent only restores missing, exact process commands after proving
that no formal worker, failure seal, or durable alert makes a restart ambiguous.
"""

from __future__ import annotations

import base64
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FULL_SHA256 = re.compile(r'^[0-9a-f]{64}$')
FULL_COMMIT = re.compile(r'^[0-9a-f]{40}$')
VERSION = 'shapo-p47-remote-recovery-v1'
DEFAULT_PROC_ROOT = Path('/proc')
EXPECTED_WORKTREE_KEYS = {
    'formal_source',
    'monitor_source',
    'milestone_source',
    'validation_source',
}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    """Atomically replace one small text file and fsync its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_json(path: Path, payload: Any) -> None:
    """Atomically write strict JSON."""
    atomic_text(
        path,
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
    )


def append_jsonl(path: Path, payload: Any) -> None:
    """Append and fsync one compact audit row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def require_absolute(path: str, label: str, *, allow_root: bool = False) -> Path:
    """Validate one explicit remote path without following it yet."""
    candidate = Path(path)
    if not candidate.is_absolute() or '..' in candidate.parts:
        raise ValueError(f'{label} must be an absolute path without parent traversal')
    if not allow_root and candidate == Path('/'):
        raise ValueError(f'{label} cannot be the filesystem root')
    return candidate


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete immutable recovery contract."""
    if payload.get('version') != VERSION:
        raise ValueError('remote recovery contract version changed')
    if not isinstance(payload.get('allow_recovery'), bool):
        raise ValueError('allow_recovery must be boolean')
    for key in (
        'python_bin',
        'formal_source',
        'run_root',
        'smoke_root',
        'monitor_source',
        'monitor_output',
        'milestone_source',
        'milestone_output',
        'validation_source',
        'completion_state',
        'recovery_root',
    ):
        require_absolute(str(payload.get(key, '')), key)
    run_root = Path(payload['run_root'])
    recovery_root = Path(payload['recovery_root'])
    if run_root == recovery_root or run_root in recovery_root.parents:
        raise ValueError('recovery state must not be inside the formal run root')
    if recovery_root == Path('/root/autodl-tmp/SAFE'):
        raise ValueError('recovery state path is too broad')
    commits = payload.get('worktree_commits')
    expected_worktrees = {payload[key] for key in EXPECTED_WORKTREE_KEYS}
    if not isinstance(commits, dict) or set(commits) != expected_worktrees:
        raise ValueError('exactly four expected worktree commit locks are required')
    for path, commit in commits.items():
        require_absolute(str(path), 'worktree')
        if not FULL_COMMIT.fullmatch(str(commit)):
            raise ValueError(f'invalid worktree commit lock for {path!r}')
    hashes = payload.get('file_sha256')
    formal = payload['formal_source']
    monitor = payload['monitor_source']
    milestone = payload['milestone_source']
    validation = payload['validation_source']
    expected_files = {
        payload['python_bin'],
        f'{formal}/scripts/star/launch_shapo_p47_formal.sh',
        f'{formal}/configs/star/shapo_p47_four_env.yaml',
        f"{payload['smoke_root']}/smoke_gate.json",
        f'{monitor}/scripts/star/watch_shapo_p47_formal.py',
        f'{monitor}/scripts/star/status_shapo_p47_formal.py',
        f'{milestone}/scripts/star/watch_shapo_p47_milestones.py',
        f'{milestone}/scripts/star/audit_shapo_p47_milestone.py',
        f'{milestone}/scripts/star/validate_shapo_p47_run.py',
        f'{validation}/scripts/star/watch_shapo_p47_completion.py',
        f'{validation}/scripts/star/finalize_shapo_p47_formal.sh',
        f'{validation}/scripts/star/finalize_shapo_p47_formal.py',
        f'{validation}/configs/star/shapo_p47_four_env.yaml',
        f'{validation}/scripts/star/audit_shapo_p47_milestone.py',
        f'{validation}/scripts/star/validate_shapo_p47_run.py',
    }
    if not isinstance(hashes, dict) or set(hashes) != expected_files:
        raise ValueError('exactly fifteen expected file SHA-256 locks are required')
    for path, digest in hashes.items():
        require_absolute(str(path), 'locked file')
        if not FULL_SHA256.fullmatch(str(digest)):
            raise ValueError(f'invalid file digest lock for {path!r}')
    config_hashes = payload.get('resolved_config_sha256')
    if not isinstance(config_hashes, dict) or set(config_hashes) != {
        'point-goal',
        'point-button',
        'ant',
        'walker2d',
    }:
        raise ValueError('exactly four resolved formal config identities are required')
    if any(not FULL_SHA256.fullmatch(str(digest)) for digest in config_hashes.values()):
        raise ValueError('resolved formal config digest is invalid')
    if (
        not isinstance(payload.get('expected_torch_version'), str)
        or not payload['expected_torch_version']
    ):
        raise ValueError('expected Torch version is required')
    if (
        not isinstance(payload.get('expected_cuda_device'), str)
        or not payload['expected_cuda_device']
    ):
        raise ValueError('expected CUDA device identity is required')
    minimum_free = payload.get('minimum_free_bytes')
    if not isinstance(minimum_free, int) or minimum_free < 0:
        raise ValueError('minimum free-byte threshold is invalid')
    return payload


def git_output(worktree: Path, *arguments: str) -> str:
    """Run one read-only Git command against a locked worktree."""
    completed = subprocess.run(
        ['git', '-C', str(worktree), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f'git {arguments[0]} failed for {worktree}: {completed.stderr.strip()}',
        )
    return completed.stdout.strip()


def validate_immutable_inputs(payload: dict[str, Any]) -> None:
    """Prove all remote code and protocol identities before considering recovery."""
    python_bin = Path(payload['python_bin'])
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise ValueError('locked remote Python interpreter is unavailable')
    for raw_path, expected in payload['worktree_commits'].items():
        path = Path(raw_path)
        if git_output(path, 'rev-parse', 'HEAD') != expected:
            raise ValueError(f'locked worktree commit changed: {path}')
        if git_output(path, 'status', '--porcelain'):
            raise ValueError(f'locked worktree became dirty: {path}')
    for raw_path, expected in payload['file_sha256'].items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f'locked recovery dependency changed: {path}')
    formal_commit = payload['worktree_commits'][payload['formal_source']]
    runs_root = Path(payload['run_root']) / 'runs'
    if runs_root.is_dir():
        for identity in runs_root.glob('*/source_commit.txt'):
            if identity.read_text(encoding='utf-8').strip() != formal_commit:
                raise ValueError(f'formal run source identity changed: {identity.parent}')
        for slug, expected in payload['resolved_config_sha256'].items():
            for config in runs_root.glob(f'{slug}-*/resolved_formal_config.yaml'):
                if sha256_file(config) != expected:
                    raise ValueError(f'resolved formal config identity changed: {config}')


def validate_restart_environment(payload: dict[str, Any]) -> dict[str, Any]:
    """Require the frozen compute software/hardware and disk floor before training restart."""
    import torch  # pylint: disable=import-outside-toplevel

    actual_torch = str(torch.__version__)
    if actual_torch != payload['expected_torch_version']:
        raise RuntimeError(
            f'Torch version changed from {payload["expected_torch_version"]!r} '
            f'to {actual_torch!r}',
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one CUDA device is required for formal recovery')
    actual_device = torch.cuda.get_device_name(0)
    if actual_device != payload['expected_cuda_device']:
        raise RuntimeError(
            f'CUDA device changed from {payload["expected_cuda_device"]!r} '
            f'to {actual_device!r}',
        )
    free_bytes = shutil.disk_usage(payload['run_root']).free
    if free_bytes < payload['minimum_free_bytes']:
        raise RuntimeError(
            f'insufficient server disk for formal recovery: {free_bytes} bytes free',
        )
    return {
        'torch_version': actual_torch,
        'cuda_device': actual_device,
        'cuda_device_count': torch.cuda.device_count(),
        'filesystem_free_bytes': free_bytes,
        'minimum_free_bytes': payload['minimum_free_bytes'],
    }


def process_argv(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> list[str] | None:
    """Read one live process argument vector, or return ``None`` if it vanished."""
    try:
        data = (proc_root / str(pid) / 'cmdline').read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return [part.decode('utf-8', errors='replace') for part in data.split(b'\0') if part]


def process_ppid(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> int | None:
    """Read one process's parent PID from procfs."""
    try:
        for line in (proc_root / str(pid) / 'status').read_text(encoding='utf-8').splitlines():
            if line.startswith('PPid:'):
                return int(line.split(':', 1)[1].strip())
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return None


def process_cwd(pid: int, proc_root: Path = DEFAULT_PROC_ROOT) -> Path | None:
    """Resolve one process working directory from procfs."""
    try:
        return (proc_root / str(pid) / 'cwd').resolve(strict=True)
    except (FileNotFoundError, PermissionError, ProcessLookupError, RuntimeError):
        return None


def proc_pids(proc_root: Path = DEFAULT_PROC_ROOT) -> list[int]:
    """Return the numeric process identifiers visible in procfs."""
    return sorted(int(path.name) for path in proc_root.iterdir() if path.name.isdigit())


def matching_pids(
    expected: list[str],
    proc_root: Path = DEFAULT_PROC_ROOT,
    *,
    expected_cwd: Path | None = None,
) -> list[int]:
    """Find exact command-line and optional working-directory matches."""
    return [
        pid
        for pid in proc_pids(proc_root)
        if process_argv(pid, proc_root) == expected
        and (expected_cwd is None or process_cwd(pid, proc_root) == expected_cwd)
    ]


def read_pid(path: Path) -> int | None:
    """Read a positive PID from a file without trusting its process identity."""
    try:
        value = int(path.read_text(encoding='utf-8').strip())
    except (FileNotFoundError, ValueError):
        return None
    return value if value > 0 else None


def resolve_exact_process(
    *,
    name: str,
    expected: list[str],
    expected_cwd: Path | None,
    pid_file: Path,
    supervisor: bool = False,
    repair_pid_file: bool = True,
    proc_root: Path = DEFAULT_PROC_ROOT,
) -> dict[str, Any]:
    """Resolve one exact singleton process, adopting an unambiguous stale PID file."""
    matches = matching_pids(expected, proc_root, expected_cwd=expected_cwd)
    recorded = read_pid(pid_file)
    candidates = (
        [pid for pid in matches if process_ppid(pid, proc_root) == 1] if supervisor else matches
    )
    if len(candidates) > 1:
        raise RuntimeError(f'ambiguous exact {name} processes: {candidates}')
    if recorded in candidates:
        return {'alive': True, 'pid': recorded, 'adopted_pid': False, 'matches': matches}
    if len(candidates) == 1:
        if repair_pid_file:
            atomic_text(pid_file, f'{candidates[0]}\n')
        return {
            'alive': True,
            'pid': candidates[0],
            'adopted_pid': repair_pid_file,
            'pid_file_repair_needed': not repair_pid_file,
            'matches': matches,
        }
    return {'alive': False, 'pid': None, 'adopted_pid': False, 'matches': matches}


def is_within(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is below ``parent`` without requiring existence."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def formal_worker_processes(
    run_root: Path,
    proc_root: Path = DEFAULT_PROC_ROOT,
) -> dict[str, int]:
    """Map formal run containers to their exact live training worker."""
    workers: dict[str, int] = {}
    configs_root = run_root / 'runs'
    for pid in proc_pids(proc_root):
        argv = process_argv(pid, proc_root) or []
        if not any(argument.endswith('scripts/star/train_star.py') for argument in argv):
            continue
        try:
            config = Path(argv[argv.index('--config') + 1])
        except (ValueError, IndexError):
            continue
        if config.is_absolute() and is_within(config, configs_root):
            container = config.parent.name
            if container in workers:
                raise RuntimeError(
                    f'ambiguous formal workers for run {container!r}: '
                    f'{workers[container]} and {pid}',
                )
            workers[container] = pid
    return workers


def formal_worker_pids(run_root: Path, proc_root: Path = DEFAULT_PROC_ROOT) -> list[int]:
    """Find only train workers whose explicit config belongs to the formal root."""
    return sorted(formal_worker_processes(run_root, proc_root).values())


def process_specs(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Construct the only process commands the recovery transaction may launch."""
    python_bin = payload['python_bin']
    formal = payload['formal_source']
    run_root = payload['run_root']
    monitor = payload['monitor_source']
    milestone = payload['milestone_source']
    validation = payload['validation_source']
    return {
        'supervisor': {
            'argv': [
                'bash',
                f'{formal}/scripts/star/launch_shapo_p47_formal.sh',
                formal,
                run_root,
                payload['smoke_root'],
                python_bin,
            ],
            'cwd': '/root',
            'pid_file': f'{run_root}/formal_supervisor.pid',
            'log': f'{run_root}/formal_supervisor.log',
            'supervisor': True,
            'completion_marker': f'{run_root}/formal_matrix_complete',
        },
        'watchdog': {
            'argv': [
                python_bin,
                'scripts/star/watch_shapo_p47_formal.py',
                '--run-root',
                run_root,
                '--matrix',
                f'{formal}/configs/star/shapo_p47_four_env.yaml',
                '--output-dir',
                payload['monitor_output'],
                '--supervisor-pid-file',
                f'{run_root}/formal_supervisor.pid',
                '--interval-seconds',
                '600',
                '--stale-after-seconds',
                '1800',
                '--min-free-gib',
                '4',
            ],
            'cwd': monitor,
            'pid_file': f"{payload['monitor_output']}/watchdog.pid",
            'log': f"{payload['monitor_output']}/watchdog.log",
            'supervisor': False,
            'completion_marker': f"{payload['monitor_output']}/completion_health.json",
        },
        'milestone': {
            'argv': [
                python_bin,
                'scripts/star/watch_shapo_p47_milestones.py',
                '--run-root',
                run_root,
                '--matrix',
                f'{milestone}/configs/star/shapo_p47_four_env.yaml',
                '--source-commit',
                payload['worktree_commits'][formal],
                '--output-root',
                payload['milestone_output'],
                '--python-bin',
                python_bin,
                '--stable-seconds',
                '120',
                '--poll-seconds',
                '600',
            ],
            'cwd': milestone,
            'pid_file': f"{payload['milestone_output']}/milestone_watch.pid",
            'log': f"{payload['milestone_output']}/milestone_watch.log",
            'supervisor': False,
            'completion_marker': f"{payload['milestone_output']}/milestone_watch_complete",
        },
        'completion': {
            'argv': [
                python_bin,
                'scripts/star/watch_shapo_p47_completion.py',
                '--run-root',
                run_root,
                '--milestone-root',
                payload['milestone_output'],
                '--watchdog-root',
                payload['monitor_output'],
                '--matrix',
                'configs/star/shapo_p47_four_env.yaml',
                '--source-commit',
                payload['worktree_commits'][formal],
                '--validation-source',
                validation,
                '--python-bin',
                python_bin,
                '--output-root',
                f'{run_root}/public_artifacts',
                '--state-dir',
                payload['completion_state'],
                '--audit-script',
                'scripts/star/audit_shapo_p47_milestone.py',
                '--finalizer-script',
                'scripts/star/finalize_shapo_p47_formal.sh',
                '--poll-seconds',
                '600',
            ],
            'cwd': validation,
            'pid_file': f"{payload['completion_state']}.pid",
            'log': f"{payload['completion_state']}/completion_watch.log",
            'supervisor': False,
            'completion_marker': (f"{payload['completion_state']}/completion_watch_complete.json"),
        },
    }


def durable_alerts(payload: dict[str, Any]) -> list[str]:
    """Return formal alert files that make automatic process recovery unsafe."""
    candidates = [
        Path(payload['monitor_output']) / 'alert.json',
        Path(payload['milestone_output']) / 'milestone_watch_alert.json',
        Path(payload['completion_state']) / 'completion_watch_alert.json',
    ]
    return [str(path) for path in candidates if path.is_file()]


def matrix_failure_count(run_root: Path) -> int | None:
    """Read the supervisor's terminal failure seal when present."""
    path = run_root / 'matrix_failure_count.txt'
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding='utf-8').strip())
    except ValueError as error:
        raise ValueError('formal matrix failure seal is invalid') from error


def formal_failure_artifacts(run_root: Path) -> list[str]:
    """Return explicit per-run failures that must never be hidden by restart."""
    failures = []
    runs_root = run_root / 'runs'
    if not runs_root.is_dir():
        return failures
    for container in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        exit_path = container / 'exit_code.txt'
        if exit_path.is_file():
            try:
                exit_code = int(exit_path.read_text(encoding='utf-8').strip())
            except ValueError as error:
                raise ValueError(f'invalid formal exit-code seal: {exit_path}') from error
            if exit_code != 0:
                failures.append(f'{exit_path}:exit={exit_code}')
        for pattern in (
            'training/**/artifacts/engineering_hard_stop.json',
            'training/**/artifacts/numerical_hard_stop.json',
        ):
            failures.extend(str(path) for path in sorted(container.glob(pattern)))
    return failures


def recoverable_checkpoints(
    run_root: Path,
    *,
    require_started: bool = False,
) -> dict[str, dict[str, Any]]:
    """Inventory full checkpoints, optionally requiring one for every started run."""
    result = {}
    runs_root = run_root / 'runs'
    if not runs_root.is_dir():
        return result
    for container in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        progress = list(container.glob('training/**/progress.csv'))
        checkpoints = list(container.glob('training/**/checkpoints/latest_full.pt'))
        if require_started and progress and not checkpoints:
            raise RuntimeError(f'started run lacks latest_full recovery: {container.name}')
        if checkpoints:
            newest = max(checkpoints, key=lambda path: path.stat().st_mtime_ns)
            if newest.stat().st_size <= 0:
                raise RuntimeError(f'empty latest_full recovery: {newest}')
            result[container.name] = {
                'path': str(newest),
                'bytes': newest.stat().st_size,
                'mtime_ns': newest.stat().st_mtime_ns,
            }
    return result


def latest_progress_step(container_root: Path) -> int:
    """Return the greatest finite recorded step across all recovery segments."""
    steps = []
    for path in container_root.glob('training/**/progress.csv'):
        try:
            with path.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                continue
            value = float(rows[-1]['TotalEnvSteps'])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f'formal progress tail is invalid: {path}') from error
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise RuntimeError(f'formal progress tail step is invalid: {path}')
        steps.append(int(value))
    if not steps:
        raise RuntimeError(f'formal run has no progress evidence: {container_root}')
    return max(steps)


def nested_subset(observed: Any, expected: Any, *, label: str) -> None:
    """Require recursively equal frozen config leaves while permitting runtime extras."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise RuntimeError(f'{label} must be a mapping')
        for key, value in expected.items():
            if key not in observed:
                raise RuntimeError(f'{label} is missing frozen key {key!r}')
            nested_subset(observed[key], value, label=f'{label}.{key}')
        return
    if observed != expected:
        raise RuntimeError(f'{label} changed from {expected!r} to {observed!r}')


def validate_checkpoint_payloads(
    run_root: Path,
    checkpoints: dict[str, dict[str, Any]],
    matrix_path: Path,
) -> None:
    """Load every restart checkpoint on CPU and verify complete STAR recovery state."""
    import torch  # pylint: disable=import-outside-toplevel

    required = {
        'format',
        'version',
        'model',
        'optimizers',
        'duals',
        'normalizer',
        'counters',
        'warmup',
        'rng',
        'replay',
        'extra',
    }
    try:
        import yaml  # pylint: disable=import-outside-toplevel

        matrix = yaml.safe_load(matrix_path.read_text(encoding='utf-8'))
    except Exception as error:
        raise RuntimeError('locked formal matrix cannot be loaded') from error
    if not isinstance(matrix, dict) or not isinstance(matrix.get('environments'), dict):
        raise RuntimeError('locked formal matrix is invalid')
    seeds = {int(seed) for seed in matrix.get('training_seeds', [])}
    slug_to_environment = {
        'point-goal': 'SafetyPointGoal1-v0',
        'point-button': 'SafetyPointButton1-v0',
        'ant': 'Ant-v4',
        'walker2d': 'Walker2d-v4',
    }
    for container, identity in checkpoints.items():
        path = Path(identity['path'])
        container_root = run_root / 'runs' / container
        try:
            resolved = yaml.safe_load(
                (container_root / 'resolved_formal_config.yaml').read_text(encoding='utf-8'),
            )
        except Exception as error:
            raise RuntimeError(f'formal resolved config cannot be loaded: {container}') from error
        if not isinstance(resolved, dict) or not isinstance(resolved.get('custom_cfgs'), dict):
            raise RuntimeError(f'formal resolved config is invalid: {container}')
        try:
            payload = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as error:
            raise RuntimeError(f'latest_full checkpoint cannot be loaded: {path}') from error
        if not isinstance(payload, dict) or set(payload) != required:
            raise RuntimeError(f'latest_full checkpoint fields changed: {path}')
        if payload.get('format') != 'star-checkpoint' or payload.get('version') != 1:
            raise RuntimeError(f'latest_full checkpoint format changed: {path}')
        if payload.get('replay') is None:
            raise RuntimeError(f'latest_full checkpoint lacks replay recovery: {path}')
        counters = payload.get('counters')
        checkpoint_steps = int(counters.get('env_steps', 0)) if isinstance(counters, dict) else 0
        if not isinstance(counters, dict) or checkpoint_steps <= 0:
            raise RuntimeError(f'latest_full checkpoint counters are invalid: {path}')
        extra = payload.get('extra')
        if not isinstance(extra, dict) or not isinstance(extra.get('resolved_config'), dict):
            raise RuntimeError(f'latest_full checkpoint configuration is invalid: {path}')
        try:
            slug, seed_text = container.rsplit('-', 1)
            expected_seed = int(seed_text)
        except (IndexError, ValueError) as error:
            raise RuntimeError(f'formal run container identity is invalid: {container}') from error
        expected_environment = slug_to_environment.get(slug, '')
        if expected_environment not in matrix['environments'] or expected_seed not in seeds:
            raise RuntimeError(f'formal run is outside the locked matrix: {container}')
        total_steps = int(matrix['environments'][expected_environment]['total_steps'])
        frequency = int(matrix.get('checkpoint', {}).get('latest_full_frequency_steps', 0))
        progress_steps = latest_progress_step(container_root)
        if (
            checkpoint_steps > total_steps
            or frequency <= 0
            or checkpoint_steps % frequency != 0
            or checkpoint_steps > progress_steps
            or progress_steps - checkpoint_steps > frequency
        ):
            raise RuntimeError(f'latest_full checkpoint step is invalid: {path}')
        if (
            resolved.get('env_id') != expected_environment
            or extra.get('environment') != expected_environment
            or int(extra.get('seed', -1)) != expected_seed
            or extra['resolved_config'].get('env_id') != expected_environment
            or int(extra['resolved_config'].get('seed', -1)) != expected_seed
        ):
            raise RuntimeError(f'latest_full checkpoint belongs to a different run: {path}')
        expected_custom = resolved['custom_cfgs']
        observed_custom = extra['resolved_config']
        expected_resume = expected_custom.get('algo_cfgs', {}).get('resume_from', '')
        observed_resume = observed_custom.get('algo_cfgs', {}).get('resume_from', '')
        if observed_resume != expected_resume:
            allowed = {
                str(candidate.resolve())
                for candidate in container_root.glob('training/**/checkpoints/latest_full.pt')
            }
            if expected_resume != '' or observed_resume not in allowed:
                raise RuntimeError(f'latest_full checkpoint resume path escaped its run: {path}')
            expected_custom = {
                **expected_custom,
                'algo_cfgs': {
                    **expected_custom['algo_cfgs'],
                    'resume_from': observed_resume,
                },
            }
        nested_subset(observed_custom, expected_custom, label=f'{container}.resolved_config')
        replay = payload['replay']
        expected_capacity = int(matrix.get('training_invariants', {}).get('replay_size', 0))
        if (
            not isinstance(replay, dict)
            or int(replay.get('capacity', -1)) != expected_capacity
            or int(replay.get('size', 0)) != min(checkpoint_steps, expected_capacity)
        ):
            raise RuntimeError(f'latest_full checkpoint replay state is invalid: {path}')
        identity['validated_format'] = 'star-checkpoint-v1'
        identity['checkpoint_env_steps'] = checkpoint_steps
        del payload
        gc.collect()


def launch_exact(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Detach one exact process and atomically publish its PID."""
    cwd = Path(spec['cwd'])
    log_path = Path(spec['log'])
    pid_file = Path(spec['pid_file'])
    if not cwd.is_dir():
        raise RuntimeError(f'{name} working directory is unavailable: {cwd}')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('ab', buffering=0) as log:
        process = subprocess.Popen(
            spec['argv'],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    atomic_text(pid_file, f'{process.pid}\n')
    time.sleep(1.0)
    alive = process_argv(process.pid) == spec['argv'] and process_cwd(process.pid) == cwd.resolve()
    if alive:
        return {'pid': process.pid, 'alive': True, 'completed': False}
    exit_code = process.poll()
    completed = Path(spec['completion_marker']).is_file() and exit_code == 0
    if completed:
        return {'pid': process.pid, 'alive': False, 'completed': True}
    raise RuntimeError(
        f'launched {name} process did not retain its exact identity; exit={exit_code}',
    )


def last_training_attempt(path: Path) -> dict[str, Any] | None:
    """Read the most recent durable automatic supervisor-restart intent."""
    if not path.is_file():
        return None
    result = None
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f'invalid recovery journal line {line_number}') from error
        if not isinstance(row, dict):
            raise RuntimeError(f'invalid recovery journal record on line {line_number}')
        if row.get('formal_training_process_restart_planned') is True:
            result = row
    return result


def startup_grace_active(
    previous: dict[str, Any] | None,
    *,
    observed_at: datetime,
    grace_seconds: float = 900.0,
) -> bool:
    """Return whether a recorded supervisor restart is still in its bounded startup grace."""
    if previous is None or previous.get('formal_training_process_restart_planned') is not True:
        return False
    try:
        started = datetime.fromisoformat(str(previous['observed_at_utc']))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError('previous supervisor recovery timestamp is invalid') from error
    if started.tzinfo is None:
        raise RuntimeError('previous supervisor recovery timestamp lacks a timezone')
    age = (observed_at - started.astimezone(timezone.utc)).total_seconds()
    if age < 0.0:
        raise RuntimeError('previous supervisor recovery timestamp is in the future')
    return age <= grace_seconds


def completed_checkpoint_runs(
    run_root: Path,
    checkpoint_runs: set[str],
) -> set[str]:
    """Identify checkpoint-bearing runs already sealed by strict formal validation."""
    completed = set()
    for name in checkpoint_runs:
        path = run_root / 'runs' / name / 'formal_validation.json'
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as error:
            raise RuntimeError(f'formal validation receipt is invalid: {path}') from error
        if not isinstance(payload, dict):
            raise RuntimeError(f'formal validation receipt is not an object: {path}')
        if payload.get('status') == 'passed':
            exit_path = path.parent / 'exit_code.txt'
            try:
                exit_code = int(exit_path.read_text(encoding='utf-8').strip())
            except (FileNotFoundError, ValueError) as error:
                raise RuntimeError(
                    f'completed formal validation lacks a valid exit seal: {path.parent}',
                ) from error
            if exit_code != 0:
                raise RuntimeError(
                    f'completed formal validation has a nonzero exit seal: {path.parent}',
                )
            completed.add(name)
    return completed


def watchdog_startup_readiness(
    run_root: Path,
    checkpoints: dict[str, dict[str, Any]],
    workers: dict[str, int],
    matrix_path: Path,
) -> dict[str, Any]:
    """Require the first unfinished seed wave to reattach before watchdog launch."""
    checkpoint_runs = set(checkpoints)
    try:
        import yaml  # pylint: disable=import-outside-toplevel

        matrix = yaml.safe_load(matrix_path.read_text(encoding='utf-8'))
        seeds = [int(seed) for seed in matrix['training_seeds']]
        environments = list(matrix['environments'])
    except Exception as error:
        raise RuntimeError('locked formal matrix cannot define startup readiness') from error
    slugs = {
        'SafetyPointGoal1-v0': 'point-goal',
        'SafetyPointButton1-v0': 'point-button',
        'Ant-v4': 'ant',
        'Walker2d-v4': 'walker2d',
    }
    if set(environments) != set(slugs) or len(seeds) != 5 or len(set(seeds)) != 5:
        raise RuntimeError('locked formal matrix wave identity changed')
    expected_by_seed = {
        seed: {f'{slugs[environment]}-{seed}' for environment in environments} for seed in seeds
    }
    completed_runs = set()
    for seed_runs in expected_by_seed.values():
        completed_runs.update(completed_checkpoint_runs(run_root, seed_runs))
    pending_wave_seed = next(
        (seed for seed in seeds if expected_by_seed[seed] - completed_runs),
        None,
    )
    required = (
        set() if pending_wave_seed is None else expected_by_seed[pending_wave_seed] - completed_runs
    )
    active = set(workers)
    return {
        'ready': required <= active,
        'checkpoint_runs': sorted(checkpoint_runs),
        'completed_checkpoint_runs': sorted(completed_runs),
        'pending_wave_seed': pending_wave_seed,
        'required_worker_runs': sorted(required),
        'active_worker_runs': sorted(active),
        'missing_worker_runs': sorted(required - active),
    }


def require_new_recovery_state(
    checkpoints: dict[str, dict[str, Any]],
    previous: dict[str, Any] | None,
) -> None:
    """Prevent a crash loop when no full checkpoint advanced after the last restart."""
    if previous is None:
        return
    prior = previous.get('recoverable_checkpoints')
    if not isinstance(prior, dict):
        raise RuntimeError('previous recovery action lacks checkpoint identities')
    current_steps = {
        name: int(identity.get('checkpoint_env_steps', -1))
        for name, identity in checkpoints.items()
    }
    prior_steps = {
        name: int(identity.get('checkpoint_env_steps', -1))
        for name, identity in prior.items()
        if isinstance(identity, dict)
    }
    regressed = [
        name
        for name, step in current_steps.items()
        if name in prior_steps and step < prior_steps[name]
    ]
    if regressed:
        raise RuntimeError(f'recovery checkpoint steps regressed: {regressed}')
    progressed = set(current_steps) > set(prior_steps) or any(
        current_steps.get(name, -1) > step for name, step in prior_steps.items()
    )
    if not progressed:
        raise RuntimeError('automatic supervisor recovery made no new checkpoint progress')


def record_recovery_attempt(
    *,
    recovery_root: Path,
    observed_at: str,
    needed: list[str],
    checkpoints: dict[str, dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Persist intent before any detached process is launched."""
    attempt = {
        'version': VERSION,
        'observed_at_utc': observed_at,
        'state': 'launch_planned',
        'needed_processes': list(needed),
        'recoverable_checkpoints': checkpoints,
        'formal_training_process_restart_planned': 'supervisor' in needed,
        'worktree_commits': payload['worktree_commits'],
        'file_sha256': payload['file_sha256'],
        'scientific_protocol_changed': False,
        'training_configuration_changed': False,
        'used_for_checkpoint_or_configuration_selection': False,
    }
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    atomic_json(recovery_root / f'attempt-{stamp}-{os.getpid()}.json', attempt)
    append_jsonl(recovery_root / 'recovery_attempts.jsonl', attempt)
    return attempt


def _run_locked(payload: dict[str, Any], recovery_root: Path) -> dict[str, Any]:
    """Inspect and restore processes while holding the remote transaction lock."""
    observed_at = datetime.now(timezone.utc).isoformat()
    run_root = Path(payload['run_root'])
    alerts = durable_alerts(payload)
    failure_count = matrix_failure_count(run_root)
    failure_artifacts = formal_failure_artifacts(run_root)
    if alerts:
        raise RuntimeError(f'durable formal alert prevents automatic recovery: {alerts}')
    if failure_count not in (None, 0):
        raise RuntimeError(f'formal matrix failure seal prevents recovery: {failure_count}')
    if failure_artifacts:
        raise RuntimeError(f'formal run failure artifacts prevent recovery: {failure_artifacts}')

    specs = process_specs(payload)
    processes = {}
    adopted = []
    for name, spec in specs.items():
        state = resolve_exact_process(
            name=name,
            expected=spec['argv'],
            expected_cwd=Path(spec['cwd']).resolve(),
            pid_file=Path(spec['pid_file']),
            supervisor=bool(spec['supervisor']),
            repair_pid_file=bool(payload['allow_recovery']),
        )
        processes[name] = state
        if state['adopted_pid']:
            adopted.append(name)

    worker_processes = formal_worker_processes(run_root)
    worker_pids = sorted(worker_processes.values())
    matrix_complete = (run_root / 'formal_matrix_complete').is_file()
    watchdog_complete = (Path(payload['monitor_output']) / 'completion_health.json').is_file()
    milestone_complete = (Path(payload['milestone_output']) / 'milestone_watch_complete').is_file()
    final_complete = (run_root / 'public_artifacts' / 'finalization_complete').is_file()
    completion_complete = (
        Path(payload['completion_state']) / 'completion_watch_complete.json'
    ).is_file()
    supervisor_restart_required = not matrix_complete and not processes['supervisor']['alive']
    checkpoints = recoverable_checkpoints(
        run_root,
        require_started=supervisor_restart_required,
    )
    restart_environment = None

    if not processes['supervisor']['alive'] and worker_pids:
        raise RuntimeError(
            f'formal workers are alive without the recorded exact supervisor: {worker_pids}',
        )
    if (
        not matrix_complete
        and not processes['supervisor']['alive']
        and processes['watchdog']['alive']
    ):
        raise RuntimeError('formal watchdog is alive without the exact supervisor')

    previous_training_attempt = last_training_attempt(
        recovery_root / 'recovery_attempts.jsonl',
    )
    if supervisor_restart_required:
        restart_environment = validate_restart_environment(payload)
        validate_checkpoint_payloads(
            run_root,
            checkpoints,
            Path(payload['formal_source']) / 'configs/star/shapo_p47_four_env.yaml',
        )
        require_new_recovery_state(
            checkpoints,
            previous_training_attempt,
        )

    needed = []
    if supervisor_restart_required:
        needed.append('supervisor')
    if not watchdog_complete and not processes['watchdog']['alive']:
        needed.append('watchdog')
    if not milestone_complete and not processes['milestone']['alive']:
        needed.append('milestone')
    if not completion_complete and not processes['completion']['alive']:
        needed.append('completion')

    launch_now = list(needed)
    deferred = []
    watchdog_readiness = None
    if payload['allow_recovery'] and 'watchdog' in launch_now and not matrix_complete:
        if 'supervisor' in launch_now:
            launch_now.remove('watchdog')
            deferred.append('watchdog')
            watchdog_readiness = {
                'ready': False,
                'reason': 'supervisor_launch_transaction',
            }
        else:
            watchdog_readiness = watchdog_startup_readiness(
                run_root,
                checkpoints,
                worker_processes,
                Path(payload['formal_source']) / 'configs/star/shapo_p47_four_env.yaml',
            )
            if not watchdog_readiness['ready']:
                if not startup_grace_active(
                    previous_training_attempt,
                    observed_at=datetime.fromisoformat(observed_at),
                ):
                    raise RuntimeError(
                        'formal workers did not reattach during the bounded startup grace: '
                        f'{watchdog_readiness["missing_worker_runs"]}',
                    )
                launch_now.remove('watchdog')
                deferred.append('watchdog')

    launched: dict[str, int] = {}
    attempt = None
    if launch_now and payload['allow_recovery']:
        attempt = record_recovery_attempt(
            recovery_root=recovery_root,
            observed_at=observed_at,
            needed=launch_now,
            checkpoints=checkpoints,
            payload=payload,
        )
        for name in ('supervisor', 'watchdog', 'milestone', 'completion'):
            if name in launch_now:
                launch = launch_exact(name, specs[name])
                launched[name] = int(launch['pid'])
                processes[name] = {
                    'alive': bool(launch['alive']),
                    'pid': launched[name],
                    'adopted_pid': False,
                    'matches': [launched[name]],
                    'completed_during_launch': bool(launch['completed']),
                }

    state = 'complete' if final_complete and completion_complete else 'running'
    if needed and not payload['allow_recovery']:
        state = 'recovery_required'
    elif launched:
        state = 'recovered'
    elif deferred:
        state = 'recovery_deferred'
    result = {
        'version': VERSION,
        'observed_at_utc': observed_at,
        'state': state,
        'allow_recovery': payload['allow_recovery'],
        'needed_processes': needed,
        'deferred_processes': deferred,
        'launched_processes': launched,
        'adopted_pid_files': adopted,
        'processes': {
            name: {'alive': row['alive'], 'pid': row['pid']} for name, row in processes.items()
        },
        'formal_worker_pids': worker_pids,
        'formal_worker_processes': worker_processes,
        'formal_matrix_complete': matrix_complete,
        'watchdog_complete': watchdog_complete,
        'milestone_complete': milestone_complete,
        'completion_complete': completion_complete,
        'finalization_complete': final_complete,
        'matrix_failure_count': failure_count,
        'formal_failure_artifacts': failure_artifacts,
        'restart_environment': restart_environment,
        'watchdog_startup_readiness': watchdog_readiness,
        'recoverable_checkpoint_runs': len(checkpoints),
        'recoverable_checkpoints': checkpoints,
        'scientific_protocol_changed': False,
        'training_configuration_changed': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'engineering_process_recovery_performed': bool(launched),
        'formal_training_process_restarted': 'supervisor' in launched,
        'recovery_attempt': attempt,
    }
    if launched or adopted:
        action = {
            **result,
            'worktree_commits': payload['worktree_commits'],
            'file_sha256': payload['file_sha256'],
        }
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        atomic_json(recovery_root / f'action-{stamp}-{os.getpid()}.json', action)
        append_jsonl(recovery_root / 'recovery_actions.jsonl', action)
    return result


def run_once(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one serialized fail-closed process recovery transaction."""
    payload = validate_payload(payload)
    recovery_root = Path(payload['recovery_root'])
    if not payload['allow_recovery']:
        validate_immutable_inputs(payload)
        return _run_locked(payload, recovery_root)
    recovery_root.mkdir(parents=True, exist_ok=True)
    with (recovery_root / '.remote-recovery.lock').open('w', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                'version': VERSION,
                'observed_at_utc': datetime.now(timezone.utc).isoformat(),
                'state': 'busy',
                'scientific_protocol_changed': False,
                'training_configuration_changed': False,
                'used_for_checkpoint_or_configuration_selection': False,
                'engineering_process_recovery_performed': False,
            }
        validate_immutable_inputs(payload)
        return _run_locked(payload, recovery_root)


def decode_payload(value: str) -> dict[str, Any]:
    """Decode the URL-safe base64 JSON contract passed as one argv token."""
    try:
        raw = base64.urlsafe_b64decode(value.encode('ascii'))
        payload = json.loads(raw.decode('utf-8'))
    except ValueError as error:
        raise ValueError('invalid encoded remote recovery payload') from error
    if not isinstance(payload, dict):
        raise ValueError('remote recovery payload must be a JSON object')
    return payload


def main() -> None:
    """Run one locked remote recovery transaction and emit exactly one JSON record."""
    if len(sys.argv) != 2:
        raise SystemExit('usage: python - ENCODED_RECOVERY_PAYLOAD')
    try:
        result = run_once(decode_payload(sys.argv[1]))
    except Exception as error:  # preserve a machine-readable fail-closed response
        result = {
            'version': VERSION,
            'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'state': 'alert',
            'error_type': type(error).__name__,
            'error': str(error),
            'scientific_protocol_changed': False,
            'training_configuration_changed': False,
            'used_for_checkpoint_or_configuration_selection': False,
            'engineering_process_recovery_performed': False,
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        raise SystemExit(2) from error
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
