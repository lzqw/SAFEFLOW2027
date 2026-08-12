#!/usr/bin/env python3
"""Wait for every frozen SHAPO gate, then finalize the matrix exactly once."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.star.status_shapo_p47_formal import SLUGS
from scripts.star.watch_shapo_p47_milestones import (
    locked_jobs,
    milestone_audit_identity,
    validate_sealed_output,
)


CHECKSUM_EXCLUSIONS = {'checksums.sha256', 'finalization_complete'}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one immutable file."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    """Atomically replace one strict JSON receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    """Append and fsync one bounded completion journal record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON object."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path} is not a JSON object')
    return payload


def expected_counts(matrix: dict[str, Any]) -> dict[str, int]:
    """Derive final artifact counts directly from the frozen matrix."""
    seeds = len(matrix['training_seeds'])
    environments = list(matrix['environments'].values())
    runs = seeds * len(environments)
    milestones = seeds * sum(len(environment['milestones']) for environment in environments)
    checkpoints = seeds * sum(3 + len(environment['milestones']) for environment in environments)
    return {
        'formal_runs': runs,
        'milestone_jobs': milestones,
        'evaluation_artifact_count': milestones,
        'provenance_artifact_count': runs * 4,
        'trajectory_artifact_count': milestones,
        'checkpoint_artifact_count': checkpoints,
    }


def matrix_failure_count(run_root: Path) -> int | None:
    """Return the durable matrix failure count when the launcher has written it."""
    path = run_root / 'matrix_failure_count.txt'
    if not path.is_file():
        return None
    try:
        count = int(path.read_text(encoding='utf-8').strip())
    except ValueError as error:
        raise ValueError(f'invalid matrix failure count: {path}') from error
    if count < 0:
        raise ValueError(f'negative matrix failure count: {path}')
    return count


def validate_training_completion(
    run_root: Path,
    matrix: dict[str, Any],
) -> dict[str, Any]:
    """Require the launcher seal and every final strict run validation."""
    counts = expected_counts(matrix)
    failure_count = matrix_failure_count(run_root)
    if failure_count not in {None, 0}:
        raise RuntimeError(f'formal matrix records {failure_count} failed runs')
    if not (run_root / 'formal_matrix_complete').is_file():
        return {
            'ready': False,
            'formal_matrix_complete': False,
            'matrix_failure_count': failure_count,
            'validated_runs': 0,
            'expected_runs': counts['formal_runs'],
        }
    if failure_count != 0:
        raise RuntimeError('formal matrix completion seal lacks an explicit zero failure count')

    validated = 0
    for seed_value in matrix['training_seeds']:
        seed = int(seed_value)
        for environment, config in matrix['environments'].items():
            container = run_root / 'runs' / f'{SLUGS[environment]}-{seed}'
            validation_path = container / 'formal_validation.json'
            validation = load_json(validation_path)
            expected = {
                'status': 'passed',
                'environment': environment,
                'training_seed': seed,
                'training_seed_is_statistical_unit': True,
                'total_steps': int(config['total_steps']),
                'protocol_version': matrix['protocol_version'],
                'protocol_sha256': matrix['protocol_sha256'],
                'frozen_method_canonical_sha256': matrix['frozen_method_canonical_sha256'],
            }
            for field, value in expected.items():
                if validation.get(field) != value:
                    raise ValueError(
                        f'{validation_path} field {field!r} does not match completion',
                    )
            validated += 1
    if validated != counts['formal_runs']:
        raise ValueError('formal validation coverage does not match the frozen matrix')
    return {
        'ready': True,
        'formal_matrix_complete': True,
        'matrix_failure_count': failure_count,
        'validated_runs': validated,
        'expected_runs': counts['formal_runs'],
    }


def validate_milestone_completion(
    run_root: Path,
    milestone_root: Path,
    matrix_path: Path,
    matrix: dict[str, Any],
    source_commit: str,
    audit_script: Path,
) -> dict[str, Any]:
    """Require and independently revalidate every diagnostic milestone seal."""
    counts = expected_counts(matrix)
    alert = milestone_root / 'milestone_watch_alert.json'
    if alert.is_file():
        raise RuntimeError(f'milestone watcher recorded an alert: {load_json(alert)}')
    complete = milestone_root / 'milestone_watch_complete'
    latest_path = milestone_root / 'milestone_watch_latest.json'
    if not complete.is_file():
        sealed = 0
        if latest_path.is_file():
            latest = load_json(latest_path)
            sealed = int(latest.get('sealed_jobs', 0))
            for field in (
                'scientific_protocol_changed',
                'training_state_changed',
                'used_for_checkpoint_or_configuration_selection',
            ):
                if latest.get(field) is not False:
                    raise ValueError(f'milestone watcher changed protected field {field!r}')
        return {
            'ready': False,
            'watcher_complete': False,
            'sealed_jobs': sealed,
            'expected_jobs': counts['milestone_jobs'],
        }

    latest = load_json(latest_path)
    expected_latest = {
        'state': 'complete',
        'total_jobs': counts['milestone_jobs'],
        'sealed_jobs': counts['milestone_jobs'],
        'waiting_jobs': 0,
        'scientific_protocol_changed': False,
        'training_state_changed': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'source_commit': source_commit,
    }
    for field, value in expected_latest.items():
        if latest.get(field) != value:
            raise ValueError(f'milestone completion field {field!r} changed')

    jobs = locked_jobs(run_root, milestone_root, matrix)
    identity = milestone_audit_identity(matrix_path, source_commit, audit_script)
    for job in jobs:
        validate_sealed_output(job, expected_identity=identity)
    return {
        'ready': True,
        'watcher_complete': True,
        'sealed_jobs': len(jobs),
        'expected_jobs': counts['milestone_jobs'],
    }


def validate_watchdog_completion(
    watchdog_root: Path,
    matrix: dict[str, Any],
) -> dict[str, Any]:
    """Require the independent runtime watchdog's clean completion receipt."""
    counts = expected_counts(matrix)
    alert_path = watchdog_root / 'alert.json'
    if alert_path.is_file():
        raise RuntimeError(f'formal watchdog recorded an alert: {load_json(alert_path)}')
    completion_path = watchdog_root / 'completion_health.json'
    if not completion_path.is_file():
        latest_path = watchdog_root / 'latest_health.json'
        state = load_json(latest_path).get('state') if latest_path.is_file() else 'unavailable'
        return {
            'ready': False,
            'completion_health_present': False,
            'latest_state': state,
            'expected_runs': counts['formal_runs'],
        }
    health = load_json(completion_path)
    expected = {
        'state': 'complete',
        'alerts': [],
        'formal_matrix_complete': True,
        'completed_runs': counts['formal_runs'],
        'failed_runs': 0,
        'stale_runs': 0,
        'total_runs': counts['formal_runs'],
        'active_workers': 0,
    }
    for field, value in expected.items():
        if health.get(field) != value:
            raise ValueError(f'watchdog completion field {field!r} changed')
    if int(health['completed_environment_steps']) != int(health['planned_environment_steps']):
        raise ValueError('watchdog completion does not cover the full training budget')
    return {
        'ready': True,
        'completion_health_present': True,
        'latest_state': health['state'],
        'expected_runs': counts['formal_runs'],
        'filesystem_free_bytes': int(health['filesystem_free_bytes']),
    }


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    """Parse one portable two-space SHA-256 manifest."""
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            digest, relative = line.split('  ', 1)
        except ValueError as error:
            raise ValueError(f'invalid checksum line {line_number}: {line!r}') from error
        candidate = Path(relative)
        if (
            len(digest) != 64
            or any(character not in '0123456789abcdef' for character in digest)
            or candidate.is_absolute()
            or '..' in candidate.parts
            or relative in entries
        ):
            raise ValueError(f'unsafe checksum line {line_number}: {line!r}')
        entries[relative] = digest
    return entries


def validate_final_output(
    output_root: Path,
    matrix: dict[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    """Revalidate the complete portable final bundle and its statistical identity."""
    if not output_root.is_dir() or not (output_root / 'finalization_complete').is_file():
        raise ValueError(f'incomplete final output: {output_root}')
    if any(path.is_symlink() for path in output_root.rglob('*')):
        raise ValueError(f'final output contains a symbolic link: {output_root}')
    checksum_path = output_root / 'checksums.sha256'
    entries = parse_checksum_manifest(checksum_path)
    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob('*')
        if path.is_file() and path.name not in CHECKSUM_EXCLUSIONS
    }
    if set(entries) != actual:
        raise ValueError('final output checksum coverage changed')
    for relative, expected in entries.items():
        if sha256_file(output_root / relative) != expected:
            raise ValueError(f'final output checksum mismatch: {relative}')

    counts = expected_counts(matrix)
    summary = load_json(output_root / 'shapo_p47_final_summary.json')
    expected_summary = {
        'source_commit': source_commit,
        'statistical_unit': 'training_seed',
        'n_training_seeds_per_environment': len(matrix['training_seeds']),
        **{
            field: counts[field]
            for field in (
                'evaluation_artifact_count',
                'provenance_artifact_count',
                'trajectory_artifact_count',
                'checkpoint_artifact_count',
            )
        },
    }
    for field, value in expected_summary.items():
        if summary.get(field) != value:
            raise ValueError(f'final summary field {field!r} changed')
    validations = list((output_root / 'strict_validations').glob('*.json'))
    if len(validations) != counts['formal_runs']:
        raise ValueError('final output has incomplete strict run validations')
    if list(output_root.rglob('*.pt')):
        raise ValueError('portable final output copied server-only checkpoints')
    return {
        'status': 'passed',
        'output_root': str(output_root),
        'bytes': sum(path.stat().st_size for path in output_root.rglob('*') if path.is_file()),
        'file_count': sum(path.is_file() for path in output_root.rglob('*')),
        'checksum_manifest_sha256': sha256_file(checksum_path),
        'strict_validation_count': len(validations),
        **expected_summary,
    }


def completion_gates(  # pylint: disable=too-many-arguments
    *,
    run_root: Path,
    milestone_root: Path,
    watchdog_root: Path,
    matrix_path: Path,
    matrix: dict[str, Any],
    source_commit: str,
    audit_script: Path,
) -> dict[str, Any]:
    """Evaluate all three independent pre-finalization gates."""
    training = validate_training_completion(run_root, matrix)
    milestones = validate_milestone_completion(
        run_root,
        milestone_root,
        matrix_path,
        matrix,
        source_commit,
        audit_script,
    )
    watchdog = validate_watchdog_completion(watchdog_root, matrix)
    ready = bool(training['ready'] and milestones['ready'] and watchdog['ready'])
    return {
        'ready': ready,
        'training': training,
        'milestones': milestones,
        'watchdog': watchdog,
        'scientific_protocol_changed': False,
        'training_state_changed': False,
        'used_for_checkpoint_or_configuration_selection': False,
    }


def invoke_finalizer(
    validation_source: Path,
    run_root: Path,
    python_bin: Path,
    output_root: Path,
    finalizer_script: Path,
) -> dict[str, Any]:
    """Invoke the existing fail-closed finalization wrapper once."""
    command = [
        'bash',
        str(finalizer_script),
        str(validation_source),
        str(run_root),
        str(python_bin),
        str(output_root),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f'formal finalizer failed with exit {completed.returncode}; '
            f'stdout={completed.stdout.strip()!r}; stderr={completed.stderr.strip()!r}',
        )
    return {
        'command': command,
        'stdout': completed.stdout.strip(),
        'stderr': completed.stderr.strip(),
    }


def complete_once(  # pylint: disable=too-many-arguments
    *,
    run_root: Path,
    milestone_root: Path,
    watchdog_root: Path,
    matrix_path: Path,
    matrix: dict[str, Any],
    source_commit: str,
    audit_script: Path,
    validation_source: Path,
    python_bin: Path,
    output_root: Path,
    finalizer_script: Path,
) -> dict[str, Any]:
    """Wait without mutation, or finalize and verify exactly once when ready."""
    if output_root.exists():
        final = validate_final_output(output_root, matrix, source_commit)
        return {'state': 'complete', 'finalization': 'already_present', 'final': final}
    gates = completion_gates(
        run_root=run_root,
        milestone_root=milestone_root,
        watchdog_root=watchdog_root,
        matrix_path=matrix_path,
        matrix=matrix,
        source_commit=source_commit,
        audit_script=audit_script,
    )
    if not gates['ready']:
        return {'state': 'waiting', 'gates': gates}
    invocation = invoke_finalizer(
        validation_source,
        run_root,
        python_bin,
        output_root,
        finalizer_script,
    )
    final = validate_final_output(output_root, matrix, source_commit)
    return {
        'state': 'complete',
        'finalization': 'invoked',
        'gates': gates,
        'invocation': invocation,
        'final': final,
    }


def git_identity(validation_source: Path, source_commit: str) -> dict[str, str]:
    """Require one clean validation commit descended from frozen training source."""
    status = subprocess.run(
        ['git', '-C', str(validation_source), 'status', '--porcelain'],
        capture_output=True,
        text=True,
        check=True,
    )
    if status.stdout.strip():
        raise ValueError(f'completion validation source is dirty: {validation_source}')
    commit = subprocess.check_output(
        ['git', '-C', str(validation_source), 'rev-parse', 'HEAD'],
        text=True,
    ).strip()
    ancestor = subprocess.run(
        ['git', '-C', str(validation_source), 'merge-base', '--is-ancestor', source_commit, commit],
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError('completion validation source is not descended from formal source')
    return {'validation_commit': commit, 'formal_source_commit': source_commit}


def main() -> None:  # pylint: disable=too-many-locals
    """Poll immutable completion evidence and finalize without touching training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--milestone-root', type=Path, required=True)
    parser.add_argument('--watchdog-root', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--validation-source', type=Path, required=True)
    parser.add_argument('--python-bin', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument(
        '--audit-script',
        type=Path,
        default=Path(__file__).resolve().with_name('audit_shapo_p47_milestone.py'),
    )
    parser.add_argument(
        '--finalizer-script',
        type=Path,
        default=Path(__file__).resolve().with_name('finalize_shapo_p47_formal.sh'),
    )
    parser.add_argument('--poll-seconds', type=float, default=600.0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if len(args.source_commit) != 40:
        parser.error('--source-commit must be a full commit identity')
    if args.poll_seconds <= 0.0:
        parser.error('--poll-seconds must be positive')
    if not args.python_bin.is_file() or not os.access(args.python_bin, os.X_OK):
        parser.error('--python-bin must be executable')

    run_root = args.run_root.resolve()
    milestone_root = args.milestone_root.resolve()
    watchdog_root = args.watchdog_root.resolve()
    matrix_path = args.matrix.resolve()
    validation_source = args.validation_source.resolve()
    output_root = args.output_root.resolve()
    state_dir = args.state_dir.resolve()
    audit_script = args.audit_script.resolve()
    finalizer_script = args.finalizer_script.resolve()
    if output_root != run_root / 'public_artifacts':
        parser.error('--output-root must be the formal run public_artifacts directory')
    if state_dir in (run_root, output_root) or output_root in state_dir.parents:
        parser.error('--state-dir must not overlap formal output')
    state_dir.mkdir(parents=True, exist_ok=True)

    matrix = yaml.safe_load(matrix_path.read_text(encoding='utf-8'))
    source_identity = git_identity(validation_source, args.source_commit)
    immutable_paths = {
        'completion_watcher': Path(__file__).resolve(),
        'matrix': matrix_path,
        'audit_script': audit_script,
        'strict_validator': audit_script.with_name('validate_shapo_p47_run.py'),
        'finalizer_script': finalizer_script,
        'finalizer_module': finalizer_script.with_name('finalize_shapo_p47_formal.py'),
    }
    immutable_hashes = {name: sha256_file(path) for name, path in immutable_paths.items()}
    identities = {
        **source_identity,
        **{f'{name}_sha256': digest for name, digest in immutable_hashes.items()},
        'scientific_protocol_changed': False,
        'training_state_changed': False,
        'used_for_checkpoint_or_configuration_selection': False,
    }

    with (state_dir / '.completion-watch.lock').open('w', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit('another completion watcher holds the state lock') from error
        while True:
            observed_at = datetime.now(timezone.utc).isoformat()
            try:
                for name, path in immutable_paths.items():
                    if sha256_file(path) != immutable_hashes[name]:
                        raise ValueError(f'completion dependency changed while running: {name}')
                if git_identity(validation_source, args.source_commit) != source_identity:
                    raise ValueError('completion validation source identity changed while running')
                snapshot = complete_once(
                    run_root=run_root,
                    milestone_root=milestone_root,
                    watchdog_root=watchdog_root,
                    matrix_path=matrix_path,
                    matrix=matrix,
                    source_commit=args.source_commit,
                    audit_script=audit_script,
                    validation_source=validation_source,
                    python_bin=args.python_bin.resolve(),
                    output_root=output_root,
                    finalizer_script=finalizer_script,
                )
                snapshot.update({'observed_at_utc': observed_at, **identities})
            except Exception as error:
                alert = {
                    'observed_at_utc': observed_at,
                    'state': 'alert',
                    'error_type': type(error).__name__,
                    'error': str(error),
                    **identities,
                }
                atomic_json(state_dir / 'completion_watch_alert.json', alert)
                raise
            atomic_json(state_dir / 'completion_watch_latest.json', snapshot)
            append_jsonl(
                state_dir / 'completion_watch_journal.jsonl',
                {
                    'observed_at_utc': observed_at,
                    'state': snapshot['state'],
                    'scientific_protocol_changed': False,
                    'training_state_changed': False,
                    'used_for_checkpoint_or_configuration_selection': False,
                },
            )
            print(
                json.dumps(
                    {
                        'observed_at_utc': observed_at,
                        'state': snapshot['state'],
                        'ready': snapshot.get('gates', {}).get('ready', True),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if snapshot['state'] == 'complete':
                atomic_json(state_dir / 'completion_watch_complete.json', snapshot)
                return
            if args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
