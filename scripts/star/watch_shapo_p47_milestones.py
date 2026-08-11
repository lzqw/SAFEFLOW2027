#!/usr/bin/env python3
"""Audit every locked SHAPO P4.7 milestone after its artifacts become stable."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.star.status_shapo_p47_formal import SLUGS


CHECKSUM_EXCLUSIONS = {'checksums.sha256', 'milestone_audit_complete'}
GLOB_SOURCES = {
    'evaluation': 'training/**/artifacts/evaluation_step_{milestone}.json',
    'trajectory': 'training/**/artifacts/trajectory_step_{milestone}.csv',
    'counterfactual': 'training/**/artifacts/counterfactual_step_{milestone}.json',
    'milestone_checkpoint': 'training/**/checkpoints/milestone_{milestone}.pt',
    'run_manifest': 'training/**/artifacts/run_manifest.json',
    'best_safe_return_selection': 'training/**/artifacts/best_safe_return_selection.json',
    'warmup_status': 'training/**/artifacts/warmup_status.json',
    'evaluation_curve': 'training/**/artifacts/evaluation.csv',
}
DIRECT_SOURCES = {
    'resolved_config': 'resolved_formal_config.yaml',
    'materialization_receipt': 'materialize.json',
    'source_commit': 'source_commit.txt',
}


@dataclass(frozen=True)
class MilestoneJob:
    """One immutable environment, training-seed, and milestone audit target."""

    environment: str
    seed: int
    milestone: int
    container: Path
    output: Path

    @property
    def key(self) -> str:
        """Return a stable human-readable job identity."""
        return f'{self.environment}/seed-{self.seed}/step-{self.milestone}'


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    """Atomically replace a strict JSON receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    """Append and fsync a compact watcher journal entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def compact_status(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Reduce one detailed scan to a bounded append-only monitoring record."""
    state_counts: dict[str, int] = {}
    newly_audited = []
    for row in snapshot['jobs']:
        state = str(row['state'])
        state_counts[state] = state_counts.get(state, 0) + 1
        if state == 'audited':
            newly_audited.append({'job': row['job'], 'output': row['output']})
    return {
        key: snapshot[key]
        for key in (
            'observed_at_utc',
            'state',
            'protocol_version',
            'protocol_sha256',
            'matrix_sha256',
            'source_commit',
            'watcher_sha256',
            'audit_script_sha256',
            'strict_evaluation_validator_sha256',
            'scientific_protocol_changed',
            'training_state_changed',
            'used_for_checkpoint_or_configuration_selection',
            'total_jobs',
            'sealed_jobs',
            'newly_audited_jobs',
            'waiting_jobs',
        )
    } | {'job_state_counts': state_counts, 'newly_audited': newly_audited}


def locked_jobs(
    run_root: Path,
    output_root: Path,
    matrix: dict[str, Any],
) -> list[MilestoneJob]:
    """Expand the sealed matrix into deterministic milestone jobs."""
    jobs = []
    for seed_value in matrix['training_seeds']:
        seed = int(seed_value)
        for environment, config in matrix['environments'].items():
            if environment not in SLUGS:
                raise ValueError(f'unsupported locked environment: {environment}')
            slug = SLUGS[environment]
            container = run_root / 'runs' / f'{slug}-{seed}'
            for milestone_value in config['milestones']:
                milestone = int(milestone_value)
                jobs.append(
                    MilestoneJob(
                        environment=environment,
                        seed=seed,
                        milestone=milestone,
                        container=container,
                        output=output_root / f'step_{milestone}' / f'{slug}-{seed}',
                    ),
                )
    return jobs


def newest_optional(container: Path, pattern: str) -> Path | None:
    """Return the newest matching path, or ``None`` while it is unavailable."""
    matches = list(container.glob(pattern))
    return max(matches, key=lambda path: path.stat().st_mtime_ns) if matches else None


def source_artifacts(job: MilestoneJob) -> tuple[dict[str, Path], list[str]]:
    """Resolve every artifact required by the strict milestone auditor."""
    resolved: dict[str, Path] = {}
    missing = []
    for label, template in GLOB_SOURCES.items():
        path = newest_optional(job.container, template.format(milestone=job.milestone))
        if path is None:
            missing.append(label)
        else:
            resolved[label] = path
    for label, relative in DIRECT_SOURCES.items():
        path = job.container / relative
        if not path.is_file():
            missing.append(label)
        else:
            resolved[label] = path
    return resolved, sorted(missing)


def stable_sources(
    job: MilestoneJob,
    *,
    now: float,
    stable_seconds: float,
) -> tuple[bool, list[str], float | None]:
    """Require all sources to exist, be nonempty, and age past a stability window."""
    artifacts, missing = source_artifacts(job)
    if missing:
        return False, missing, None
    empty = sorted(label for label, path in artifacts.items() if path.stat().st_size <= 0)
    if empty:
        return False, [f'empty:{label}' for label in empty], None
    newest_mtime = max(path.stat().st_mtime for path in artifacts.values())
    stable_age = max(0.0, now - newest_mtime)
    return stable_age >= stable_seconds, [], stable_age


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    """Parse the auditor's portable two-space SHA-256 manifest."""
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            digest, relative = line.split('  ', 1)
        except ValueError as error:
            raise ValueError(f'invalid checksum line {line_number}: {line!r}') from error
        candidate = Path(relative)
        if len(digest) != 64 or any(character not in '0123456789abcdef' for character in digest):
            raise ValueError(f'invalid checksum digest on line {line_number}')
        if candidate.is_absolute() or '..' in candidate.parts or relative in entries:
            raise ValueError(f'unsafe or duplicate checksum path on line {line_number}')
        entries[relative] = digest
    return entries


def milestone_audit_identity(
    matrix_path: Path,
    source_commit: str,
    audit_script: Path,
) -> dict[str, Any]:
    """Return the complete immutable identity expected in every sealed audit."""
    matrix = yaml.safe_load(matrix_path.read_text(encoding='utf-8'))
    validator = audit_script.with_name('validate_shapo_p47_run.py')
    return {
        'audit_version': 'shapo-p47-milestone-v1',
        'audit_script_sha256': sha256_file(audit_script),
        'strict_evaluation_validator_sha256': sha256_file(validator),
        'protocol_version': matrix['protocol_version'],
        'protocol_sha256': matrix['protocol_sha256'],
        'matrix_sha256': sha256_file(matrix_path),
        'frozen_method_canonical_sha256': matrix['frozen_method_canonical_sha256'],
        'source_commit': source_commit,
    }


def validate_sealed_output(
    job: MilestoneJob,
    *,
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Revalidate a compact milestone seal before treating it as complete."""
    root = job.output
    if not root.is_dir() or not (root / 'milestone_audit_complete').is_file():
        raise ValueError(f'incomplete milestone audit output: {root}')
    checksum_path = root / 'checksums.sha256'
    if not checksum_path.is_file():
        raise ValueError(f'milestone checksum manifest is missing: {root}')
    entries = parse_checksum_manifest(checksum_path)
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob('*')
        if path.is_file() and path.name not in CHECKSUM_EXCLUSIONS
    }
    if set(entries) != actual_files:
        raise ValueError(f'milestone checksum coverage mismatch: {root}')
    for relative, expected in entries.items():
        path = root / relative
        if path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f'milestone artifact identity mismatch: {path}')
    audit = json.loads((root / 'milestone_audit.json').read_text(encoding='utf-8'))
    expected_fields = {
        'status': 'passed',
        'environment': job.environment,
        'training_seed': job.seed,
        'milestone': job.milestone,
        'training_seed_is_statistical_unit': True,
        'diagnostic_only': True,
        'used_for_checkpoint_or_configuration_selection': False,
    }
    for field, expected in expected_fields.items():
        if audit.get(field) != expected:
            raise ValueError(f'milestone audit field {field!r} changed: {root}')
    for field, expected in (expected_identity or {}).items():
        if audit.get(field) != expected:
            raise ValueError(f'milestone audit identity {field!r} changed: {root}')
    checkpoint = audit.get('server_only_checkpoint', {})
    if checkpoint.get('server_only') is not True:
        raise ValueError(f'milestone checkpoint is not marked server-only: {root}')
    copied_checkpoints = list(root.rglob('*.pt'))
    if copied_checkpoints:
        raise ValueError(
            f'milestone compact output copied checkpoint payloads: {copied_checkpoints}',
        )
    return audit


def invoke_auditor(
    job: MilestoneJob,
    *,
    matrix_path: Path,
    source_commit: str,
    audit_script: Path,
    python_bin: Path,
) -> dict[str, Any]:
    """Run the existing strict auditor for one stable locked milestone."""
    command = [
        str(python_bin),
        str(audit_script),
        '--container',
        str(job.container),
        '--matrix',
        str(matrix_path),
        '--env-id',
        job.environment,
        '--seed',
        str(job.seed),
        '--milestone',
        str(job.milestone),
        '--source-commit',
        source_commit,
        '--output-dir',
        str(job.output),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f'milestone auditor failed for {job.key} with exit {completed.returncode}; '
            f'stdout={completed.stdout.strip()!r}; stderr={completed.stderr.strip()!r}',
        )
    return {
        'command': command,
        'stdout': completed.stdout.strip(),
        'stderr': completed.stderr.strip(),
    }


def scan_once(  # pylint: disable=too-many-arguments
    jobs: list[MilestoneJob],
    *,
    matrix_path: Path,
    source_commit: str,
    audit_script: Path,
    python_bin: Path,
    stable_seconds: float,
    now: float | None = None,
) -> dict[str, Any]:
    """Validate sealed jobs and audit newly stable jobs exactly once."""
    observed_at = time.time() if now is None else now
    expected_identity = milestone_audit_identity(matrix_path, source_commit, audit_script)
    rows = []
    for job in jobs:
        if job.output.exists():
            validate_sealed_output(job, expected_identity=expected_identity)
            rows.append({'job': job.key, 'state': 'sealed', 'output': str(job.output)})
            continue
        ready, missing, stable_age = stable_sources(
            job,
            now=observed_at,
            stable_seconds=stable_seconds,
        )
        if not ready:
            state = 'waiting_for_sources' if missing else 'waiting_for_stability'
            rows.append(
                {
                    'job': job.key,
                    'state': state,
                    'missing': missing,
                    'stable_age_seconds': stable_age,
                },
            )
            continue
        invocation = invoke_auditor(
            job,
            matrix_path=matrix_path,
            source_commit=source_commit,
            audit_script=audit_script,
            python_bin=python_bin,
        )
        validate_sealed_output(job, expected_identity=expected_identity)
        rows.append(
            {
                'job': job.key,
                'state': 'audited',
                'output': str(job.output),
                'stable_age_seconds': stable_age,
                'auditor_stdout': invocation['stdout'],
            },
        )
    sealed = sum(row['state'] in {'sealed', 'audited'} for row in rows)
    return {
        'observed_at_utc': datetime.fromtimestamp(observed_at, timezone.utc).isoformat(),
        'state': 'complete' if sealed == len(jobs) else 'watching',
        'scientific_protocol_changed': False,
        'training_state_changed': False,
        'used_for_checkpoint_or_configuration_selection': False,
        'total_jobs': len(jobs),
        'sealed_jobs': sealed,
        'newly_audited_jobs': sum(row['state'] == 'audited' for row in rows),
        'waiting_jobs': len(jobs) - sealed,
        'jobs': rows,
    }


def main() -> None:  # pylint: disable=too-many-locals
    """Poll until all locked milestone audits are independently sealed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument(
        '--audit-script',
        type=Path,
        default=Path(__file__).resolve().with_name('audit_shapo_p47_milestone.py'),
    )
    parser.add_argument('--python-bin', type=Path, default=Path(sys.executable))
    parser.add_argument('--poll-seconds', type=float, default=600.0)
    parser.add_argument('--stable-seconds', type=float, default=120.0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if len(args.source_commit) != 40:
        parser.error('--source-commit must be a full commit identity')
    if args.poll_seconds <= 0.0:
        parser.error('--poll-seconds must be positive')
    if args.stable_seconds < 0.0:
        parser.error('--stable-seconds must be non-negative')
    if not args.audit_script.is_file() or not os.access(args.python_bin, os.X_OK):
        parser.error('auditor or Python interpreter is unavailable')

    run_root = args.run_root.resolve()
    output_root = args.output_root.resolve()
    runs_root = run_root / 'runs'
    if output_root in (run_root, runs_root) or runs_root in output_root.parents:
        parser.error('--output-root must not overlap formal training containers')
    matrix_path = args.matrix.resolve()
    matrix = yaml.safe_load(matrix_path.read_text(encoding='utf-8'))
    watcher_path = Path(__file__).resolve()
    watcher_sha256 = sha256_file(watcher_path)
    audit_script = args.audit_script.resolve()
    audit_identity = milestone_audit_identity(matrix_path, args.source_commit, audit_script)
    matrix_sha256 = audit_identity['matrix_sha256']
    validator_path = audit_script.with_name('validate_shapo_p47_run.py')
    validator_sha256 = audit_identity['strict_evaluation_validator_sha256']
    identities = {
        'protocol_version': audit_identity['protocol_version'],
        'protocol_sha256': audit_identity['protocol_sha256'],
        'matrix_sha256': matrix_sha256,
        'source_commit': args.source_commit,
        'watcher_sha256': watcher_sha256,
        'audit_script_sha256': audit_identity['audit_script_sha256'],
        'strict_evaluation_validator_sha256': validator_sha256,
    }
    jobs = locked_jobs(run_root, output_root, matrix)
    output_root.mkdir(parents=True, exist_ok=True)

    with (output_root / '.milestone-watch.lock').open('w', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit('another milestone watcher holds the output lock') from error
        while True:
            try:
                if sha256_file(matrix_path) != matrix_sha256:
                    raise ValueError('locked milestone matrix changed while watcher was running')
                if sha256_file(watcher_path) != watcher_sha256:
                    raise ValueError('milestone watcher source changed while it was running')
                if sha256_file(audit_script) != audit_identity['audit_script_sha256']:
                    raise ValueError('strict milestone auditor changed while watcher was running')
                if sha256_file(validator_path) != validator_sha256:
                    raise ValueError(
                        'strict evaluation validator changed while watcher was running',
                    )
                snapshot = scan_once(
                    jobs,
                    matrix_path=matrix_path,
                    source_commit=args.source_commit,
                    audit_script=audit_script,
                    python_bin=args.python_bin.resolve(),
                    stable_seconds=args.stable_seconds,
                )
                snapshot.update(identities)
            except Exception as error:  # preserve a durable engineering alert before exiting
                alert = {
                    'observed_at_utc': datetime.now(timezone.utc).isoformat(),
                    'state': 'alert',
                    'error_type': type(error).__name__,
                    'error': str(error),
                    'scientific_protocol_changed': False,
                    'training_state_changed': False,
                }
                atomic_json(output_root / 'milestone_watch_alert.json', alert)
                raise
            atomic_json(output_root / 'milestone_watch_latest.json', snapshot)
            journal_entry = compact_status(snapshot)
            append_jsonl(output_root / 'milestone_watch_journal.jsonl', journal_entry)
            print(json.dumps(journal_entry, sort_keys=True))
            if snapshot['state'] == 'complete':
                (output_root / 'milestone_watch_complete').touch()
                return
            if args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
