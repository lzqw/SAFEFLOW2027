#!/usr/bin/env python3
"""Publish verified SHAPO host evidence to the existing GitHub branch exactly once."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.star.watch_shapo_p47_final_sync import validate_local_output  # noqa: E402
from scripts.star.watch_shapo_p47_milestones import (  # noqa: E402
    locked_jobs,
    milestone_audit_identity,
    validate_sealed_output,
)


FULL_COMMIT = re.compile(r'^[0-9a-f]{40}$')
SAFE_REMOTE = re.compile(r'^[A-Za-z0-9_.-]+$')
SAFE_BRANCH = re.compile(r'^(?!/)(?!.*\.\.)(?!.*//)[A-Za-z0-9._/-]+(?<!/)$')
SAFE_GITHUB_REMOTE = re.compile(
    r'^(?:https://github\.com/|git@github\.com:)[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$',
)
ALLOWED_UNTRACKED = ('METHOD/', 'PAPER/', 'REF-PAPER/')
PUBLISH_PATHS = (
    'reports/star/shapo_p47_four_env_artifacts/milestones',
    'reports/star/shapo_p47_four_env_artifacts/final',
)
MAX_GITHUB_FILE_BYTES = 95 * 1024 * 1024
MAX_ALERT_BYTES = 1024 * 1024
GIT_REMOTE_READ_TIMEOUT_SECONDS = 120
GIT_PUSH_TIMEOUT_SECONDS = 3600
GIT_LOCAL_TIMEOUT_SECONDS = 600
GIT_KILL_GRACE_SECONDS = 30


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
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
    """Append and fsync one compact publication journal row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def archive_resolved_alert(state_dir: Path, resolved_at: str) -> dict[str, str] | None:
    """Atomically archive one prior active alert after a successful poll."""
    active = state_dir / 'github_publish_watch_alert.json'
    if not os.path.lexists(active):
        return None
    if active.is_symlink() or not active.is_file():
        raise ValueError('active GitHub publication alert is not a regular file')
    raw = active.read_bytes()
    if not raw or len(raw) > MAX_ALERT_BYTES:
        raise ValueError('active GitHub publication alert has an invalid size')
    try:
        alert = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('active GitHub publication alert is not strict UTF-8 JSON') from error
    if not isinstance(alert, dict) or alert.get('state') != 'alert':
        raise ValueError('active GitHub publication alert has an invalid state')
    observed_at = alert.get('observed_at_utc')
    if not isinstance(observed_at, str):
        raise ValueError('active GitHub publication alert lacks its observation time')
    try:
        observed = datetime.fromisoformat(observed_at.replace('Z', '+00:00'))
        resolved = datetime.fromisoformat(resolved_at.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('GitHub publication alert timestamps are invalid') from error
    if observed.tzinfo is None or resolved.tzinfo is None:
        raise ValueError('GitHub publication alert timestamps must be timezone-aware')
    if resolved < observed:
        raise ValueError('GitHub publication alert cannot resolve before it was observed')

    digest = hashlib.sha256(raw).hexdigest()
    resolved_dir = state_dir / 'resolved-alerts'
    if os.path.lexists(resolved_dir):
        if resolved_dir.is_symlink() or not resolved_dir.is_dir():
            raise ValueError('resolved GitHub publication alert path is not a directory')
    else:
        resolved_dir.mkdir(mode=0o700)
    os.chmod(resolved_dir, 0o700)
    stamp = observed.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = resolved_dir / f'github_publish_watch_alert-{stamp}-{digest[:12]}.json'
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_file():
            raise ValueError('resolved GitHub publication alert is not a regular file')
        if destination.read_bytes() != raw:
            raise ValueError('resolved GitHub publication alert identity collision')
        os.chmod(destination, 0o600)
        active.unlink()
    else:
        os.chmod(active, 0o600)
        os.replace(active, destination)
    return {
        'alert_observed_at_utc': observed_at,
        'resolved_at_utc': resolved_at,
        'archive_path': str(destination.relative_to(state_dir)),
        'archive_sha256': digest,
    }


def git(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run Git without invoking a shell or an interactive credential prompt."""
    environment = {**os.environ, 'GIT_TERMINAL_PROMPT': '0'}
    command = ['git', '-C', str(repo), *arguments]
    operation = arguments[0] if arguments else ''
    timeout = {
        'ls-remote': GIT_REMOTE_READ_TIMEOUT_SECONDS,
        'push': GIT_PUSH_TIMEOUT_SECONDS,
    }.get(operation, GIT_LOCAL_TIMEOUT_SECONDS)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        try:
            process.communicate(timeout=GIT_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired as reap_error:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            try:
                process.kill()
                process.wait(timeout=GIT_KILL_GRACE_SECONDS)
            except (ProcessLookupError, subprocess.TimeoutExpired) as kill_error:
                raise RuntimeError(
                    f'git {operation or "command"} timed out and could not be reaped',
                ) from kill_error
            raise RuntimeError(
                f'git {operation or "command"} timed out with inherited pipes still open',
            ) from reap_error
        raise RuntimeError(f'git {operation or "command"} exceeded {timeout} seconds') from error
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f'git {arguments[0]} failed with exit {completed.returncode}: '
            f'{completed.stderr.strip()}',
        )
    return completed


def status_entries(repo: Path) -> list[tuple[str, str]]:
    """Parse strict porcelain-v1 status records without losing special characters."""
    output = git(repo, 'status', '--porcelain=v1', '-z', '--untracked-files=all').stdout
    records = output.split('\0')
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2] != ' ':
            raise ValueError(f'invalid porcelain record: {record!r}')
        status, path = record[:2], record[3:]
        if 'R' in status or 'C' in status:
            if index >= len(records) or not records[index]:
                raise ValueError('incomplete rename/copy porcelain record')
            path = f'{path} -> {records[index]}'
            index += 1
        entries.append((status, path))
    return entries


def allowed_untracked(path: str) -> bool:
    """Return whether an untracked user path is explicitly outside publication scope."""
    return any(path.startswith(prefix) for prefix in ALLOWED_UNTRACKED)


def validate_prepublication_tree(
    repo: Path,
    *,
    allow_staged_publish_paths: bool = False,
) -> dict[str, Any]:
    """Require only verified SHAPO artifacts plus known user-owned untracked files."""
    entries = status_entries(repo)
    unexpected = [
        (status, path)
        for status, path in entries
        if not (
            (status == '??' and allowed_untracked(path))
            or (
                any(path.startswith(f'{root}/') for root in PUBLISH_PATHS)
                and (
                    status == '??'
                    or (allow_staged_publish_paths and status[0] != ' ' and status[1] == ' ')
                )
            )
        )
    ]
    if unexpected:
        raise ValueError(f'unexpected worktree changes before publication: {unexpected[:10]}')
    publish_entries = [
        (status, path)
        for status, path in entries
        if any(path.startswith(f'{root}/') for root in PUBLISH_PATHS)
    ]
    return {
        'status_entries': len(entries),
        'publish_entries': len(publish_entries),
        'preserved_user_entries': len(entries) - len(publish_entries),
    }


def validate_repository_ancestry(
    repo: Path,
    remote: str,
    branch: str,
    expected_head: str,
    commit_message: str = '',
    expected_remote_url: str = '',
) -> dict[str, str]:
    """Require the locked HEAD or one exact locally committed publication transaction."""
    if expected_remote_url:
        if not SAFE_GITHUB_REMOTE.fullmatch(expected_remote_url):
            raise ValueError('expected GitHub remote URL is unsafe or contains credentials')
        push_url = git(repo, 'remote', 'get-url', '--push', remote).stdout.strip()
        if push_url != expected_remote_url:
            raise ValueError(f'Git push URL changed for remote {remote!r}')
    if local_branch(repo) != branch:
        raise ValueError(f'checked-out branch changed from {branch!r}')
    head = git(repo, 'rev-parse', 'HEAD').stdout.strip()
    remote_head = remote_branch_oid(repo, remote, branch)
    if head == expected_head:
        require_ancestor(repo, remote_head, head)
        return {'local_head': head, 'remote_head': remote_head, 'state': 'precommit'}
    if not commit_message:
        raise ValueError('local HEAD changed after publication watcher launch')
    parents = git(repo, 'show', '-s', '--format=%P', head).stdout.strip().split()
    subject = git(repo, 'show', '-s', '--format=%s', head).stdout.rstrip('\n')
    changed = [
        path
        for path in git(
            repo,
            'diff-tree',
            '--no-commit-id',
            '--name-only',
            '-r',
            '-z',
            head,
        ).stdout.split('\0')
        if path
    ]
    if (
        parents != [expected_head]
        or subject != commit_message
        or not changed
        or any(
            not any(path == root or path.startswith(f'{root}/') for root in PUBLISH_PATHS)
            for path in changed
        )
    ):
        raise ValueError('local HEAD is not the exact recoverable publication transaction')
    if remote_head not in {expected_head, head}:
        raise ValueError('remote branch changed outside the recoverable publication transaction')
    return {'local_head': head, 'remote_head': remote_head, 'state': 'committed'}


def validate_host_evidence(  # pylint: disable=too-many-arguments
    *,
    repo: Path,
    matrix_path: Path,
    source_commit: str,
    audit_script: Path,
    milestone_root: Path,
    final_root: Path,
    remote_host: str,
    remote_run_root: str,
) -> dict[str, Any]:
    """Revalidate every host milestone seal and the complete final transfer."""
    import yaml  # pylint: disable=import-outside-toplevel

    matrix = yaml.safe_load(matrix_path.read_text(encoding='utf-8'))
    identity = milestone_audit_identity(matrix_path, source_commit, audit_script)
    jobs = locked_jobs(Path('/server-only/formal'), milestone_root, matrix)
    if len(jobs) != 60:
        raise ValueError('frozen matrix does not expand to 60 milestone jobs')
    for job in jobs:
        validate_sealed_output(job, expected_identity=identity)
    expected_milestone_files = {
        path.resolve() for job in jobs for path in job.output.rglob('*') if path.is_file()
    }
    actual_milestone_files = {
        path.resolve() for path in milestone_root.rglob('*') if path.is_file()
    }
    if actual_milestone_files != expected_milestone_files:
        raise ValueError('milestone publication root contains files outside the 60 sealed jobs')
    final = validate_local_output(
        final_root,
        remote_host=remote_host,
        remote_run_root=remote_run_root,
        source_commit=source_commit,
    )
    if repo not in milestone_root.parents or repo not in final_root.parents:
        raise ValueError('host evidence is outside the publication repository')
    publish_files = [
        path for root in (milestone_root, final_root) for path in root.rglob('*') if path.is_file()
    ]
    oversized = [
        (str(path.relative_to(repo)), path.stat().st_size)
        for path in publish_files
        if path.stat().st_size > MAX_GITHUB_FILE_BYTES
    ]
    if oversized:
        raise ValueError(f'host evidence exceeds the GitHub file-size safety limit: {oversized}')
    return {
        'milestone_jobs': len(jobs),
        'final': final,
        'publish_file_count': len(publish_files),
        'largest_publish_file_bytes': max(path.stat().st_size for path in publish_files),
        'maximum_github_file_bytes': MAX_GITHUB_FILE_BYTES,
    }


def host_evidence_readiness(
    matrix_path: Path,
    milestone_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    """Return bounded readiness without treating independent sync timing as failure."""
    import yaml  # pylint: disable=import-outside-toplevel

    matrix = yaml.safe_load(matrix_path.read_text(encoding='utf-8'))
    jobs = locked_jobs(Path('/server-only/formal'), milestone_root, matrix)
    if len(jobs) != 60:
        raise ValueError('frozen matrix does not expand to 60 milestone jobs')
    sealed = sum((job.output / 'milestone_audit_complete').is_file() for job in jobs)
    final_complete = (final_root / 'transfer_complete').is_file()
    return {
        'ready': bool(final_complete and sealed == len(jobs)),
        'final_host_transfer_complete': final_complete,
        'host_milestone_seals': sealed,
        'expected_host_milestone_seals': len(jobs),
    }


def local_branch(repo: Path) -> str:
    """Return the checked-out symbolic branch, rejecting a detached HEAD."""
    completed = git(repo, 'symbolic-ref', '--quiet', '--short', 'HEAD', check=False)
    branch = completed.stdout.strip()
    if completed.returncode != 0 or not branch:
        raise ValueError('GitHub publication requires a checked-out branch')
    return branch


def remote_branch_oid(repo: Path, remote: str, branch: str) -> str:
    """Read exactly one remote branch identity without changing local refs."""
    completed = git(repo, 'ls-remote', '--heads', remote, branch)
    rows = [line.split() for line in completed.stdout.splitlines() if line.strip()]
    expected_ref = f'refs/heads/{branch}'
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != expected_ref:
        raise ValueError(f'remote branch {remote}/{branch} is unavailable or ambiguous')
    oid = rows[0][0]
    if not FULL_COMMIT.fullmatch(oid):
        raise ValueError('remote branch returned an invalid commit identity')
    return oid


def require_ancestor(repo: Path, ancestor: str, descendant: str) -> None:
    """Require one commit to be an ancestor of another without rewriting history."""
    result = git(repo, 'merge-base', '--is-ancestor', ancestor, descendant, check=False)
    if result.returncode != 0:
        raise ValueError(f'{ancestor} is not an ancestor of {descendant}')


def commit_and_push(  # pylint: disable=too-many-locals
    *,
    repo: Path,
    remote: str,
    branch: str,
    expected_head: str,
    commit_message: str,
    expected_remote_url: str = '',
) -> dict[str, Any]:
    """Stage exact report roots, commit once, and push without force or implicit refs."""
    ancestry = validate_repository_ancestry(
        repo,
        remote,
        branch,
        expected_head,
        commit_message,
        expected_remote_url,
    )
    head_before = ancestry['local_head']
    remote_before = ancestry['remote_head']
    scope = validate_prepublication_tree(repo, allow_staged_publish_paths=True)
    if ancestry['state'] == 'precommit':
        git(repo, 'add', '--', *PUBLISH_PATHS)
        staged = git(repo, 'diff', '--cached', '--name-only', '-z').stdout.split('\0')
        staged = [path for path in staged if path]
        if not staged:
            raise ValueError('verified SHAPO evidence produced no staged publication changes')
        if any(
            not any(path == root or path.startswith(f'{root}/') for root in PUBLISH_PATHS)
            for path in staged
        ):
            raise ValueError('Git index contains a path outside the exact publication roots')
        git(repo, 'diff', '--cached', '--check')
        git(repo, 'commit', '-m', commit_message, '--', *PUBLISH_PATHS)
        commit = git(repo, 'rev-parse', 'HEAD').stdout.strip()
        if commit == head_before or not FULL_COMMIT.fullmatch(commit):
            raise ValueError('publication commit was not created')
        require_ancestor(repo, head_before, commit)
    else:
        commit = head_before
        staged = [
            path
            for path in git(repo, 'show', '--format=', '--name-only', '-z', commit).stdout.split(
                '\0',
            )
            if path
        ]
    remote_recheck = remote_branch_oid(repo, remote, branch)
    if remote_recheck not in {remote_before, commit}:
        raise RuntimeError('remote branch changed before push; refusing non-reviewed integration')
    if remote_recheck == commit:
        push_stdout = 'remote already identifies publication commit'
    else:
        push = git(repo, 'push', '--porcelain', remote, f'{commit}:refs/heads/{branch}')
        push_stdout = push.stdout.strip()
    remote_after = remote_branch_oid(repo, remote, branch)
    if remote_after != commit:
        raise RuntimeError('remote branch does not identify the publication commit after push')
    remaining = status_entries(repo)
    if any(not (status == '??' and allowed_untracked(path)) for status, path in remaining):
        raise ValueError('unexpected worktree state remains after exact publication')
    return {
        'head_before': head_before,
        'remote_before': remote_before,
        'commit': commit,
        'remote_after': remote_after,
        'staged_file_count': len(staged),
        'preserved_user_entries': len(remaining),
        'push_stdout': push_stdout,
        'recovered_existing_commit': ancestry['state'] == 'committed',
        'prepublication_scope': scope,
    }


def publish_once(  # pylint: disable=too-many-arguments
    *,
    repo: Path,
    matrix_path: Path,
    source_commit: str,
    audit_script: Path,
    milestone_root: Path,
    final_root: Path,
    remote_host: str,
    remote_run_root: str,
    remote: str,
    branch: str,
    expected_head: str,
    commit_message: str,
    expected_remote_url: str = '',
) -> dict[str, Any]:
    """Wait for complete host evidence or publish it after independent validation."""
    readiness = host_evidence_readiness(matrix_path, milestone_root, final_root)
    if not readiness['ready']:
        ancestry = validate_repository_ancestry(
            repo,
            remote,
            branch,
            expected_head,
            expected_remote_url=expected_remote_url,
        )
        scope = validate_prepublication_tree(repo)
        return {
            'state': 'waiting',
            'reason': 'host_evidence_incomplete',
            'readiness': readiness,
            'repository': ancestry,
            'prepublication_scope': scope,
            'scientific_protocol_changed': False,
            'training_state_changed': False,
            'used_for_checkpoint_or_configuration_selection': False,
        }
    evidence = validate_host_evidence(
        repo=repo,
        matrix_path=matrix_path,
        source_commit=source_commit,
        audit_script=audit_script,
        milestone_root=milestone_root,
        final_root=final_root,
        remote_host=remote_host,
        remote_run_root=remote_run_root,
    )
    publication = commit_and_push(
        repo=repo,
        remote=remote,
        branch=branch,
        expected_head=expected_head,
        commit_message=commit_message,
        expected_remote_url=expected_remote_url,
    )
    return {
        'state': 'complete',
        'evidence': evidence,
        'publication': publication,
        'scientific_protocol_changed': False,
        'training_state_changed': False,
        'used_for_checkpoint_or_configuration_selection': False,
    }


def main() -> None:  # pylint: disable=too-many-locals
    """Poll host final evidence and publish the exact audited result roots once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--audit-script', type=Path, required=True)
    parser.add_argument('--milestone-root', type=Path, required=True)
    parser.add_argument('--final-root', type=Path, required=True)
    parser.add_argument('--remote-host', required=True)
    parser.add_argument('--remote-run-root', required=True)
    parser.add_argument('--remote', default='origin')
    parser.add_argument('--expected-remote-url', required=True)
    parser.add_argument('--branch', required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--commit-message', default='publish final SHAPO four-environment evidence')
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=600.0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not FULL_COMMIT.fullmatch(args.source_commit) or not FULL_COMMIT.fullmatch(
        args.expected_head,
    ):
        parser.error('source commit and expected HEAD must be full lowercase commit identities')
    if not SAFE_REMOTE.fullmatch(args.remote) or not SAFE_BRANCH.fullmatch(args.branch):
        parser.error('remote or branch name is unsafe')
    if args.poll_seconds <= 0.0:
        parser.error('--poll-seconds must be positive')

    repo = args.repo.resolve()
    matrix_path = args.matrix.resolve()
    audit_script = args.audit_script.resolve()
    milestone_root = args.milestone_root.resolve()
    final_root = args.final_root.resolve()
    state_dir = args.state_dir.resolve()
    if not (repo / '.git').exists():
        parser.error('--repo is not a Git worktree')
    if not matrix_path.is_file() or not audit_script.is_file():
        parser.error('matrix or milestone audit script is unavailable')
    initial_repository = validate_repository_ancestry(
        repo,
        args.remote,
        args.branch,
        args.expected_head,
        args.commit_message,
        args.expected_remote_url,
    )
    initial_remote = initial_repository['remote_head']
    state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(state_dir, 0o700)
    immutable_paths = {
        'github_publish_watcher': Path(__file__).resolve(),
        'matrix': matrix_path,
        'milestone_audit_script': audit_script,
        'strict_validator': audit_script.with_name('validate_shapo_p47_run.py'),
        'final_sync_validator': Path(__file__)
        .resolve()
        .with_name(
            'watch_shapo_p47_final_sync.py',
        ),
    }
    immutable_hashes = {name: sha256_file(path) for name, path in immutable_paths.items()}
    identities = {
        **{f'{name}_sha256': digest for name, digest in immutable_hashes.items()},
        'source_commit': args.source_commit,
        'expected_head': args.expected_head,
        'branch': args.branch,
        'remote': args.remote,
        'expected_remote_url': args.expected_remote_url,
        'initial_remote_commit': initial_remote,
        'initial_repository_state': initial_repository['state'],
        'scientific_protocol_changed': False,
        'training_state_changed': False,
        'used_for_checkpoint_or_configuration_selection': False,
    }

    with (state_dir / '.github-publish-watch.lock').open('w', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit('another GitHub publication watcher holds the state lock') from error
        while True:
            observed_at = datetime.now(timezone.utc).isoformat()
            try:
                for name, path in immutable_paths.items():
                    if sha256_file(path) != immutable_hashes[name]:
                        raise ValueError(f'publication dependency changed while running: {name}')
                snapshot = publish_once(
                    repo=repo,
                    matrix_path=matrix_path,
                    source_commit=args.source_commit,
                    audit_script=audit_script,
                    milestone_root=milestone_root,
                    final_root=final_root,
                    remote_host=args.remote_host,
                    remote_run_root=args.remote_run_root,
                    remote=args.remote,
                    branch=args.branch,
                    expected_head=args.expected_head,
                    commit_message=args.commit_message,
                    expected_remote_url=args.expected_remote_url,
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
                atomic_json(state_dir / 'github_publish_watch_alert.json', alert)
                raise
            resolved_alert = archive_resolved_alert(state_dir, observed_at)
            if resolved_alert is not None:
                snapshot['resolved_alert'] = resolved_alert
            atomic_json(state_dir / 'github_publish_watch_latest.json', snapshot)
            append_jsonl(
                state_dir / 'github_publish_watch_journal.jsonl',
                {
                    'observed_at_utc': observed_at,
                    'state': snapshot['state'],
                    'resolved_alert': resolved_alert is not None,
                    'scientific_protocol_changed': False,
                    'training_state_changed': False,
                    'used_for_checkpoint_or_configuration_selection': False,
                },
            )
            print(
                json.dumps(
                    {'observed_at_utc': observed_at, 'state': snapshot['state']},
                    sort_keys=True,
                ),
                flush=True,
            )
            if snapshot['state'] == 'complete':
                atomic_json(state_dir / 'github_publish_watch_complete.json', snapshot)
                return
            if args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
