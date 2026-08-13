# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Tests for fail-closed publication of complete SHAPO host evidence."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
from scripts.star.watch_shapo_p47_github_publish import (
    GIT_LOCAL_TIMEOUT_SECONDS,
    GIT_PUSH_TIMEOUT_SECONDS,
    GIT_REMOTE_READ_TIMEOUT_SECONDS,
    MAX_GITHUB_FILE_BYTES,
    PUBLISH_PATHS,
    archive_resolved_alert,
    commit_and_push,
)
from scripts.star.watch_shapo_p47_github_publish import git as publication_git
from scripts.star.watch_shapo_p47_github_publish import (
    host_evidence_readiness,
    publish_once,
    status_entries,
    validate_prepublication_tree,
    validate_repository_ancestry,
)


SOURCE = 'a' * 40


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ['git', '-C', str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / 'remote.git'
    repo = tmp_path / 'repo'
    subprocess.run(['git', 'init', '--bare', '--initial-branch=main', remote], check=True)
    subprocess.run(['git', 'init', '--initial-branch=main', repo], check=True)
    _git(repo, 'config', 'user.name', 'Publication Test')
    _git(repo, 'config', 'user.email', 'publication@example.invalid')
    (repo / 'baseline.txt').write_text('baseline\n', encoding='utf-8')
    _git(repo, 'add', 'baseline.txt')
    _git(repo, 'commit', '-m', 'baseline')
    _git(repo, 'remote', 'add', 'origin', str(remote))
    _git(repo, 'push', '-u', 'origin', 'main')
    return repo, remote


def _write_publish_artifacts(repo: Path) -> None:
    milestone = repo / PUBLISH_PATHS[0]
    final = repo / PUBLISH_PATHS[1]
    milestone.mkdir(parents=True)
    final.mkdir(parents=True)
    (milestone / 'milestone.txt').write_text('sealed\n', encoding='utf-8')
    (final / 'final.txt').write_text('verified\n', encoding='utf-8')


def test_status_parser_and_scope_preserve_known_user_directories(tmp_path: Path) -> None:
    repo, _remote = _repository(tmp_path)
    _write_publish_artifacts(repo)
    (repo / 'METHOD').mkdir()
    (repo / 'METHOD/notes.md').write_text('mine\n', encoding='utf-8')
    entries = status_entries(repo)
    assert ('??', 'METHOD/notes.md') in entries
    result = validate_prepublication_tree(repo)
    assert result['preserved_user_entries'] == 1
    assert result['publish_entries'] == 2


def test_scope_rejects_unrelated_change(tmp_path: Path) -> None:
    repo, _remote = _repository(tmp_path)
    _write_publish_artifacts(repo)
    (repo / 'unexpected.txt').write_text('no\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unexpected worktree changes'):
        validate_prepublication_tree(repo)


def test_ready_transaction_accepts_only_staged_publish_paths(tmp_path: Path) -> None:
    repo, _remote = _repository(tmp_path)
    _write_publish_artifacts(repo)
    _git(repo, 'add', *PUBLISH_PATHS)
    result = validate_prepublication_tree(repo, allow_staged_publish_paths=True)
    assert result['publish_entries'] == 2
    with pytest.raises(ValueError, match='unexpected worktree changes'):
        validate_prepublication_tree(repo)


def test_repository_identity_rejects_changed_push_url(tmp_path: Path) -> None:
    repo, _remote = _repository(tmp_path)
    with pytest.raises(ValueError, match='push URL changed'):
        validate_repository_ancestry(
            repo,
            'origin',
            'main',
            _git(repo, 'rev-parse', 'HEAD'),
            expected_remote_url='git@github.com:lzqw/SAFEFLOW2027.git',
        )


def test_repository_identity_accepts_credential_free_https_and_rejects_userinfo(
    tmp_path: Path,
) -> None:
    repo, remote = _repository(tmp_path)
    expected = _git(repo, 'rev-parse', 'HEAD')
    https_url = 'https://github.com/lzqw/SAFEFLOW2027.git'
    _git(repo, 'remote', 'set-url', '--push', 'origin', https_url)
    result = validate_repository_ancestry(
        repo,
        'origin',
        'main',
        expected,
        expected_remote_url=https_url,
    )
    assert result['state'] == 'precommit'
    assert result['remote_head'] == expected
    with pytest.raises(ValueError, match='unsafe or contains credentials'):
        validate_repository_ancestry(
            repo,
            'origin',
            'main',
            expected,
            expected_remote_url='https://token@github.com/lzqw/SAFEFLOW2027.git',
        )


def test_github_file_limit_is_conservative() -> None:
    assert MAX_GITHUB_FILE_BYTES == 95 * 1024 * 1024
    assert MAX_GITHUB_FILE_BYTES < 100 * 1024 * 1024


def test_git_commands_have_bounded_operation_specific_timeouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, int]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.append((command[3], int(kwargs['timeout'])))
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_github_publish.subprocess.run',
        fake_run,
    )
    publication_git(tmp_path, 'ls-remote', 'origin')
    publication_git(tmp_path, 'push', 'origin', 'HEAD')
    publication_git(tmp_path, 'status', '--short')
    assert observed == [
        ('ls-remote', GIT_REMOTE_READ_TIMEOUT_SECONDS),
        ('push', GIT_PUSH_TIMEOUT_SECONDS),
        ('status', GIT_LOCAL_TIMEOUT_SECONDS),
    ]


def test_git_timeout_is_fail_closed_without_leaking_command_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(['git', 'ls-remote', 'secret'], 120)

    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_github_publish.subprocess.run',
        timeout,
    )
    with pytest.raises(RuntimeError, match='git ls-remote exceeded 120 seconds') as error:
        publication_git(tmp_path, 'ls-remote', 'origin')
    assert 'secret' not in str(error.value)


def test_successful_poll_archives_active_alert_without_changing_evidence(tmp_path: Path) -> None:
    state_dir = tmp_path / 'state'
    state_dir.mkdir()
    active = state_dir / 'github_publish_watch_alert.json'
    raw = (
        b'{\n'
        b'  "observed_at_utc": "2026-08-13T14:38:56.584897+00:00",\n'
        b'  "state": "alert",\n'
        b'  "error": "temporary authentication failure"\n'
        b'}\n'
    )
    active.write_bytes(raw)

    result = archive_resolved_alert(state_dir, '2026-08-13T14:43:32.107887+00:00')

    assert result is not None
    assert result['alert_observed_at_utc'] == '2026-08-13T14:38:56.584897+00:00'
    assert result['resolved_at_utc'] == '2026-08-13T14:43:32.107887+00:00'
    archived = state_dir / result['archive_path']
    assert archived.read_bytes() == raw
    assert result['archive_sha256'] == hashlib.sha256(raw).hexdigest()
    assert not active.exists()
    assert archived.stat().st_mode & 0o777 == 0o600


def test_resolved_alert_archive_is_idempotent_and_rejects_symlink(tmp_path: Path) -> None:
    state_dir = tmp_path / 'state'
    state_dir.mkdir()
    active = state_dir / 'github_publish_watch_alert.json'
    raw = b'{"observed_at_utc":"2026-08-13T14:38:56+00:00","state":"alert"}\n'
    active.write_bytes(raw)
    first = archive_resolved_alert(state_dir, '2026-08-13T14:43:32+00:00')
    active.write_bytes(raw)
    second = archive_resolved_alert(state_dir, '2026-08-13T14:44:32+00:00')
    assert first is not None and second is not None
    assert first['archive_path'] == second['archive_path']
    assert not active.exists()
    assert (state_dir / second['archive_path']).stat().st_mode & 0o777 == 0o600

    target = state_dir / 'outside.json'
    target.write_bytes(raw)
    active.symlink_to(target)
    with pytest.raises(ValueError, match='not a regular file'):
        archive_resolved_alert(state_dir, '2026-08-13T14:45:32+00:00')


def test_resolved_alert_rejects_time_reversal(tmp_path: Path) -> None:
    state_dir = tmp_path / 'state'
    state_dir.mkdir()
    (state_dir / 'github_publish_watch_alert.json').write_text(
        '{"observed_at_utc":"2026-08-13T14:43:33+00:00","state":"alert"}\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='cannot resolve before'):
        archive_resolved_alert(state_dir, '2026-08-13T14:43:32+00:00')


def test_commit_and_push_exact_paths_once(tmp_path: Path) -> None:
    repo, remote = _repository(tmp_path)
    _write_publish_artifacts(repo)
    (repo / 'PAPER').mkdir()
    (repo / 'PAPER/draft.md').write_text('private draft\n', encoding='utf-8')
    head = _git(repo, 'rev-parse', 'HEAD')
    result = commit_and_push(
        repo=repo,
        remote='origin',
        branch='main',
        expected_head=head,
        commit_message='publish evidence',
    )
    assert result['remote_before'] == head
    assert result['remote_after'] == result['commit']
    assert result['staged_file_count'] == 2
    assert _git(repo, 'show', '--format=', '--name-only', 'HEAD').splitlines() == [
        f'{PUBLISH_PATHS[1]}/final.txt',
        f'{PUBLISH_PATHS[0]}/milestone.txt',
    ]
    assert (repo / 'PAPER/draft.md').is_file()
    assert 'PAPER/draft.md' not in _git(remote, 'ls-tree', '-r', '--name-only', 'main')


def test_commit_and_push_recovers_preexisting_exact_index(tmp_path: Path) -> None:
    repo, _remote = _repository(tmp_path)
    _write_publish_artifacts(repo)
    expected = _git(repo, 'rev-parse', 'HEAD')
    _git(repo, 'add', *PUBLISH_PATHS)
    result = commit_and_push(
        repo=repo,
        remote='origin',
        branch='main',
        expected_head=expected,
        commit_message='publish evidence',
    )
    assert result['commit'] != expected
    assert result['remote_after'] == result['commit']
    assert result['staged_file_count'] == 2


def test_commit_refuses_remote_divergence(tmp_path: Path) -> None:
    repo, remote = _repository(tmp_path)
    _write_publish_artifacts(repo)
    peer = tmp_path / 'peer'
    subprocess.run(['git', 'clone', '-q', remote, peer], check=True)
    _git(peer, 'config', 'user.name', 'Peer')
    _git(peer, 'config', 'user.email', 'peer@example.invalid')
    (peer / 'peer.txt').write_text('peer\n', encoding='utf-8')
    _git(peer, 'add', 'peer.txt')
    _git(peer, 'commit', '-m', 'peer update')
    _git(peer, 'push', 'origin', 'main')
    with pytest.raises(ValueError, match='is not an ancestor'):
        commit_and_push(
            repo=repo,
            remote='origin',
            branch='main',
            expected_head=_git(repo, 'rev-parse', 'HEAD'),
            commit_message='publish evidence',
        )


def test_commit_recovers_exact_local_publication_transaction(tmp_path: Path) -> None:
    repo, _remote = _repository(tmp_path)
    _write_publish_artifacts(repo)
    expected = _git(repo, 'rev-parse', 'HEAD')
    _git(repo, 'add', *PUBLISH_PATHS)
    _git(repo, 'commit', '-m', 'publish evidence')
    committed = _git(repo, 'rev-parse', 'HEAD')
    result = commit_and_push(
        repo=repo,
        remote='origin',
        branch='main',
        expected_head=expected,
        commit_message='publish evidence',
    )
    assert result['commit'] == committed
    assert result['remote_after'] == committed
    assert result['recovered_existing_commit'] is True


def test_commit_rejects_nonpublication_local_commit(tmp_path: Path) -> None:
    repo, _remote = _repository(tmp_path)
    _write_publish_artifacts(repo)
    expected = _git(repo, 'rev-parse', 'HEAD')
    (repo / 'unrelated.txt').write_text('unrelated\n', encoding='utf-8')
    _git(repo, 'add', 'unrelated.txt')
    _git(repo, 'commit', '-m', 'publish evidence')
    with pytest.raises(ValueError, match='exact recoverable publication transaction'):
        commit_and_push(
            repo=repo,
            remote='origin',
            branch='main',
            expected_head=expected,
            commit_message='publish evidence',
        )


def test_startup_ancestry_accepts_exact_recoverable_commit(tmp_path: Path) -> None:
    repo, _remote = _repository(tmp_path)
    _write_publish_artifacts(repo)
    expected = _git(repo, 'rev-parse', 'HEAD')
    _git(repo, 'add', *PUBLISH_PATHS)
    _git(repo, 'commit', '-m', 'publish evidence')
    result = validate_repository_ancestry(
        repo,
        'origin',
        'main',
        expected,
        'publish evidence',
    )
    assert result['state'] == 'committed'
    assert result['remote_head'] == expected


def test_publish_once_waits_without_validating_or_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _remote = _repository(tmp_path)
    matrix = repo / 'matrix.yaml'
    matrix.write_text('{}\n', encoding='utf-8')
    _git(repo, 'add', 'matrix.yaml')
    _git(repo, 'commit', '-m', 'add matrix')
    _git(repo, 'push', 'origin', 'main')
    calls: list[str] = []
    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_github_publish.host_evidence_readiness',
        lambda *_args: {
            'ready': False,
            'final_host_transfer_complete': False,
            'host_milestone_seals': 0,
            'expected_host_milestone_seals': 60,
        },
    )
    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_github_publish.validate_host_evidence',
        lambda **_kwargs: calls.append('validate'),
    )
    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_github_publish.commit_and_push',
        lambda **_kwargs: calls.append('publish'),
    )
    result = publish_once(
        repo=repo,
        matrix_path=matrix,
        source_commit=SOURCE,
        audit_script=repo / 'audit.py',
        milestone_root=repo / PUBLISH_PATHS[0],
        final_root=repo / PUBLISH_PATHS[1],
        remote_host='seeta-nmb2',
        remote_run_root='/remote/formal',
        remote='origin',
        branch='main',
        expected_head=_git(repo, 'rev-parse', 'HEAD'),
        commit_message='publish evidence',
    )
    assert result['state'] == 'waiting'
    assert calls == []
    assert _git(repo, 'status', '--porcelain') == ''


def test_publish_once_ready_validates_then_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _remote = _repository(tmp_path)
    matrix = repo / 'matrix.yaml'
    matrix.write_text('{}\n', encoding='utf-8')
    _git(repo, 'add', 'matrix.yaml')
    _git(repo, 'commit', '-m', 'add matrix')
    _git(repo, 'push', 'origin', 'main')
    calls: list[str] = []
    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_github_publish.host_evidence_readiness',
        lambda *_args: {'ready': True},
    )

    def fake_validate(**_kwargs: object) -> dict[str, object]:
        calls.append('validate')
        return {'milestone_jobs': 60, 'status': 'passed'}

    def fake_publish(**_kwargs: object) -> dict[str, object]:
        calls.append('publish')
        return {'commit': 'b' * 40}

    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_github_publish.validate_host_evidence',
        fake_validate,
    )
    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_github_publish.commit_and_push',
        fake_publish,
    )
    result = publish_once(
        repo=repo,
        matrix_path=matrix,
        source_commit=SOURCE,
        audit_script=repo / 'audit.py',
        milestone_root=repo / PUBLISH_PATHS[0],
        final_root=repo / PUBLISH_PATHS[1],
        remote_host='seeta-nmb2',
        remote_run_root='/remote/formal',
        remote='origin',
        branch='main',
        expected_head=_git(repo, 'rev-parse', 'HEAD'),
        commit_message='publish evidence',
    )
    assert result['state'] == 'complete'
    assert result['evidence']['milestone_jobs'] == 60
    assert result['publication']['commit'] == 'b' * 40
    assert calls == ['validate', 'publish']


def test_waiting_state_rejects_unrelated_worktree_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _remote = _repository(tmp_path)
    matrix = repo / 'matrix.yaml'
    matrix.write_text('{}\n', encoding='utf-8')
    _git(repo, 'add', 'matrix.yaml')
    _git(repo, 'commit', '-m', 'add matrix')
    _git(repo, 'push', 'origin', 'main')
    (repo / 'unexpected.txt').write_text('no\n', encoding='utf-8')
    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_github_publish.host_evidence_readiness',
        lambda *_args: {'ready': False},
    )
    with pytest.raises(ValueError, match='unexpected worktree changes'):
        publish_once(
            repo=repo,
            matrix_path=matrix,
            source_commit=SOURCE,
            audit_script=repo / 'audit.py',
            milestone_root=repo / PUBLISH_PATHS[0],
            final_root=repo / PUBLISH_PATHS[1],
            remote_host='seeta-nmb2',
            remote_run_root='/remote/formal',
            remote='origin',
            branch='main',
            expected_head=_git(repo, 'rev-parse', 'HEAD'),
            commit_message='publish evidence',
        )


def test_readiness_waits_for_both_independent_host_syncs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = tmp_path / 'matrix.yaml'
    matrix.write_text('{}\n', encoding='utf-8')
    milestone_root = tmp_path / 'milestones'
    final_root = tmp_path / 'final'
    jobs = []
    for index in range(60):
        output = milestone_root / str(index)
        output.mkdir(parents=True)
        jobs.append(type('Job', (), {'output': output})())
    monkeypatch.setattr(
        'scripts.star.watch_shapo_p47_github_publish.locked_jobs',
        lambda *_args: jobs,
    )
    final_root.mkdir()
    (final_root / 'transfer_complete').touch()
    first = host_evidence_readiness(matrix, milestone_root, final_root)
    assert first == {
        'ready': False,
        'final_host_transfer_complete': True,
        'host_milestone_seals': 0,
        'expected_host_milestone_seals': 60,
    }
    for job in jobs:
        (job.output / 'milestone_audit_complete').touch()
    assert host_evidence_readiness(matrix, milestone_root, final_root)['ready'] is True
