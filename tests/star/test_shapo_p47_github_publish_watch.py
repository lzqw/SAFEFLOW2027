# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Tests for fail-closed publication of complete SHAPO host evidence."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.star.watch_shapo_p47_github_publish import (
    PUBLISH_PATHS,
    commit_and_push,
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
