# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Tests for fail-closed SHAPO remote process recovery."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
import yaml

from scripts.star.shapo_p47_remote_recovery_agent import (
    _run_locked,
    formal_worker_processes,
    formal_worker_pids,
    process_argv,
    recoverable_checkpoints,
    resolve_exact_process,
    startup_grace_active,
    validate_checkpoint_payloads,
    watchdog_startup_readiness,
)
from scripts.star.watch_shapo_p47_remote_recovery import (
    RemoteRecoveryError,
    build_payload,
    invoke_remote,
    parse_lock,
)


COMMIT = 'a' * 40
DIGEST = 'b' * 64


def _cmdline(
    proc_root: Path,
    pid: int,
    argv: list[str],
    *,
    ppid: int = 1,
    cwd: Path | None = None,
) -> None:
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    (process / 'cmdline').write_bytes(b'\0'.join(value.encode() for value in argv) + b'\0')
    (process / 'status').write_text(f'Name:\ttest\nPPid:\t{ppid}\n', encoding='utf-8')
    os.symlink(cwd or proc_root, process / 'cwd')


def _arguments(tmp_path: Path) -> argparse.Namespace:
    prefix = '/remote'
    args = argparse.Namespace(
        remote_python_bin=f'{prefix}/python',
        formal_source=f'{prefix}/formal',
        formal_source_commit=COMMIT,
        run_root=f'{prefix}/run',
        smoke_root=f'{prefix}/smoke',
        monitor_source=f'{prefix}/monitor',
        monitor_source_commit='c' * 40,
        monitor_output=f'{prefix}/run/watchdog',
        milestone_source=f'{prefix}/milestone',
        milestone_source_commit='d' * 40,
        milestone_output=f'{prefix}/run/milestones',
        validation_source=f'{prefix}/validation',
        validation_source_commit='e' * 40,
        completion_state=f'{prefix}/completion',
        recovery_root=f'{prefix}/recovery',
        locked_file=[],
        resolved_config_sha256=[
            f'{slug}={DIGEST}' for slug in ('point-goal', 'point-button', 'ant', 'walker2d')
        ],
        expected_torch_version='2.8.0+cu128',
        expected_cuda_device='NVIDIA GeForce RTX 3090',
        minimum_free_gib=4.0,
        state_dir=tmp_path,
        poll_seconds=300.0,
        observe_only=False,
        once=True,
    )
    paths = {
        args.remote_python_bin,
        f'{args.formal_source}/scripts/star/launch_shapo_p47_formal.sh',
        f'{args.formal_source}/configs/star/shapo_p47_four_env.yaml',
        f'{args.smoke_root}/smoke_gate.json',
        f'{args.monitor_source}/scripts/star/watch_shapo_p47_formal.py',
        f'{args.monitor_source}/scripts/star/status_shapo_p47_formal.py',
        f'{args.milestone_source}/scripts/star/watch_shapo_p47_milestones.py',
        f'{args.milestone_source}/scripts/star/audit_shapo_p47_milestone.py',
        f'{args.milestone_source}/scripts/star/validate_shapo_p47_run.py',
        f'{args.validation_source}/scripts/star/watch_shapo_p47_completion.py',
        f'{args.validation_source}/scripts/star/finalize_shapo_p47_formal.sh',
        f'{args.validation_source}/scripts/star/finalize_shapo_p47_formal.py',
        f'{args.validation_source}/configs/star/shapo_p47_four_env.yaml',
        f'{args.validation_source}/scripts/star/audit_shapo_p47_milestone.py',
        f'{args.validation_source}/scripts/star/validate_shapo_p47_run.py',
    }
    args.locked_file = [f'{path}={DIGEST}' for path in sorted(paths)]
    return args


def test_process_identity_requires_exact_argv(tmp_path: Path) -> None:
    proc = tmp_path / 'proc'
    proc.mkdir()
    exact = ['/python', 'worker.py', '--frozen']
    _cmdline(proc, 10, exact)
    _cmdline(proc, 11, [*exact, '--changed'])
    assert process_argv(10, proc) == exact
    pid_file = tmp_path / 'worker.pid'
    pid_file.write_text('11\n', encoding='utf-8')
    state = resolve_exact_process(
        name='worker',
        expected=exact,
        expected_cwd=proc.resolve(),
        pid_file=pid_file,
        proc_root=proc,
    )
    assert state == {
        'alive': True,
        'pid': 10,
        'adopted_pid': True,
        'pid_file_repair_needed': False,
        'matches': [10],
    }
    assert pid_file.read_text(encoding='utf-8') == '10\n'


def test_supervisor_adopts_only_parent_one_and_rejects_duplicates(tmp_path: Path) -> None:
    proc = tmp_path / 'proc'
    proc.mkdir()
    exact = ['bash', '/formal/launch.sh']
    _cmdline(proc, 20, exact, ppid=1)
    _cmdline(proc, 21, exact, ppid=20)
    state = resolve_exact_process(
        name='supervisor',
        expected=exact,
        expected_cwd=proc.resolve(),
        pid_file=tmp_path / 'supervisor.pid',
        supervisor=True,
        proc_root=proc,
    )
    assert state['pid'] == 20
    assert state['matches'] == [20, 21]
    _cmdline(proc, 22, exact, ppid=1)
    with pytest.raises(RuntimeError, match='ambiguous exact supervisor'):
        resolve_exact_process(
            name='supervisor',
            expected=exact,
            expected_cwd=proc.resolve(),
            pid_file=tmp_path / 'supervisor.pid',
            supervisor=True,
            proc_root=proc,
        )


def test_formal_worker_scan_is_scoped_to_explicit_config_root(tmp_path: Path) -> None:
    proc = tmp_path / 'proc'
    proc.mkdir()
    run_root = Path('/formal')
    _cmdline(
        proc,
        30,
        ['/python', 'scripts/star/train_star.py', '--config', '/formal/runs/ant/config.yaml'],
    )
    _cmdline(
        proc,
        31,
        ['/python', 'scripts/star/train_star.py', '--config', '/other/runs/ant/config.yaml'],
    )
    _cmdline(proc, 32, ['/python', 'scripts/star/train_star.py'])
    assert formal_worker_pids(run_root, proc) == [30]


def test_formal_worker_scan_rejects_duplicate_run_workers(tmp_path: Path) -> None:
    proc = tmp_path / 'proc'
    proc.mkdir()
    exact = ['/python', 'scripts/star/train_star.py', '--config', '/formal/runs/ant/config.yaml']
    _cmdline(proc, 33, exact)
    _cmdline(proc, 34, exact)
    with pytest.raises(RuntimeError, match='ambiguous formal workers'):
        formal_worker_processes(Path('/formal'), proc)


def test_payload_is_fully_locked_and_rejects_duplicate_files(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    payload = build_payload(args, allow_recovery=True)
    assert payload['allow_recovery'] is True
    assert len(payload['worktree_commits']) == 4
    assert payload['file_sha256'] == dict(
        parse_lock(value, digest=True) for value in args.locked_file
    )
    assert parse_lock(f'/locked/file={DIGEST}', digest=True) == ('/locked/file', DIGEST)
    args.locked_file *= 2
    with pytest.raises(ValueError, match='nonempty and unique'):
        build_payload(args, allow_recovery=True)


def test_payload_rejects_unsafe_path_and_invalid_identity(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    args.run_root = '/remote/../unsafe'
    with pytest.raises(ValueError, match='safe absolute'):
        build_payload(args, allow_recovery=False)
    args = _arguments(tmp_path)
    args.formal_source_commit = 'short'
    with pytest.raises(ValueError, match='worktree commits'):
        build_payload(args, allow_recovery=False)


def _completed(
    stdout: str,
    *,
    returncode: int = 0,
    stderr: str = '',
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(['ssh'], returncode, stdout=stdout, stderr=stderr)


def test_remote_invocation_parses_one_success_record(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {'version': 'shapo-p47-remote-recovery-v1'}
    response = {'version': payload['version'], 'state': 'running'}
    seen = {}

    def fake_run(command, **kwargs):
        seen['command'] = command
        seen.update(kwargs)
        return _completed(json.dumps(response) + '\n')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    assert (
        invoke_remote(
            host='seeta-nmb2',
            remote_python='/remote/python',
            payload=payload,
            agent_source='print()\n',
        )
        == response
    )
    assert seen['command'][:6] == [
        'ssh',
        '-o',
        'BatchMode=yes',
        '-o',
        'ConnectTimeout=15',
        'seeta-nmb2',
    ]
    assert seen['input'] == 'print()\n'
    assert 'shell' not in seen


def test_remote_invocation_distinguishes_connection_and_agent_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {'version': 'shapo-p47-remote-recovery-v1'}
    monkeypatch.setattr(
        subprocess,
        'run',
        lambda *_args, **_kwargs: _completed('', returncode=255, stderr='offline'),
    )
    with pytest.raises(ConnectionError, match='SSH failed'):
        invoke_remote(
            host='seeta-nmb2',
            remote_python='/remote/python',
            payload=payload,
            agent_source='source',
        )
    alert = {'version': payload['version'], 'state': 'alert', 'error': 'orphan worker'}
    monkeypatch.setattr(
        subprocess,
        'run',
        lambda *_args, **_kwargs: _completed(json.dumps(alert), returncode=2),
    )
    with pytest.raises(RemoteRecoveryError, match='orphan worker') as error:
        invoke_remote(
            host='seeta-nmb2',
            remote_python='/remote/python',
            payload=payload,
            agent_source='source',
        )
    assert error.value.payload == alert


def _remote_payload(tmp_path: Path) -> dict:
    run_root = tmp_path / 'formal'
    monitor = run_root / 'watchdog'
    milestones = run_root / 'milestones'
    completion = tmp_path / 'completion'
    recovery = tmp_path / 'recovery'
    for path in (run_root, monitor, milestones, completion, recovery):
        path.mkdir(parents=True, exist_ok=True)
    return {
        'python_bin': '/python',
        'formal_source': '/formal-source',
        'run_root': str(run_root),
        'smoke_root': '/smoke',
        'monitor_source': '/monitor-source',
        'monitor_output': str(monitor),
        'milestone_source': '/milestone-source',
        'milestone_output': str(milestones),
        'validation_source': '/validation-source',
        'completion_state': str(completion),
        'recovery_root': str(recovery),
        'worktree_commits': {'/formal-source': COMMIT},
        'file_sha256': {'/formal-source/matrix.yaml': DIGEST},
        'resolved_config_sha256': dict.fromkeys(
            ('point-goal', 'point-button', 'ant', 'walker2d'),
            DIGEST,
        ),
        'expected_torch_version': '2.8.0+cu128',
        'expected_cuda_device': 'NVIDIA GeForce RTX 3090',
        'minimum_free_bytes': 4 * 1024**3,
        'allow_recovery': True,
    }


def test_checkpoint_inventory_only_requires_started_runs_during_restart(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / 'formal'
    progress = run_root / 'runs/ant-2027/training/segment/progress.csv'
    progress.parent.mkdir(parents=True)
    progress.write_text('TotalEnvSteps\n5000\n', encoding='utf-8')
    assert recoverable_checkpoints(run_root) == {}
    with pytest.raises(RuntimeError, match='lacks latest_full recovery'):
        recoverable_checkpoints(run_root, require_started=True)


def _patch_transaction(
    monkeypatch: pytest.MonkeyPatch,
    *,
    alive: dict[str, bool],
    workers: dict[str, int] | None = None,
) -> tuple[list[str], list[str]]:
    launches = []
    checkpoint_validations = []
    monkeypatch.setattr(
        'scripts.star.shapo_p47_remote_recovery_agent.process_specs',
        lambda _payload: {
            name: {
                'argv': [name],
                'cwd': '/',
                'pid_file': f'/{name}.pid',
                'log': f'/{name}.log',
                'supervisor': name == 'supervisor',
            }
            for name in ('supervisor', 'watchdog', 'milestone', 'completion')
        },
    )

    def resolve(*, name, **_kwargs):
        return {
            'alive': alive[name],
            'pid': 100 if alive[name] else None,
            'adopted_pid': False,
            'matches': [100] if alive[name] else [],
        }

    monkeypatch.setattr(
        'scripts.star.shapo_p47_remote_recovery_agent.resolve_exact_process',
        resolve,
    )
    monkeypatch.setattr(
        'scripts.star.shapo_p47_remote_recovery_agent.formal_worker_processes',
        lambda _root: dict(workers or {}),
    )
    monkeypatch.setattr(
        'scripts.star.shapo_p47_remote_recovery_agent.recoverable_checkpoints',
        lambda _root, **_kwargs: {'ant-2027': {'path': '/checkpoint', 'bytes': 1}},
    )
    monkeypatch.setattr(
        'scripts.star.shapo_p47_remote_recovery_agent.validate_checkpoint_payloads',
        lambda *_args: checkpoint_validations.append('validated'),
    )
    monkeypatch.setattr(
        'scripts.star.shapo_p47_remote_recovery_agent.validate_restart_environment',
        lambda _payload: {'status': 'passed'},
    )

    def launch(name, _spec):
        launches.append(name)
        return {'pid': 200 + len(launches), 'alive': True, 'completed': False}

    monkeypatch.setattr(
        'scripts.star.shapo_p47_remote_recovery_agent.launch_exact',
        launch,
    )
    monkeypatch.setattr(
        'scripts.star.shapo_p47_remote_recovery_agent.durable_alerts',
        lambda _payload: [],
    )
    monkeypatch.setattr(
        'scripts.star.shapo_p47_remote_recovery_agent.matrix_failure_count',
        lambda _root: None,
    )
    return launches, checkpoint_validations


def test_recovery_transaction_is_noop_while_exact_graph_is_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _remote_payload(tmp_path)
    launches, validations = _patch_transaction(
        monkeypatch,
        alive=dict.fromkeys(('supervisor', 'watchdog', 'milestone', 'completion'), True),
        workers={f'run-{pid}': pid for pid in (11, 12, 13, 14)},
    )
    result = _run_locked(payload, Path(payload['recovery_root']))
    assert result['state'] == 'running'
    assert result['formal_worker_pids'] == [11, 12, 13, 14]
    assert launches == []
    assert validations == []


def test_recovery_transaction_rejects_orphan_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _remote_payload(tmp_path)
    alive = dict.fromkeys(('supervisor', 'watchdog', 'milestone', 'completion'), False)
    launches, validations = _patch_transaction(
        monkeypatch,
        alive=alive,
        workers={'ant-2027': 77},
    )
    with pytest.raises(RuntimeError, match='workers are alive without'):
        _run_locked(payload, Path(payload['recovery_root']))
    assert launches == []
    assert validations == []


def test_recovery_transaction_validates_checkpoints_then_restores_full_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _remote_payload(tmp_path)
    alive = dict.fromkeys(('supervisor', 'watchdog', 'milestone', 'completion'), False)
    launches, validations = _patch_transaction(monkeypatch, alive=alive)
    result = _run_locked(payload, Path(payload['recovery_root']))
    assert result['state'] == 'recovered'
    assert validations == ['validated']
    assert launches == ['supervisor', 'milestone', 'completion']
    assert result['deferred_processes'] == ['watchdog']
    assert sorted(result['launched_processes']) == sorted(launches)
    assert result['formal_training_process_restarted'] is True


def test_followup_recovery_launches_watchdog_after_workers_reattach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _remote_payload(tmp_path)
    alive = {
        'supervisor': True,
        'watchdog': False,
        'milestone': True,
        'completion': True,
    }
    launches, validations = _patch_transaction(
        monkeypatch,
        alive=alive,
        workers={'ant-2027': 77},
    )
    result = _run_locked(payload, Path(payload['recovery_root']))
    assert launches == ['watchdog']
    assert validations == []
    assert result['deferred_processes'] == []
    assert result['watchdog_startup_readiness']['ready'] is True


def test_watchdog_waits_only_during_bounded_supervisor_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _remote_payload(tmp_path)
    recovery_root = Path(payload['recovery_root'])
    attempt = {
        'observed_at_utc': '2026-08-12T13:00:00+00:00',
        'formal_training_process_restart_planned': True,
        'recoverable_checkpoints': {},
    }
    (recovery_root / 'recovery_attempts.jsonl').write_text(
        json.dumps(attempt) + '\n',
        encoding='utf-8',
    )
    alive = {
        'supervisor': True,
        'watchdog': False,
        'milestone': True,
        'completion': True,
    }
    _patch_transaction(monkeypatch, alive=alive)
    monkeypatch.setattr(
        'scripts.star.shapo_p47_remote_recovery_agent.datetime',
        _FixedDatetime,
    )
    result = _run_locked(payload, recovery_root)
    assert result['state'] == 'recovery_deferred'
    assert result['launched_processes'] == {}
    assert result['deferred_processes'] == ['watchdog']
    assert result['watchdog_startup_readiness']['missing_worker_runs'] == ['ant-2027']


def test_startup_grace_is_timezone_aware_and_bounded() -> None:
    attempt = {
        'observed_at_utc': '2026-08-12T13:00:00+00:00',
        'formal_training_process_restart_planned': True,
    }
    assert startup_grace_active(
        attempt,
        observed_at=datetime.fromisoformat('2026-08-12T13:15:00+00:00'),
    )
    assert not startup_grace_active(
        attempt,
        observed_at=datetime.fromisoformat('2026-08-12T13:15:00.001+00:00'),
    )
    assert not startup_grace_active(
        {'formal_training_process_restart_planned': False},
        observed_at=datetime.now(timezone.utc),
    )


def test_watchdog_readiness_excludes_only_durably_completed_runs(tmp_path: Path) -> None:
    run_root = tmp_path / 'formal'
    completed = run_root / 'runs/ant-2027'
    completed.mkdir(parents=True)
    (completed / 'formal_validation.json').write_text(
        json.dumps({'status': 'passed'}),
        encoding='utf-8',
    )
    checkpoints = {
        'ant-2027': {'path': '/ant-checkpoint'},
        'walker2d-2027': {'path': '/walker-checkpoint'},
    }
    with pytest.raises(RuntimeError, match='exit seal'):
        watchdog_startup_readiness(run_root, checkpoints, {})
    (completed / 'exit_code.txt').write_text('0\n', encoding='utf-8')
    readiness = watchdog_startup_readiness(
        run_root,
        checkpoints,
        {'walker2d-2027': 77},
    )
    assert readiness['ready'] is True
    assert readiness['completed_checkpoint_runs'] == ['ant-2027']


class _FixedDatetime:
    """Minimal datetime facade for deterministic startup-grace tests."""

    @staticmethod
    def now(_timezone):
        return datetime.fromisoformat('2026-08-12T13:05:00+00:00')

    @staticmethod
    def fromisoformat(value):
        return datetime.fromisoformat(value)


def test_finalization_window_restores_only_completion_watcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _remote_payload(tmp_path)
    run_root = Path(payload['run_root'])
    (run_root / 'formal_matrix_complete').touch()
    (Path(payload['monitor_output']) / 'completion_health.json').touch()
    (Path(payload['milestone_output']) / 'milestone_watch_complete').touch()
    public = run_root / 'public_artifacts'
    public.mkdir()
    (public / 'finalization_complete').touch()
    alive = {
        'supervisor': False,
        'watchdog': False,
        'milestone': False,
        'completion': False,
    }
    launches, validations = _patch_transaction(monkeypatch, alive=alive)
    result = _run_locked(payload, Path(payload['recovery_root']))
    assert result['state'] == 'recovered'
    assert launches == ['completion']
    assert validations == []


def test_complete_transaction_launches_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _remote_payload(tmp_path)
    run_root = Path(payload['run_root'])
    (run_root / 'formal_matrix_complete').touch()
    (Path(payload['monitor_output']) / 'completion_health.json').touch()
    (Path(payload['milestone_output']) / 'milestone_watch_complete').touch()
    public = run_root / 'public_artifacts'
    public.mkdir()
    (public / 'finalization_complete').touch()
    (Path(payload['completion_state']) / 'completion_watch_complete.json').touch()
    alive = dict.fromkeys(('supervisor', 'watchdog', 'milestone', 'completion'), False)
    launches, validations = _patch_transaction(monkeypatch, alive=alive)
    result = _run_locked(payload, Path(payload['recovery_root']))
    assert result['state'] == 'complete'
    assert result['needed_processes'] == []
    assert launches == []
    assert validations == []


def _checkpoint_fixture(tmp_path: Path) -> tuple[Path, dict, Path, Path]:
    run_root = tmp_path / 'formal'
    container = run_root / 'runs/ant-2027'
    checkpoint = container / 'training/segment/checkpoints/latest_full.pt'
    checkpoint.parent.mkdir(parents=True)
    resolved = {
        'env_id': 'Ant-v4',
        'custom_cfgs': {
            'algo_cfgs': {
                'directional_loss_weight': 1.0,
                'checkpoint_replay': True,
                'resume_from': '',
            },
            'train_cfgs': {'total_steps': 5_000_000},
        },
    }
    (container / 'resolved_formal_config.yaml').write_text(
        yaml.safe_dump(resolved),
        encoding='utf-8',
    )
    matrix = tmp_path / 'matrix.yaml'
    matrix.write_text(
        yaml.safe_dump(
            {
                'training_seeds': [2027, 2028, 2029, 2030, 2031],
                'environments': {'Ant-v4': {'total_steps': 5_000_000}},
                'checkpoint': {'latest_full_frequency_steps': 50_000},
                'training_invariants': {'replay_size': 400_000},
            },
        ),
        encoding='utf-8',
    )
    progress = checkpoint.parents[1] / 'progress.csv'
    with progress.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['TotalEnvSteps'])
        writer.writeheader()
        writer.writerow({'TotalEnvSteps': 50_000})
    payload = {
        'format': 'star-checkpoint',
        'version': 1,
        'model': {},
        'optimizers': {},
        'duals': {},
        'normalizer': {},
        'counters': {'env_steps': 50_000},
        'warmup': {},
        'rng': {},
        'replay': {'size': 50_000, 'capacity': 400_000},
        'extra': {
            'environment': 'Ant-v4',
            'seed': 2027,
            'resolved_config': {
                'env_id': 'Ant-v4',
                'seed': 2027,
                **resolved['custom_cfgs'],
            },
        },
    }
    return run_root, payload, checkpoint, matrix


def test_checkpoint_restart_validation_binds_format_replay_and_run(
    tmp_path: Path,
) -> None:
    run_root, payload, checkpoint, matrix = _checkpoint_fixture(tmp_path)
    torch.save(payload, checkpoint)
    checkpoints = {'ant-2027': {'path': str(checkpoint), 'bytes': checkpoint.stat().st_size}}
    validate_checkpoint_payloads(run_root, checkpoints, matrix)
    assert checkpoints['ant-2027']['validated_format'] == 'star-checkpoint-v1'
    assert checkpoints['ant-2027']['checkpoint_env_steps'] == 50_000

    payload['replay'] = None
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match='lacks replay recovery'):
        validate_checkpoint_payloads(run_root, checkpoints, matrix)

    payload['replay'] = {'size': 50_000, 'capacity': 400_000}
    payload['extra']['resolved_config']['algo_cfgs']['directional_loss_weight'] = 0.5
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match=r'directional_loss_weight.*changed'):
        validate_checkpoint_payloads(run_root, checkpoints, matrix)
