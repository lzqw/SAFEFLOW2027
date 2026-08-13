# Copyright 2026 SAFEFLOW2027 Authors. All Rights Reserved.
"""Tests for the saved Seeta root SSH public-key-only contract."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from scripts.star import seeta_root_ssh_hardening_agent as agent
from scripts.star import watch_seeta_root_ssh_hardening as watcher


FINGERPRINT = 'SHA256:' + 'A' * 43
DIGEST = 'b' * 64


def _encoded_payload(**updates):
    payload = {
        'version': agent.VERSION,
        'allow_repair': True,
        'force_listener_reload': False,
        'expected_authorized_key_fingerprint': FINGERPRINT,
        'expected_dropin_sha256': agent.sha256_bytes(agent.REQUIRED_DROPIN),
    }
    payload.update(updates)
    return (
        base64.urlsafe_b64encode(
            json.dumps(payload, sort_keys=True, separators=(',', ':')).encode(),
        )
        .decode()
        .rstrip('=')
    )


def _ed25519_line(seed: bytes = b'x' * 32) -> str:
    key_type = b'ssh-ed25519'
    blob = len(key_type).to_bytes(4, 'big') + key_type + len(seed).to_bytes(4, 'big') + seed
    return f'ssh-ed25519 {base64.b64encode(blob).decode()} test-key'


def _completed(
    stdout: str = '',
    *,
    returncode: int = 0,
    stderr: str = '',
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(['ssh'], returncode, stdout=stdout, stderr=stderr)


def test_remote_payload_is_exact_and_binds_the_compiled_policy() -> None:
    payload = agent.decode_payload(_encoded_payload())
    assert payload['allow_repair'] is True
    assert payload['force_listener_reload'] is False
    with pytest.raises(ValueError, match='fields changed'):
        agent.decode_payload(_encoded_payload(extra='unexpected'))
    with pytest.raises(ValueError, match='compiled root policy'):
        agent.decode_payload(_encoded_payload(expected_dropin_sha256='c' * 64))
    with pytest.raises(ValueError, match='repair policy'):
        agent.decode_payload(_encoded_payload(allow_repair=False, force_listener_reload=True))


def test_ed25519_wire_parser_rejects_malformed_or_non_ed25519_keys() -> None:
    valid = _ed25519_line()
    identity = watcher.parse_public_key_line(valid)
    assert identity['key_type'] == 'ssh-ed25519'
    assert identity['key_fingerprint'].startswith('SHA256:')
    blob = base64.b64decode(valid.split()[1])
    assert agent.ed25519_blob_valid(blob)
    assert watcher.ed25519_blob_valid(blob)
    assert not agent.ed25519_blob_valid(blob[:-1])
    with pytest.raises(ValueError, match='not ED25519'):
        watcher.parse_public_key_line(f'ssh-rsa {valid.split()[1]}')


def test_effective_root_policy_parser_is_exact() -> None:
    text = '\n'.join(f'{key} {value}' for key, value in agent.EXPECTED_ROOT_SETTINGS.items())
    assert agent.parse_sshd_settings(text) == agent.EXPECTED_ROOT_SETTINGS
    with pytest.raises(ValueError, match='incomplete'):
        agent.parse_sshd_settings('passwordauthentication no')
    with pytest.raises(ValueError, match='duplicate'):
        agent.parse_sshd_settings(f'{text}\npasswordauthentication no')


def test_listener_command_accepts_only_daemon_listener_forms() -> None:
    assert agent.listener_command_is_safe(b'/usr/sbin/sshd\0-D\0')
    assert agent.listener_command_is_safe(
        b'sshd: /usr/sbin/sshd -D [listener] 0 of 10-100 startups\0',
    )
    assert not agent.listener_command_is_safe(b'/usr/sbin/sshd\0-D\0-o\0Unsafe=yes\0')
    assert not agent.listener_command_is_safe(b'sshd: root@pts/1\0')


def test_compliant_agent_never_rewrites_or_reloads(monkeypatch: pytest.MonkeyPatch) -> None:
    key_identity = {'authorized_key_fingerprint': FINGERPRINT}
    dropin = {
        'dropin_present': True,
        'dropin_mode': '0600',
        'dropin_sha256': DIGEST,
    }
    monkeypatch.setattr(agent, 'authorized_key_identity', lambda: key_identity)
    monkeypatch.setattr(agent, 'dropin_identity', lambda: dropin)
    monkeypatch.setattr(agent, 'syntax_check', lambda: None)
    monkeypatch.setattr(agent, 'effective_root_settings', lambda: agent.EXPECTED_ROOT_SETTINGS)
    monkeypatch.setattr(agent, 'listener_pid', lambda: 818)
    monkeypatch.setattr(
        agent,
        'install_dropin',
        lambda: pytest.fail('compliant policy must not be rewritten'),
    )
    result = agent.run_agent(
        {
            'expected_authorized_key_fingerprint': FINGERPRINT,
            'expected_dropin_sha256': DIGEST,
            'allow_repair': True,
            'force_listener_reload': False,
        },
    )
    assert result['state'] == 'compliant'
    assert result['listener_pid'] == 818
    assert result['repair_performed'] is False
    assert result['listener_reload_performed'] is False


def test_agent_refuses_key_drift_before_any_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent,
        'authorized_key_identity',
        lambda: {'authorized_key_fingerprint': 'SHA256:' + 'C' * 43},
    )
    monkeypatch.setattr(
        agent,
        'install_dropin',
        lambda: pytest.fail('key drift must prevent repair'),
    )
    with pytest.raises(ValueError, match='fingerprint changed'):
        agent.run_agent(
            {
                'expected_authorized_key_fingerprint': FINGERPRINT,
                'expected_dropin_sha256': DIGEST,
                'allow_repair': True,
                'force_listener_reload': False,
            },
        )


def test_authorized_key_accepts_one_plain_or_restricted_ed25519_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ssh_directory = tmp_path / '.ssh'
    ssh_directory.mkdir(mode=0o700)
    authorized = ssh_directory / 'authorized_keys'
    authorized.write_text(_ed25519_line() + '\n', encoding='utf-8')
    authorized.chmod(0o600)
    monkeypatch.setattr(
        agent,
        'regular_root_file',
        lambda path, mode=None: path.lstat(),
    )
    monkeypatch.setattr(agent.os, 'getuid', lambda: 0)
    monkeypatch.setattr(agent.os, 'getgid', lambda: 0)
    original_lstat = Path.lstat

    def root_lstat(path):
        value = original_lstat(path)
        return os.stat_result(
            (
                value.st_mode,
                value.st_ino,
                value.st_dev,
                value.st_nlink,
                0,
                0,
                value.st_size,
                value.st_atime,
                value.st_mtime,
                value.st_ctime,
            ),
        )

    monkeypatch.setattr(Path, 'lstat', root_lstat)
    assert agent.authorized_key_identity(authorized)['authorized_key_count'] == 1
    authorized.write_text(f'command="false" {_ed25519_line()}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unsafe or ambiguous options'):
        agent.authorized_key_identity(authorized)


def test_agent_refuses_to_mask_policy_drift_outside_fixed_dropin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Lock:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def fileno(self):
            return 1

    monkeypatch.setattr(
        agent,
        'authorized_key_identity',
        lambda: {'authorized_key_fingerprint': FINGERPRINT},
    )
    monkeypatch.setattr(
        agent,
        'dropin_identity',
        lambda: {
            'dropin_present': True,
            'dropin_mode': '0600',
            'dropin_sha256': DIGEST,
        },
    )
    monkeypatch.setattr(agent, 'syntax_check', lambda: None)
    drift = {**agent.EXPECTED_ROOT_SETTINGS, 'passwordauthentication': 'yes'}
    monkeypatch.setattr(agent, 'effective_root_settings', lambda: drift)
    monkeypatch.setattr(agent, 'locked_transaction', lambda: Lock())
    monkeypatch.setattr(agent.fcntl, 'flock', lambda *_args: None)
    monkeypatch.setattr(
        agent,
        'install_dropin',
        lambda: pytest.fail('outside drift must not be masked by rewriting the same drop-in'),
    )
    with pytest.raises(ValueError, match='outside the fixed policy'):
        agent.run_agent(
            {
                'expected_authorized_key_fingerprint': FINGERPRINT,
                'expected_dropin_sha256': DIGEST,
                'allow_repair': True,
                'force_listener_reload': False,
            },
        )


def test_password_probe_accepts_only_publickey_only_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = {
        'hostname': 'server.example',
        'port': 11281,
        'user': 'root',
        'known_hosts_file': Path('/safe/known_hosts'),
    }
    monkeypatch.setattr(
        subprocess,
        'run',
        lambda *_args, **_kwargs: _completed(
            returncode=255,
            stderr='root@server.example: Permission denied (publickey).\r\n',
        ),
    )
    assert watcher.password_method_rejected(endpoint)['password_method_offered'] is False
    monkeypatch.setattr(
        subprocess,
        'run',
        lambda *_args, **_kwargs: _completed(
            returncode=255,
            stderr='root@server.example: Permission denied (publickey,password).\n',
        ),
    )
    with pytest.raises(watcher.PasswordMethodOfferedError, match='still offers a password'):
        watcher.password_method_rejected(endpoint)


def test_host_key_fingerprint_requires_exact_endpoint_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint = 'SHA256:' + 'H' * 43
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[1] == '-F':
            return _completed('hashed-host ssh-ed25519 AAAA\n')
        return _completed(f'256 {fingerprint} host (ED25519)\n')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    assert watcher.host_key_fingerprints(
        'server.example',
        11281,
        Path('/safe/known_hosts'),
    ) == [fingerprint]
    assert calls[0][2] == '[server.example]:11281'


def test_remote_invocation_uses_argument_vector_and_validates_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = watcher.build_payload(
        expected_fingerprint=FINGERPRINT,
        expected_dropin_sha256=DIGEST,
        allow_repair=True,
        force_listener_reload=False,
    )
    response = {
        'version': watcher.REMOTE_VERSION,
        'state': 'compliant',
        'root_settings': watcher.EXPECTED_ROOT_SETTINGS,
        'authorized_key_fingerprint': FINGERPRINT,
        'dropin_sha256': DIGEST,
    }
    seen = {}

    def fake_run(command, **kwargs):
        seen['command'] = command
        seen.update(kwargs)
        return _completed(json.dumps(response) + '\n')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    assert (
        watcher.invoke_remote(
            alias='seeta-nmb2',
            config_path=Path('/safe/config'),
            remote_python='/root/miniconda3/bin/python3',
            payload=payload,
            agent_source='print()\n',
        )
        == response
    )
    assert seen['command'][-4:-1] == ['seeta-nmb2', '/root/miniconda3/bin/python3', '-']
    assert seen['input'] == 'print()\n'
    assert 'shell' not in seen


def test_local_key_validation_does_not_persist_private_key_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private = tmp_path / 'key'
    public = tmp_path / 'key.pub'
    private.write_text('private material\n', encoding='utf-8')
    public.write_text(_ed25519_line() + '\n', encoding='utf-8')
    private.chmod(0o600)
    public.chmod(0o644)
    fingerprint = watcher.parse_public_key(public)['key_fingerprint']
    monkeypatch.setattr(
        subprocess,
        'run',
        lambda *_args, **_kwargs: _completed(_ed25519_line() + '\n'),
    )
    identity = watcher.validate_local_key(private.resolve(), fingerprint)
    assert identity['key_fingerprint'] == fingerprint
    assert all(key != 'private_key_sha256' for key in identity)
    assert hashlib.sha256(private.read_bytes()).hexdigest() not in json.dumps(identity)
