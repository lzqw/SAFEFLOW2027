#!/usr/bin/env python3
"""Validate and, when authorized, restore the fixed root public-key SSH policy."""

from __future__ import annotations

import base64
import binascii
import contextlib
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


VERSION = 'safeflow-root-ssh-hardening-agent-v1'
DROPIN_PATH = Path('/etc/ssh/sshd_config.d/00-safeflow-root-key-only.conf')
AUTHORIZED_KEYS_PATH = Path('/root/.ssh/authorized_keys')
SSHD_BIN = Path('/usr/sbin/sshd')
LOCK_PATH = Path('/run/safeflow-root-ssh-hardening.lock')
REQUIRED_DROPIN = (
    b'# SAFEFLOW2027: the shared root password must not remain a remote login method.\n'
    b'# The validated ED25519 key remains available through /root/.ssh/authorized_keys.\n'
    b'PermitRootLogin prohibit-password\n'
    b'PubkeyAuthentication yes\n'
    b'\n'
    b'Match User root\n'
    b'    AuthenticationMethods publickey\n'
    b'    PasswordAuthentication no\n'
    b'    KbdInteractiveAuthentication no\n'
    b'Match all\n'
)
EXPECTED_ROOT_SETTINGS = {
    'permitrootlogin': 'without-password',
    'pubkeyauthentication': 'yes',
    'passwordauthentication': 'no',
    'kbdinteractiveauthentication': 'no',
    'authenticationmethods': 'publickey',
}
FULL_SHA256 = re.compile(r'^[0-9a-f]{64}$')
KEY_FINGERPRINT = re.compile(r'^SHA256:[A-Za-z0-9+/]{43}$')


def sha256_bytes(value: bytes) -> str:
    """Return one lowercase SHA-256 identity."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Return one streaming SHA-256 identity."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def decode_payload(value: str) -> dict[str, Any]:
    """Decode and constrain the host-provided hardening contract."""
    try:
        padding = '=' * (-len(value) % 4)
        payload = json.loads(
            base64.b64decode(value + padding, altchars=b'-_', validate=True),
        )
    except (ValueError, json.JSONDecodeError) as error:
        raise ValueError('hardening payload is not valid URL-safe JSON') from error
    if not isinstance(payload, dict) or set(payload) != {
        'version',
        'allow_repair',
        'force_listener_reload',
        'expected_authorized_key_fingerprint',
        'expected_dropin_sha256',
    }:
        raise ValueError('hardening payload fields changed')
    if (
        payload.get('version') != VERSION
        or not isinstance(payload.get('allow_repair'), bool)
        or not isinstance(payload.get('force_listener_reload'), bool)
        or (payload['force_listener_reload'] and not payload['allow_repair'])
    ):
        raise ValueError('hardening payload version or repair policy changed')
    fingerprint = str(payload.get('expected_authorized_key_fingerprint', ''))
    dropin_sha = str(payload.get('expected_dropin_sha256', ''))
    if not KEY_FINGERPRINT.fullmatch(fingerprint) or not FULL_SHA256.fullmatch(dropin_sha):
        raise ValueError('hardening payload identity is invalid')
    if dropin_sha != sha256_bytes(REQUIRED_DROPIN):
        raise ValueError('hardening payload does not match the compiled root policy')
    return payload


def regular_root_file(path: Path, *, mode: int | None = None) -> os.stat_result:
    """Require one root-owned, non-symbolic regular file with an optional exact mode."""
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError(f'protected SSH path is not a regular file: {path}')
    if metadata.st_uid != 0 or metadata.st_gid != 0:
        raise ValueError(f'protected SSH path is not root-owned: {path}')
    if mode is not None and stat.S_IMODE(metadata.st_mode) != mode:
        raise ValueError(f'protected SSH path has unsafe permissions: {path}')
    return metadata


def authorized_key_identity(path: Path = AUTHORIZED_KEYS_PATH) -> dict[str, Any]:
    """Require exactly one protected ED25519 key and return non-secret identities."""
    ssh_directory = path.parent
    directory = ssh_directory.lstat()
    if (
        not stat.S_ISDIR(directory.st_mode)
        or ssh_directory.is_symlink()
        or directory.st_uid != 0
        or directory.st_gid != 0
        or stat.S_IMODE(directory.st_mode) != 0o700
    ):
        raise ValueError('root SSH directory ownership or permissions changed')
    regular_root_file(path, mode=0o600)
    keys = []
    for raw_line in path.read_text(encoding='utf-8', errors='strict').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        key_indexes = [index for index, field in enumerate(fields) if field == 'ssh-ed25519']
        if len(key_indexes) != 1 or key_indexes[0] + 1 >= len(fields):
            raise ValueError('root authorized_keys contains an unsupported record')
        key_index = key_indexes[0]
        if key_index > 0:
            options = fields[:key_index]
            if len(options) != 1 or options[0].startswith(('command=', 'environment=')):
                raise ValueError('root authorized_keys contains unsafe or ambiguous options')
        try:
            blob = base64.b64decode(fields[key_index + 1], validate=True)
        except binascii.Error as error:
            raise ValueError('root authorized key payload is invalid') from error
        if not ed25519_blob_valid(blob):
            raise ValueError('root authorized key is not an ED25519 wire record')
        fingerprint = base64.b64encode(hashlib.sha256(blob).digest()).decode('ascii').rstrip('=')
        keys.append(f'SHA256:{fingerprint}')
    if len(keys) != 1:
        raise ValueError('root authorized_keys must contain exactly one active key')
    return {
        'authorized_key_count': 1,
        'authorized_key_type': 'ssh-ed25519',
        'authorized_key_fingerprint': keys[0],
        'authorized_keys_sha256': sha256_file(path),
        'authorized_keys_mode': '0600',
        'ssh_directory_mode': '0700',
    }


def ed25519_blob_valid(blob: bytes) -> bool:
    """Return whether an SSH wire blob is exactly one 32-byte ED25519 public key."""
    if len(blob) < 4:
        return False
    type_length = int.from_bytes(blob[:4], byteorder='big')
    offset = 4 + type_length
    if blob[4:offset] != b'ssh-ed25519' or len(blob) < offset + 4:
        return False
    key_length = int.from_bytes(blob[offset : offset + 4], byteorder='big')
    return key_length == 32 and len(blob) == offset + 4 + key_length


def parse_sshd_settings(text: str) -> dict[str, str]:
    """Parse the exact security-relevant subset of ``sshd -T`` output."""
    observed: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(' ')
        if not separator or key not in EXPECTED_ROOT_SETTINGS:
            continue
        if key in observed:
            raise ValueError(f'duplicate effective sshd setting: {key}')
        observed[key] = value.strip()
    if set(observed) != set(EXPECTED_ROOT_SETTINGS):
        raise ValueError('effective root SSH settings are incomplete')
    return observed


def run_sshd(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the fixed system SSH daemon binary without a shell."""
    if SSHD_BIN.is_symlink() or not SSHD_BIN.is_file() or not os.access(SSHD_BIN, os.X_OK):
        raise ValueError('the fixed sshd binary is unavailable')
    return subprocess.run(
        [str(SSHD_BIN), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def syntax_check() -> None:
    """Require the complete SSH daemon configuration to parse."""
    completed = run_sshd('-t')
    if completed.returncode != 0:
        raise ValueError(f'sshd syntax validation failed: {completed.stderr.strip()}')


def effective_root_settings() -> dict[str, str]:
    """Return the effective policy for root from the complete daemon configuration."""
    completed = run_sshd('-T', '-C', 'user=root,host=localhost,addr=127.0.0.1')
    if completed.returncode != 0:
        raise ValueError(f'sshd effective-policy query failed: {completed.stderr.strip()}')
    return parse_sshd_settings(completed.stdout)


def dropin_identity(path: Path = DROPIN_PATH) -> dict[str, Any]:
    """Return strict identity fields for the installed root-policy drop-in."""
    if not path.exists():
        return {
            'dropin_present': False,
            'dropin_mode': None,
            'dropin_sha256': None,
        }
    metadata = regular_root_file(path)
    return {
        'dropin_present': True,
        'dropin_mode': f'{stat.S_IMODE(metadata.st_mode):04o}',
        'dropin_sha256': sha256_file(path),
    }


def fsync_directory(path: Path) -> None:
    """Persist directory entry changes."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_root_file(path: Path, content: bytes, mode: int = 0o600) -> None:
    """Atomically install one fixed root-owned file without following links."""
    parent = path.parent
    parent_metadata = parent.lstat()
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or parent.is_symlink()
        or parent_metadata.st_uid != 0
        or parent_metadata.st_gid != 0
        or stat.S_IMODE(parent_metadata.st_mode) & 0o022
    ):
        raise ValueError(f'protected SSH directory is unsafe: {parent}')
    temporary = parent / f'.{path.name}.tmp-{os.getpid()}'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, mode)
    try:
        try:
            os.fchown(descriptor, 0, 0)
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, 'wb', closefd=False) as stream:
                stream.write(content)
                stream.flush()
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        fsync_directory(parent)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def install_dropin(path: Path = DROPIN_PATH) -> None:
    """Install the compiled policy with syntax-checked rollback."""
    prior: tuple[bytes, int] | None = None
    if path.exists():
        metadata = regular_root_file(path)
        prior = (path.read_bytes(), stat.S_IMODE(metadata.st_mode))
    atomic_write_root_file(path, REQUIRED_DROPIN)
    try:
        syntax_check()
    except Exception:
        if prior is None:
            path.unlink()
            fsync_directory(path.parent)
        else:
            atomic_write_root_file(path, prior[0], prior[1])
        syntax_check()
        raise


def listener_command_is_safe(raw_command: bytes) -> bool:
    """Accept the daemon's initial argv or OpenSSH's listener process title."""
    fields = [field for field in raw_command.split(b'\0') if field]
    if fields == [str(SSHD_BIN).encode('ascii'), b'-D']:
        return True
    title = raw_command.rstrip(b'\0').decode('utf-8', errors='replace')
    return title.startswith(f'sshd: {SSHD_BIN} -D [listener] ')


def listener_pid(proc_root: Path | None = None) -> int:
    """Resolve the one fixed sshd listener from its configured PID file."""
    if proc_root is None:
        proc_root = Path('/proc')
    completed = run_sshd('-T')
    if completed.returncode != 0:
        raise ValueError(f'sshd PID-file query failed: {completed.stderr.strip()}')
    pid_files = [
        line.partition(' ')[2].strip()
        for line in completed.stdout.splitlines()
        if line.startswith('pidfile ')
    ]
    if len(pid_files) != 1:
        raise ValueError('sshd effective configuration does not expose one PID file')
    pid_path = Path(pid_files[0])
    if not pid_path.is_absolute() or '..' in pid_path.parts or pid_path.is_symlink():
        raise ValueError('sshd PID file path is unsafe')
    try:
        pid_metadata = pid_path.lstat()
        if (
            not stat.S_ISREG(pid_metadata.st_mode)
            or pid_metadata.st_uid != 0
            or pid_metadata.st_gid != 0
            or stat.S_IMODE(pid_metadata.st_mode) & 0o022
        ):
            raise ValueError('sshd PID file ownership or permissions are unsafe')
        pid = int(pid_path.read_text(encoding='ascii').strip())
    except (OSError, ValueError) as error:
        raise ValueError('sshd PID file is unavailable or invalid') from error
    if pid <= 1:
        raise ValueError('sshd listener PID is invalid')
    process = proc_root / str(pid)
    try:
        executable = (process / 'exe').resolve(strict=True)
        command = (process / 'cmdline').read_bytes()
    except OSError as error:
        raise ValueError('sshd listener process identity is unavailable') from error
    if executable != SSHD_BIN.resolve() or not listener_command_is_safe(command):
        raise ValueError('sshd PID file does not identify the listener process')
    return pid


def locked_transaction(path: Path = LOCK_PATH) -> TextIO:
    """Open one root-owned no-follow lock descriptor for the repair transaction."""
    parent = path.parent
    parent_metadata = parent.lstat()
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or parent.is_symlink()
        or parent_metadata.st_uid != 0
        or parent_metadata.st_gid != 0
        or stat.S_IMODE(parent_metadata.st_mode) & 0o022
    ):
        raise ValueError('hardening lock directory is unsafe')
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise ValueError('hardening lock file identity changed')
    os.fchmod(descriptor, 0o600)
    return os.fdopen(descriptor, 'r+', encoding='utf-8')


def signal_reload(pid: int) -> None:
    """Ask the validated listener to reread its already syntax-checked configuration."""
    os.kill(pid, signal.SIGHUP)


def policy_compliant(
    identity: dict[str, Any],
    settings: dict[str, str],
    *,
    expected_dropin_sha256: str,
) -> bool:
    """Return whether file identity and effective root policy both match the contract."""
    return (
        identity.get('dropin_present') is True
        and identity.get('dropin_mode') == '0600'
        and identity.get('dropin_sha256') == expected_dropin_sha256
        and settings == EXPECTED_ROOT_SETTINGS
    )


def run_agent(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the root key and atomically restore only the fixed SSH policy when needed."""
    key_identity = authorized_key_identity()
    expected_fingerprint = payload['expected_authorized_key_fingerprint']
    if key_identity['authorized_key_fingerprint'] != expected_fingerprint:
        raise ValueError('root authorized key fingerprint changed; refusing automatic repair')
    identity = dropin_identity()
    settings: dict[str, str] | None = None
    try:
        syntax_check()
        settings = effective_root_settings()
    except ValueError:
        pass
    compliant = settings is not None and policy_compliant(
        identity,
        settings,
        expected_dropin_sha256=payload['expected_dropin_sha256'],
    )
    if compliant:
        if payload['force_listener_reload']:
            with locked_transaction() as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                locked_key_identity = authorized_key_identity()
                if locked_key_identity['authorized_key_fingerprint'] != expected_fingerprint:
                    raise ValueError('root authorized key changed before listener reload')
                syntax_check()
                locked_identity = dropin_identity()
                locked_settings = effective_root_settings()
                if not policy_compliant(
                    locked_identity,
                    locked_settings,
                    expected_dropin_sha256=payload['expected_dropin_sha256'],
                ):
                    raise ValueError('root SSH policy changed before listener reload')
                pid = listener_pid()
                signal_reload(pid)
            return {
                'version': VERSION,
                'observed_at_utc': utc_now(),
                'state': 'reloaded',
                'repair_performed': False,
                'listener_reload_performed': True,
                'listener_pid': pid,
                'configuration_valid': True,
                'root_settings': locked_settings,
                **locked_identity,
                **locked_key_identity,
            }
        pid = listener_pid()
        return {
            'version': VERSION,
            'observed_at_utc': utc_now(),
            'state': 'compliant',
            'repair_performed': False,
            'listener_reload_performed': False,
            'listener_pid': pid,
            'configuration_valid': True,
            'root_settings': settings,
            **identity,
            **key_identity,
        }
    if not payload['allow_repair']:
        return {
            'version': VERSION,
            'observed_at_utc': utc_now(),
            'state': 'repair_required',
            'repair_performed': False,
            'listener_reload_performed': False,
            'configuration_valid': settings is not None,
            'root_settings': settings,
            **identity,
            **key_identity,
        }
    with locked_transaction() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        locked_key_identity = authorized_key_identity()
        if locked_key_identity['authorized_key_fingerprint'] != expected_fingerprint:
            raise ValueError('root authorized key changed before atomic repair')
        locked_identity = dropin_identity()
        locked_settings: dict[str, str] | None = None
        try:
            syntax_check()
            locked_settings = effective_root_settings()
        except ValueError:
            pass
        if locked_settings is not None and policy_compliant(
            locked_identity,
            locked_settings,
            expected_dropin_sha256=payload['expected_dropin_sha256'],
        ):
            pid = listener_pid()
            return {
                'version': VERSION,
                'observed_at_utc': utc_now(),
                'state': 'compliant',
                'repair_performed': False,
                'listener_reload_performed': False,
                'listener_pid': pid,
                'configuration_valid': True,
                'root_settings': locked_settings,
                **locked_identity,
                **locked_key_identity,
            }
        if (
            locked_identity.get('dropin_present') is True
            and locked_identity.get('dropin_mode') == '0600'
            and locked_identity.get('dropin_sha256') == payload['expected_dropin_sha256']
        ):
            raise ValueError('effective root SSH drift is outside the fixed policy drop-in')
        install_dropin()
        repaired_identity = dropin_identity()
        repaired_settings = effective_root_settings()
        if not policy_compliant(
            repaired_identity,
            repaired_settings,
            expected_dropin_sha256=payload['expected_dropin_sha256'],
        ):
            raise ValueError('root SSH policy remains noncompliant after atomic repair')
        pid = listener_pid()
        signal_reload(pid)
    return {
        'version': VERSION,
        'observed_at_utc': utc_now(),
        'state': 'repaired',
        'repair_performed': True,
        'listener_reload_performed': True,
        'listener_pid': pid,
        'configuration_valid': True,
        'root_settings': repaired_settings,
        **repaired_identity,
        **locked_key_identity,
    }


def main() -> None:
    """Execute one strict host-authorized SSH hardening transaction."""
    exit_code = 0
    try:
        if len(sys.argv) != 2:
            raise ValueError('hardening agent requires exactly one encoded contract')
        payload = decode_payload(sys.argv[1])
        result = run_agent(payload)
    except Exception as error:  # noqa: BLE001 - emit a machine-readable fail-closed response
        result = {
            'version': VERSION,
            'observed_at_utc': utc_now(),
            'state': 'alert',
            'error_type': type(error).__name__,
            'error': str(error),
            'repair_performed': False,
            'listener_reload_performed': False,
        }
        exit_code = 2
    result.update(
        {
            'scientific_result': False,
            'training_state_changed': False,
            'training_configuration_changed': False,
            'scientific_protocol_changed': False,
            'used_for_checkpoint_or_configuration_selection': False,
            'scientific_metric_values_used_for_decision': False,
            'scientific_metric_values_reported': False,
        },
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
