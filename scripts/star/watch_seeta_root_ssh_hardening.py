#!/usr/bin/env python3
"""Keep the saved Seeta SSH login public-key-only without retaining a password."""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_ALIAS = re.compile(r'^[A-Za-z0-9_.-]+$')
SAFE_HOSTNAME = re.compile(r'^[A-Za-z0-9.-]+$')
SAFE_REMOTE_PATH = re.compile(r'^/[A-Za-z0-9_./-]+$')
FULL_SHA256 = re.compile(r'^[0-9a-f]{64}$')
KEY_FINGERPRINT = re.compile(r'^SHA256:[A-Za-z0-9+/]{43}$')
REMOTE_AGENT = Path(__file__).resolve().with_name('seeta_root_ssh_hardening_agent.py')
WATCH_VERSION = 'safeflow-root-ssh-hardening-watch-v1'
REMOTE_VERSION = 'safeflow-root-ssh-hardening-agent-v1'
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
SCIENTIFIC_BOUNDARY = {
    'scientific_result': False,
    'training_state_changed': False,
    'training_configuration_changed': False,
    'scientific_protocol_changed': False,
    'used_for_checkpoint_or_configuration_selection': False,
    'scientific_metric_values_used_for_decision': False,
    'scientific_metric_values_reported': False,
}


class RemoteHardeningError(RuntimeError):
    """Represent one machine-readable fail-closed response from the remote agent."""

    def __init__(self, payload: dict[str, Any], stderr: str = '') -> None:
        self.payload = payload
        self.stderr = stderr
        message = f"remote hardening alert: {payload.get('error', 'unknown error')}"
        if stderr:
            message = f'{message}; stderr={stderr}'
        super().__init__(message)


class PasswordMethodOfferedError(RuntimeError):
    """Indicate that a fresh daemon connection advertised a password method."""


def utc_now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


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


def atomic_json(path: Path, payload: Any) -> None:
    """Atomically replace one owner-private JSON receipt."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8', closefd=False) as stream:
                json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write('\n')
                stream.flush()
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def append_jsonl(path: Path, payload: Any) -> None:
    """Append and fsync one compact owner-private monitoring row."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ValueError('hardening journal ownership or permissions changed')
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, 'a', encoding='utf-8', closefd=False) as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n')
            stream.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def parse_public_key_line(line: str) -> dict[str, str]:
    """Validate one ED25519 public-key line and return non-secret identities."""
    fields = line.strip().split()
    if len(fields) < 2 or fields[0] != 'ssh-ed25519':
        raise ValueError('local public key is not ED25519')
    try:
        blob = base64.b64decode(fields[1], validate=True)
    except binascii.Error as error:
        raise ValueError('local public-key payload is invalid') from error
    if not ed25519_blob_valid(blob):
        raise ValueError('local public-key payload is not an ED25519 wire record')
    fingerprint = base64.b64encode(hashlib.sha256(blob).digest()).decode('ascii').rstrip('=')
    return {
        'key_type': fields[0],
        'key_fingerprint': f'SHA256:{fingerprint}',
    }


def parse_public_key(path: Path) -> dict[str, str]:
    """Validate one local ED25519 public-key file."""
    rows = [row.strip() for row in path.read_text(encoding='utf-8').splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError('local public-key file must contain exactly one active key')
    return {
        **parse_public_key_line(rows[0]),
        'public_key_sha256': sha256_file(path),
    }


def validate_local_key(private_key: Path, expected_fingerprint: str) -> dict[str, str]:
    """Require a protected private key whose derived public identity is frozen."""
    if not private_key.is_absolute() or private_key.is_symlink():
        raise ValueError('local private-key path is not an absolute regular path')
    metadata = private_key.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ValueError('local private key ownership or permissions changed')
    public_key = Path(f'{private_key}.pub')
    if public_key.is_symlink():
        raise ValueError('local public-key path must not be symbolic')
    public_metadata = public_key.lstat()
    if (
        not stat.S_ISREG(public_metadata.st_mode)
        or public_metadata.st_uid != os.getuid()
        or stat.S_IMODE(public_metadata.st_mode) & 0o022
    ):
        raise ValueError('local public key ownership or permissions changed')
    identity = parse_public_key(public_key)
    if identity['key_fingerprint'] != expected_fingerprint:
        raise ValueError('local public-key fingerprint changed')
    try:
        derived = subprocess.run(
            ['ssh-keygen', '-y', '-f', str(private_key)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError('local private-key identity query timed out') from error
    if derived.returncode != 0:
        raise ValueError('local private key cannot supply unattended public-key authentication')
    derived_identity = parse_public_key_line(derived.stdout)
    if derived_identity['key_fingerprint'] != expected_fingerprint:
        raise ValueError('local private and public SSH keys do not match')
    return {
        **identity,
        'private_key_mode': '0600',
    }


def parse_ssh_g(text: str) -> dict[str, list[str]]:
    """Parse OpenSSH's effective client configuration without losing repeated values."""
    settings: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        key, separator, value = raw_line.partition(' ')
        if not separator:
            continue
        settings.setdefault(key.lower(), []).append(value.strip())
    return settings


def one_setting(settings: dict[str, list[str]], key: str) -> str:
    """Return one required effective SSH setting."""
    values = settings.get(key, [])
    if len(values) != 1 or not values[0]:
        raise ValueError(f'effective SSH setting is missing or ambiguous: {key}')
    return values[0]


def host_key_fingerprints(hostname: str, port: int, known_hosts: Path) -> list[str]:
    """Return sorted host-key fingerprints bound to the saved endpoint."""
    lookup = f'[{hostname}]:{port}' if port != 22 else hostname
    completed = subprocess.run(
        ['ssh-keygen', '-F', lookup, '-f', str(known_hosts)],
        capture_output=True,
        text=True,
        check=False,
    )
    records = [row for row in completed.stdout.splitlines() if row and not row.startswith('#')]
    fingerprints = []
    for record in records:
        identified = subprocess.run(
            ['ssh-keygen', '-lf', '-'],
            input=f'{record}\n',
            capture_output=True,
            text=True,
            check=False,
        )
        if identified.returncode != 0:
            raise ValueError('saved SSH host-key record is invalid')
        fields = identified.stdout.split()
        if len(fields) < 2 or not KEY_FINGERPRINT.fullmatch(fields[1]):
            raise ValueError('saved SSH host-key fingerprint is invalid')
        fingerprints.append(fields[1])
    if not fingerprints:
        raise ValueError('saved SSH endpoint has no pinned host key')
    return sorted(set(fingerprints))


def effective_client_config(
    alias: str,
    config_path: Path,
    expected_host_key_fingerprint: str,
) -> dict[str, Any]:
    """Resolve and strictly validate the saved noninteractive SSH alias."""
    if not SAFE_ALIAS.fullmatch(alias):
        raise ValueError('SSH alias contains unsafe characters')
    if not config_path.is_absolute() or config_path.is_symlink():
        raise ValueError('SSH config path is not one fixed regular file')
    metadata = config_path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ValueError('local SSH config ownership or permissions changed')
    ssh_directory = config_path.parent
    directory_metadata = ssh_directory.lstat()
    if (
        not stat.S_ISDIR(directory_metadata.st_mode)
        or ssh_directory.is_symlink()
        or directory_metadata.st_uid != os.getuid()
        or stat.S_IMODE(directory_metadata.st_mode) != 0o700
    ):
        raise ValueError('local SSH directory ownership or permissions changed')
    completed = subprocess.run(
        ['ssh', '-G', '-F', str(config_path), alias],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f'effective SSH client query failed: {completed.stderr.strip()}')
    settings = parse_ssh_g(completed.stdout)
    hostname = one_setting(settings, 'hostname')
    user = one_setting(settings, 'user')
    port_text = one_setting(settings, 'port')
    try:
        port = int(port_text)
    except ValueError as error:
        raise ValueError('effective SSH port is invalid') from error
    if not SAFE_HOSTNAME.fullmatch(hostname) or user != 'root' or not 1 <= port <= 65535:
        raise ValueError('saved SSH endpoint identity changed')
    exact = {
        'batchmode': {'yes'},
        'identitiesonly': {'yes'},
        'passwordauthentication': {'no'},
        'kbdinteractiveauthentication': {'no'},
        'pubkeyauthentication': {'true', 'yes'},
        'stricthostkeychecking': {'true', 'yes'},
    }
    for key, expected in exact.items():
        if one_setting(settings, key).lower() not in expected:
            raise ValueError(f'effective SSH client policy changed: {key}')
    identities = settings.get('identityfile', [])
    if len(identities) != 1:
        raise ValueError('saved SSH alias must have exactly one identity file')
    identity_path = Path(os.path.expanduser(identities[0]))
    if not identity_path.is_absolute() or '..' in identity_path.parts:
        raise ValueError('saved SSH identity path is unsafe')
    known_host_tokens = [
        token for value in settings.get('userknownhostsfile', []) for token in shlex.split(value)
    ]
    known_hosts = [Path(os.path.expanduser(value)) for value in known_host_tokens]
    if not known_hosts:
        raise ValueError('saved SSH alias has no known-hosts file')
    if not known_hosts[0].is_absolute() or '..' in known_hosts[0].parts:
        raise ValueError('local known-hosts path is unsafe')
    known_hosts_metadata = known_hosts[0].lstat()
    if (
        not stat.S_ISREG(known_hosts_metadata.st_mode)
        or known_hosts[0].is_symlink()
        or known_hosts_metadata.st_uid != os.getuid()
        or stat.S_IMODE(known_hosts_metadata.st_mode) != 0o600
    ):
        raise ValueError('local known-hosts file ownership or permissions changed')
    fingerprints = host_key_fingerprints(hostname, port, known_hosts[0])
    if fingerprints != [expected_host_key_fingerprint]:
        raise ValueError('saved SSH host-key identity changed')
    endpoint_sha256 = sha256_bytes(
        json.dumps(
            {'hostname': hostname, 'port': port, 'user': user},
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8'),
    )
    return {
        'alias': alias,
        'hostname': hostname,
        'port': port,
        'user': user,
        'identity_file': identity_path,
        'known_hosts_file': known_hosts[0],
        'host_key_fingerprint': fingerprints[0],
        'endpoint_sha256': endpoint_sha256,
        'config_sha256': sha256_file(config_path),
    }


def key_probe(alias: str, config_path: Path, private_key: Path) -> None:
    """Require a fresh command connection using only the frozen public key."""
    completed = subprocess.run(
        [
            'ssh',
            '-F',
            str(config_path),
            '-o',
            'BatchMode=yes',
            '-o',
            'ConnectTimeout=15',
            '-o',
            'IdentitiesOnly=yes',
            '-o',
            'PreferredAuthentications=publickey',
            '-o',
            'PasswordAuthentication=no',
            '-o',
            'KbdInteractiveAuthentication=no',
            '-i',
            str(private_key),
            alias,
            'true',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ConnectionError(f'fresh public-key SSH probe failed with exit {completed.returncode}')


def password_method_rejected(endpoint: dict[str, Any]) -> dict[str, Any]:
    """Require a new connection to reject password-only authentication pre-prompt."""
    completed = subprocess.run(
        [
            'ssh',
            '-F',
            '/dev/null',
            '-o',
            'BatchMode=yes',
            '-o',
            'ConnectTimeout=15',
            '-o',
            'PreferredAuthentications=password',
            '-o',
            'PubkeyAuthentication=no',
            '-o',
            'PasswordAuthentication=yes',
            '-o',
            'KbdInteractiveAuthentication=no',
            '-o',
            'NumberOfPasswordPrompts=0',
            '-o',
            'StrictHostKeyChecking=yes',
            '-o',
            f'UserKnownHostsFile={endpoint["known_hosts_file"]}',
            '-p',
            str(endpoint['port']),
            f'{endpoint["user"]}@{endpoint["hostname"]}',
            'true',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    stderr = completed.stderr.replace('\r', '').strip()
    final_line = stderr.splitlines()[-1] if stderr else ''
    denied = re.search(r'Permission denied \(([^)]+)\)\.$', final_line)
    if completed.returncode != 255 or denied is None:
        raise ConnectionError(
            f'password-method SSH probe failed before authentication with exit '
            f'{completed.returncode}',
        )
    offered = {value.strip() for value in denied.group(1).split(',') if value.strip()}
    if offered != {'publickey'}:
        if offered & {'password', 'keyboard-interactive'}:
            raise PasswordMethodOfferedError('remote SSH daemon still offers a password method')
        raise ValueError('remote SSH daemon advertised an unexpected authentication policy')
    return {
        'password_probe_exit_code': 255,
        'password_method_offered': False,
        'server_offered_authentication': 'publickey',
    }


def build_payload(
    *,
    expected_fingerprint: str,
    expected_dropin_sha256: str,
    allow_repair: bool,
    force_listener_reload: bool,
) -> dict[str, Any]:
    """Build the complete remote root-policy contract."""
    if not KEY_FINGERPRINT.fullmatch(expected_fingerprint):
        raise ValueError('expected SSH key fingerprint is invalid')
    if not FULL_SHA256.fullmatch(expected_dropin_sha256):
        raise ValueError('expected SSH drop-in identity is invalid')
    if force_listener_reload and not allow_repair:
        raise ValueError('listener reload is not allowed in observe-only mode')
    return {
        'version': REMOTE_VERSION,
        'allow_repair': allow_repair,
        'force_listener_reload': force_listener_reload,
        'expected_authorized_key_fingerprint': expected_fingerprint,
        'expected_dropin_sha256': expected_dropin_sha256,
    }


def invoke_remote(
    *,
    alias: str,
    config_path: Path,
    remote_python: str,
    payload: dict[str, Any],
    agent_source: str,
) -> dict[str, Any]:
    """Execute one hardening transaction over key-only SSH and parse strict JSON."""
    remote_path = Path(remote_python)
    if (
        not SAFE_ALIAS.fullmatch(alias)
        or not SAFE_REMOTE_PATH.fullmatch(remote_python)
        or not remote_path.is_absolute()
        or '..' in remote_path.parts
    ):
        raise ValueError('remote Python path is unsafe')
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8'),
    ).decode('ascii')
    completed = subprocess.run(
        [
            'ssh',
            '-F',
            str(config_path),
            '-o',
            'BatchMode=yes',
            '-o',
            'ConnectTimeout=15',
            alias,
            remote_python,
            '-',
            encoded,
        ],
        input=agent_source,
        capture_output=True,
        text=True,
        check=False,
    )
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 and not rows:
        raise ConnectionError(
            f'remote SSH hardening connection failed with exit {completed.returncode}',
        )
    if len(rows) != 1:
        raise RuntimeError(f'remote SSH hardening returned {len(rows)} stdout records')
    try:
        result = json.loads(rows[0])
    except json.JSONDecodeError as error:
        raise RuntimeError('remote SSH hardening returned invalid JSON') from error
    if not isinstance(result, dict) or result.get('version') != payload['version']:
        raise RuntimeError('remote SSH hardening response identity changed')
    if completed.returncode != 0 or result.get('state') == 'alert':
        raise RemoteHardeningError(result, completed.stderr.strip())
    allowed_states = {'compliant', 'reloaded', 'repaired', 'repair_required'}
    if result.get('state') not in allowed_states:
        raise RuntimeError(f'remote SSH hardening returned invalid state: {result.get("state")!r}')
    if result.get('root_settings') != EXPECTED_ROOT_SETTINGS:
        raise RuntimeError('remote SSH hardening response has a noncompliant root policy')
    if result.get('authorized_key_fingerprint') != payload['expected_authorized_key_fingerprint']:
        raise RuntimeError('remote SSH authorized-key identity changed')
    if result.get('dropin_sha256') != payload['expected_dropin_sha256']:
        raise RuntimeError('remote SSH policy-file identity changed')
    return result


def load_last_listener(path: Path) -> int | None:
    """Read the previous validated listener PID, if any."""
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get('listener_pid') if isinstance(payload, dict) else None
    return value if isinstance(value, int) and value > 1 else None


def listener_reload_needed(previous_listener: int | None, result: dict[str, Any]) -> bool:
    """Return whether a newly replaced listener still needs the fixed policy reloaded."""
    listener = result.get('listener_pid')
    return (
        previous_listener is not None
        and isinstance(listener, int)
        and listener != previous_listener
        and not result.get('listener_reload_performed', False)
    )


def compact_record(result: dict[str, Any], identities: dict[str, Any]) -> dict[str, Any]:
    """Build a bounded append-only monitoring record without endpoint data."""
    return {
        'observed_at_utc': result['observed_at_utc'],
        'state': result['state'],
        'listener_pid': result.get('listener_pid'),
        'repair_performed': result.get('repair_performed', False),
        'listener_reload_performed': result.get('listener_reload_performed', False),
        'public_key_probe_passed': result.get('public_key_probe_passed'),
        'password_method_offered': result.get('password_method_offered'),
        **SCIENTIFIC_BOUNDARY,
        **identities,
    }


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Continuously verify and, within the fixed contract, restore public-key-only SSH."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-host', required=True)
    parser.add_argument('--ssh-config', type=Path, required=True)
    parser.add_argument('--local-private-key', type=Path, required=True)
    parser.add_argument('--expected-key-fingerprint', required=True)
    parser.add_argument('--expected-host-key-fingerprint', required=True)
    parser.add_argument('--expected-endpoint-sha256', required=True)
    parser.add_argument('--expected-ssh-config-sha256', required=True)
    parser.add_argument('--remote-python-bin', required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=300.0)
    parser.add_argument('--observe-only', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.poll_seconds <= 0.0:
        parser.error('--poll-seconds must be positive')
    if not KEY_FINGERPRINT.fullmatch(args.expected_key_fingerprint):
        parser.error('--expected-key-fingerprint is invalid')
    if not KEY_FINGERPRINT.fullmatch(args.expected_host_key_fingerprint):
        parser.error('--expected-host-key-fingerprint is invalid')
    if not FULL_SHA256.fullmatch(args.expected_endpoint_sha256):
        parser.error('--expected-endpoint-sha256 is invalid')
    if not FULL_SHA256.fullmatch(args.expected_ssh_config_sha256):
        parser.error('--expected-ssh-config-sha256 is invalid')
    if not REMOTE_AGENT.is_file():
        parser.error(f'remote hardening agent is unavailable: {REMOTE_AGENT}')

    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state_dir, 0o700)
    watcher_path = Path(__file__).resolve()
    immutable_hashes = {
        'host_watcher_sha256': sha256_file(watcher_path),
        'remote_agent_sha256': sha256_file(REMOTE_AGENT),
    }
    agent_source = REMOTE_AGENT.read_text(encoding='utf-8')
    expected_dropin = sha256_bytes(REQUIRED_DROPIN)
    ssh_config = args.ssh_config
    private_key = args.local_private_key
    endpoint = effective_client_config(
        args.remote_host,
        ssh_config,
        args.expected_host_key_fingerprint,
    )
    if endpoint['endpoint_sha256'] != args.expected_endpoint_sha256:
        parser.error('saved SSH endpoint does not match the frozen identity')
    if endpoint['config_sha256'] != args.expected_ssh_config_sha256:
        parser.error('saved SSH config does not match the frozen identity')
    key_identity = validate_local_key(private_key, args.expected_key_fingerprint)
    if endpoint['identity_file'] != private_key.resolve():
        parser.error('saved SSH alias does not resolve to the frozen private key')
    identities = {
        'watch_version': WATCH_VERSION,
        'remote_alias': args.remote_host,
        'observe_only': args.observe_only,
        'expected_key_fingerprint': args.expected_key_fingerprint,
        'expected_host_key_fingerprint': args.expected_host_key_fingerprint,
        'expected_endpoint_sha256': args.expected_endpoint_sha256,
        'expected_ssh_config_sha256': args.expected_ssh_config_sha256,
        'expected_dropin_sha256': expected_dropin,
        'ssh_config_sha256': endpoint['config_sha256'],
        **key_identity,
        **immutable_hashes,
    }
    latest_path = state_dir / 'root_ssh_hardening_latest.json'
    previous_listener = load_last_listener(latest_path)
    lock_path = state_dir / '.root-ssh-hardening-watch.lock'
    lock_flags = os.O_WRONLY | os.O_CREAT
    if hasattr(os, 'O_NOFOLLOW'):
        lock_flags |= os.O_NOFOLLOW
    lock_descriptor = os.open(lock_path, lock_flags, 0o600)
    lock_metadata = os.fstat(lock_descriptor)
    if (
        not stat.S_ISREG(lock_metadata.st_mode)
        or lock_metadata.st_uid != os.getuid()
        or stat.S_IMODE(lock_metadata.st_mode) != 0o600
    ):
        os.close(lock_descriptor)
        parser.error('hardening watcher lock ownership or permissions changed')
    with os.fdopen(lock_descriptor, 'w', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit('another root SSH hardening watcher holds the state lock') from error
        while True:
            observed_at = utc_now()
            try:
                if sha256_file(watcher_path) != immutable_hashes['host_watcher_sha256']:
                    raise ValueError('host SSH hardening watcher changed while running')
                if sha256_file(REMOTE_AGENT) != immutable_hashes['remote_agent_sha256']:
                    raise ValueError('remote SSH hardening agent changed while running')
                current_endpoint = effective_client_config(
                    args.remote_host,
                    ssh_config,
                    args.expected_host_key_fingerprint,
                )
                if current_endpoint != endpoint:
                    raise ValueError('effective saved SSH alias changed while running')
                current_key = validate_local_key(
                    private_key,
                    args.expected_key_fingerprint,
                )
                if current_key != key_identity:
                    raise ValueError('local SSH key identity changed while running')
                payload = build_payload(
                    expected_fingerprint=args.expected_key_fingerprint,
                    expected_dropin_sha256=expected_dropin,
                    allow_repair=not args.observe_only,
                    force_listener_reload=False,
                )
                result = invoke_remote(
                    alias=args.remote_host,
                    config_path=ssh_config,
                    remote_python=args.remote_python_bin,
                    payload=payload,
                    agent_source=agent_source,
                )
                if result['state'] == 'repair_required':
                    raise ValueError('remote SSH root policy requires repair in observe-only mode')
                listener = result.get('listener_pid')
                if not isinstance(listener, int) or listener <= 1:
                    raise RuntimeError('remote SSH listener identity is unavailable')
                if listener_reload_needed(previous_listener, result):
                    reload_payload = build_payload(
                        expected_fingerprint=args.expected_key_fingerprint,
                        expected_dropin_sha256=expected_dropin,
                        allow_repair=not args.observe_only,
                        force_listener_reload=True,
                    )
                    result = invoke_remote(
                        alias=args.remote_host,
                        config_path=ssh_config,
                        remote_python=args.remote_python_bin,
                        payload=reload_payload,
                        agent_source=agent_source,
                    )
                key_probe(args.remote_host, ssh_config, private_key)
                try:
                    password_probe = password_method_rejected(endpoint)
                except PasswordMethodOfferedError:
                    if args.observe_only:
                        raise
                    reload_payload = build_payload(
                        expected_fingerprint=args.expected_key_fingerprint,
                        expected_dropin_sha256=expected_dropin,
                        allow_repair=True,
                        force_listener_reload=True,
                    )
                    result = invoke_remote(
                        alias=args.remote_host,
                        config_path=ssh_config,
                        remote_python=args.remote_python_bin,
                        payload=reload_payload,
                        agent_source=agent_source,
                    )
                    key_probe(
                        args.remote_host,
                        ssh_config,
                        private_key,
                    )
                    password_probe = password_method_rejected(endpoint)
                result.update(
                    {
                        'public_key_probe_passed': True,
                        **password_probe,
                        **SCIENTIFIC_BOUNDARY,
                        **identities,
                    },
                )
                previous_listener = result['listener_pid']
            except ConnectionError as error:
                result = {
                    'observed_at_utc': observed_at,
                    'state': 'unreachable',
                    'error_type': type(error).__name__,
                    'error': str(error),
                    **SCIENTIFIC_BOUNDARY,
                    **identities,
                }
            except Exception as error:  # noqa: BLE001 - persist fail-closed security alerts
                remote_alert = error.payload if isinstance(error, RemoteHardeningError) else {}
                result = {
                    'observed_at_utc': observed_at,
                    'state': 'alert',
                    'error_type': type(error).__name__,
                    'error': str(error),
                    'remote_alert': remote_alert,
                    **SCIENTIFIC_BOUNDARY,
                    **identities,
                }
                atomic_json(state_dir / 'root_ssh_hardening_alert.json', result)
            atomic_json(latest_path, result)
            append_jsonl(
                state_dir / 'root_ssh_hardening_journal.jsonl',
                compact_record(result, identities),
            )
            if result.get('repair_performed') or result.get('listener_reload_performed'):
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                atomic_json(state_dir / 'actions' / f'action-{stamp}.json', result)
            print(
                json.dumps(
                    {
                        'observed_at_utc': result['observed_at_utc'],
                        'state': result['state'],
                        'repair_performed': result.get('repair_performed', False),
                        'public_key_probe_passed': result.get('public_key_probe_passed', False),
                        'password_method_offered': result.get('password_method_offered'),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.once:
                if result['state'] in {'alert', 'unreachable'}:
                    raise SystemExit(2)
                return
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
