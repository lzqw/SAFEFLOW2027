#!/usr/bin/env python3
"""Replicate changed checkpoint snapshots, keeping large archives on the 4T disk."""
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

STATE = Path(__file__).resolve().parent / 'backup_state.json'
HOSTS = {
    'tjucta-a100': '/home/lzqw/SAFEFLOW2027_A100',
    '4080': '/home/carla/SAFEFLOW2027_4080',
    '5080': '/home/autodrive/LZQW/SAFEFLOW2027',
}
TRAINING_DISK_RESERVE_BYTES = 2 * 1024**3
SSH_OPTIONS = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
               '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=15',
               '-o', 'ServerAliveCountMax=3']


def log(message):
    print(datetime.datetime.now().astimezone().isoformat(), message, flush=True)


def ssh(host, command, timeout=60):
    return subprocess.run([*SSH_OPTIONS, host, command], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=timeout)


def state_save(state):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    os.replace(temporary, STATE)


def replicate(source, destination, changed):
    source_export = HOSTS[source] + '/recovery_backups/export'
    source_label = 'a100' if source == 'tjucta-a100' else source
    target_root = HOSTS[destination] + '/recovery_backups/from_' + source_label
    ssh(destination, 'mkdir -p ' + shlex.quote(target_root))
    logical_target = target_root
    target_root = ssh(destination, 'readlink -f -- ' + shlex.quote(logical_target)).stdout.strip()
    allowed = {logical_target}
    if destination == 'tjucta-a100' and source == '4080':
        allowed.add('/storage_0/SAFEFLOW2027_lzqw_uid1007/recovery_backups/from_4080')
    if destination == '4080':
        allowed.add('/home/carla/4T/SAFEFLOW2027_4080/recovery_backups/from_' + source_label)
    if target_root not in allowed:
        raise RuntimeError('unapproved checkpoint archive destination: ' + target_root)
    names = ' '.join(shlex.quote(name) for name in changed)
    sizes = ssh(source, 'du -sb -- ' + ' '.join(shlex.quote(source_export + '/' + name) for name in changed)).stdout
    transfer_bytes = sum(int(row.split()[0]) for row in sizes.splitlines())
    available = int(ssh(destination, 'python3 -c ' + shlex.quote(
        'import shutil; print(shutil.disk_usage(' + repr(target_root) + ').free)')).stdout.strip())
    if available < transfer_bytes + TRAINING_DISK_RESERVE_BYTES:
        log(f'{source} -> {destination}: checkpoint transfer deferred to preserve 2 GiB for training; free={available}, incoming={transfer_bytes}')
        return False
    send = 'tar -czf - -C ' + shlex.quote(source_export) + ' -- ' + names
    # Stage on the destination archive filesystem, including the 4080 4T disk.
    staging_parent = target_root.rsplit('/', 1)[0]
    installer = HOSTS[destination] + '/deploy/export_checkpoints.py'
    cleanup = ('python3 ' + shlex.quote(installer)
               + ' --discard-staging "$backup_stage" --export-root ' + shlex.quote(target_root))
    receive = ('set -e; backup_stage=$(mktemp -d ' + shlex.quote(staging_parent + '/.incoming.XXXXXX')
               + '); trap ' + shlex.quote(cleanup) + ' EXIT; tar -xzf - --no-same-owner -C "$backup_stage"; python3 ' + shlex.quote(installer)
               + ' --install-staging "$backup_stage" --export-root ' + shlex.quote(target_root))
    with subprocess.Popen([*SSH_OPTIONS, source, send], stdout=subprocess.PIPE) as producer:
        try:
            result = subprocess.run([*SSH_OPTIONS, destination, receive], stdin=producer.stdout, timeout=14400)
        finally:
            # A timed-out receiver must not leave the sender blocked on a pipe
            # while Popen.__exit__ waits indefinitely for it.
            producer.stdout.close()
            try:
                producer.wait(timeout=30)
            except subprocess.TimeoutExpired:
                producer.terminate()
                try:
                    producer.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    producer.kill()
                    producer.wait()
        if producer.returncode != 0 or result.returncode != 0:
            raise RuntimeError('checkpoint replication did not finish successfully')
    log(f'Backed up {len(changed)} checkpoint snapshots from {source} to {destination}')
    return True


def backup_source(state, source, root):
    command = ('python3 ' + shlex.quote(root + '/deploy/export_checkpoints.py') + ' --run-root ' + shlex.quote(root + '/runs/shapo_p47_four_env') + ' --export-root ' + shlex.quote(root + '/recovery_backups/export'))
    snapshot = json.loads(ssh(source, command).stdout)
    state.setdefault('observations', {})[source] = snapshot
    previous = state.setdefault('replicated', {}).setdefault(source, {})
    changed = []
    identities = {}
    for row in snapshot['runs']:
        if 'checkpoint' not in row:
            continue
        identity = hashlib.sha256(json.dumps(row['checkpoint'], sort_keys=True).encode()).hexdigest()
        identities[row['job']] = identity
        if previous.get(row['job']) != identity:
            changed.append(row['job'])
    state_save(state)
    if changed:
        destination = 'tjucta-a100' if source == '4080' else '4080'
        # Bound staging space to one run, rather than buffering a full wave.
        for name in changed:
            if replicate(source, destination, [name]):
                previous[name] = identities[name]
                state_save(state)
    else:
        log(f'{source}: no newer full checkpoints; existing backups retained')


def cycle(state):
    for source, root in HOSTS.items():
        try:
            backup_source(state, source, root)
        except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
            # An offline A100 must not prevent 5080 -> 4080/4T protection.
            log(f'{source}: backup deferred; other hosts continue: {error}')


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    while True:
        try:
            cycle(state)
        except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
            log('Backup observation/transfer failed; leaving training unchanged: ' + str(error))
        time.sleep(1800)


if __name__ == '__main__':
    main()
