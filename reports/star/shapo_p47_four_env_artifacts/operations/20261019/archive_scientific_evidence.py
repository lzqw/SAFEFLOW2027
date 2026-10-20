#!/usr/bin/env python3
"""Archive raw experiment evidence on the 4080 4T disk; never evaluate a policy."""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
HOSTS = {'tjucta-a100': '/home/lzqw/SAFEFLOW2027_A100',
         '4080': '/home/carla/SAFEFLOW2027_4080',
         '5080': '/home/autodrive/LZQW/SAFEFLOW2027'}
ARCHIVE = Path('/home/carla/4T/SAFEFLOW2027_4080/scientific_evidence_archive')
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
STATE = BASE / 'scientific_evidence_archive_state.json'
JOBS = {f'{slug}-{seed}' for slug in ('point-goal', 'point-button', 'ant', 'walker2d')
        for seed in range(2027, 2032)}
SUFFIXES = {'.json', '.yaml', '.yml', '.txt', '.csv', '.log', '.npz', '.pt'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, path)


def check_manifest(manifest):
    if (manifest.get('format') != 'safeflow-raw-evidence-snapshot-v1'
            or manifest.get('origin') not in HOSTS or manifest.get('job') not in JOBS
            or manifest.get('source_commit') != SOURCE):
        raise ValueError('unexpected evidence identity')
    expected = HOSTS[manifest['origin']] + '/runs/shapo_p47_four_env/runs/' + manifest['job']
    if manifest.get('source_container') != expected:
        raise ValueError('unexpected original evidence container')
    paths = set()
    for row in manifest['files']:
        p = Path(row['path'])
        if (p.is_absolute() or '..' in p.parts or not p.parts or row['path'] in paths
                or not re.fullmatch('[0-9a-f]{64}', row['sha256']) or row['bytes'] < 0):
            raise ValueError('invalid raw evidence entry')
        paths.add(row['path'])


def snapshot(origin, job):
    if origin not in HOSTS or job not in JOBS:
        raise ValueError('out-of-scope evidence job')
    root = Path(HOSTS[origin])
    container = root / 'runs/shapo_p47_four_env/runs' / job
    if container.is_symlink() or not container.is_dir():
        raise ValueError('missing or symlinked source container')
    if (container / 'source_commit.txt').read_text().strip() != SOURCE:
        raise ValueError('scientific source changed')
    parent = root / 'recovery_backups'
    parent.mkdir(exist_ok=True)
    candidates = [p for p in sorted(container.rglob('*'))
                  if p.suffix in SUFFIXES and not p.is_symlink() and p.is_file()]
    snapshot_bytes = sum(p.stat().st_size for p in candidates if p.exists())
    if shutil.disk_usage(parent).free < snapshot_bytes + 2 * 1024**3:
        raise RuntimeError('preserving source training disk reserve')
    stage = Path(tempfile.mkdtemp(prefix='.evidence-snapshot.', dir=parent))
    print(json.dumps({'snapshot_stage_started': str(stage)}), flush=True)
    (stage / 'blobs').mkdir()
    rows = []
    for p in candidates:
        temporary = stage / 'file.pending'
        try:
            # Do not link raw checkpoint inodes: changing their link count/ctime
            # interferes with a concurrent rolling-backup tar reader. Frozen
            # training replaces checkpoint paths atomically, so an open FD keeps
            # stable bytes while the raw pathname may advance to a new inode.
            # Bound live-log reads so an appending writer cannot prolong copying.
            with p.open('rb') as source, temporary.open('wb') as target:
                source_stat = os.fstat(source.fileno())
                remaining = source_stat.st_size
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    target.write(block)
                    remaining -= len(block)
        except FileNotFoundError:
            if temporary.exists():
                temporary.unlink()
            continue
        sha = digest(temporary)
        size = temporary.stat().st_size
        blob = stage / 'blobs' / sha
        if blob.exists():
            temporary.unlink()
        else:
            temporary.rename(blob)
        rows.append({'path': p.relative_to(container).as_posix(), 'sha256': sha, 'bytes': size,
                     'mtime_ns': source_stat.st_mtime_ns, 'mode': source_stat.st_mode & 0o777})
    manifest = {'format': 'safeflow-raw-evidence-snapshot-v1', 'origin': origin, 'job': job,
                'source_container': str(container), 'source_commit': SOURCE,
                'created_at_ns': time.time_ns(), 'files': rows,
                'scientific_parameters_changed': False, 'additional_evaluation': False,
                'is_final_scientific_audit': False,
                'restore_metadata_note': 'Restore each relative path from its content-hash blob and apply its recorded mode/mtime_ns; blob filesystem timestamps are not original artifact timestamps.',
                'live_snapshot_caveat': 'Mutable logs may end in a partial row. This raw backup is not a sealed final report or proof of complete provenance. Older lost evidence is not reconstructed.'}
    check_manifest(manifest)
    write_json(stage / 'manifest.json', manifest)
    return {'stage': str(stage), 'manifest': manifest}


def archive_root():
    if not os.path.ismount('/home/carla/4T'):
        raise RuntimeError('4T disk is not mounted; do not write to the system disk')
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.is_symlink() or ARCHIVE.resolve() != ARCHIVE:
        raise ValueError('unexpected archive location')
    (ARCHIVE / 'blobs').mkdir(exist_ok=True)
    return ARCHIVE


def missing(manifest):
    check_manifest(manifest)
    root = archive_root()
    result = []
    for sha, size in {r['sha256']: r['bytes'] for r in manifest['files']}.items():
        p = root / 'blobs' / sha
        if p.exists():
            if p.is_symlink() or not p.is_file() or p.stat().st_size != size:
                raise ValueError('existing archived blob needs inspection')
        else:
            result.append(sha)
    sizes = {r['sha256']: r['bytes'] for r in manifest['files']}
    required = sum(sizes[sha] for sha in result) + 2 * 1024**3
    if shutil.disk_usage(root).free < required:
        raise RuntimeError('insufficient 4T archive headroom; original evidence retained')
    return result


def install(stage):
    root = archive_root()
    if stage.is_symlink():
        raise ValueError('refusing symlink transfer staging directory')
    stage = stage.resolve()
    if (stage.parent != root.parent or not stage.name.startswith('.evidence-incoming.')
            or stage.is_symlink() or (stage / 'blobs').is_symlink()
            or (stage / 'manifest.json').is_symlink()):
        raise ValueError('unexpected incoming evidence staging directory')
    manifest = json.loads((stage / 'manifest.json').read_text())
    check_manifest(manifest)
    sizes = {r['sha256']: r['bytes'] for r in manifest['files']}
    for p in (stage / 'blobs').iterdir():
        if (p.name not in sizes or p.is_symlink() or not p.is_file()
                or p.stat().st_size != sizes[p.name] or digest(p) != p.name):
            raise ValueError('incoming evidence checksum mismatch')
        destination = root / 'blobs' / p.name
        if destination.exists():
            if destination.is_symlink() or destination.stat().st_size != sizes[p.name]:
                raise ValueError('existing evidence blob conflict')
            p.unlink()  # Verified redundant incoming copy, not original evidence.
        else:
            p.rename(destination)
    if missing(manifest):
        raise ValueError('archive manifest has missing raw blobs')
    directory = root / 'manifests' / manifest['origin'] / manifest['job']
    directory.mkdir(parents=True, exist_ok=True)
    name = str(manifest['created_at_ns']) + '-' + digest(stage / 'manifest.json') + '.json'
    target = directory / name
    if target.exists():
        if target.read_bytes() != (stage / 'manifest.json').read_bytes():
            raise ValueError('immutable evidence manifest conflict')
        (stage / 'manifest.json').unlink()
    else:
        (stage / 'manifest.json').rename(target)
    (stage / 'blobs').rmdir()
    stage.rmdir()
    return {'archived': True, 'manifest': str(target), 'raw_files': len(manifest['files']),
            'is_final_scientific_audit': False}


def cleanup(origin, stage):
    p = Path(stage)
    expected = Path(HOSTS[origin]) / 'recovery_backups'
    if p.is_symlink() or p.parent.resolve() != expected.resolve() or not p.name.startswith('.evidence-snapshot.'):
        raise ValueError('refusing unrelated snapshot cleanup')
    if p.exists():
        shutil.rmtree(p)


def background_prefix(host):
    """Only archive helpers: keep training processes and their affinities untouched."""
    assert host in HOSTS
    affinity = 'taskset -c 16-27 ' if host == '4080' else ''
    return affinity + 'nice -n 10 '


def remote(host, arguments, *, input_text=None, timeout=600):
    command = background_prefix(host) + 'python3 ' + shlex.quote(HOSTS[host] + '/deploy/archive_scientific_evidence.py')
    command += ' ' + ' '.join(shlex.quote(str(x)) for x in arguments)
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', host, command],
                          input=input_text, text=True, capture_output=True, check=True, timeout=timeout).stdout


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), message, flush=True)


def transfer(host, pending):
    manifest = pending['manifest']
    needed = json.loads(remote('4080', ['--missing'], input_text=json.dumps(manifest)))
    names = ['manifest.json'] + ['blobs/' + sha for sha in needed]
    for sha in needed:
        if not re.fullmatch('[0-9a-f]{64}', sha):
            raise ValueError('invalid requested blob')
    send = background_prefix(host) + 'tar -czf - -C ' + shlex.quote(pending['stage']) + ' -- ' + ' '.join(map(shlex.quote, names))
    receiver = HOSTS['4080'] + '/deploy/archive_scientific_evidence.py'
    receive = ('set -e; test -d /home/carla/4T; mountpoint -q /home/carla/4T; '
               'evidence_stage=$(mktemp -d ' + shlex.quote(str(ARCHIVE.parent / '.evidence-incoming.XXXXXX'))
               + '); mkdir "$evidence_stage/blobs"; ' + background_prefix('4080')
               + 'tar -xzf - --no-same-owner -C "$evidence_stage"; '
               + background_prefix('4080') + 'python3 ' + shlex.quote(receiver) + ' --install "$evidence_stage"')
    with subprocess.Popen(['ssh', '-o', 'BatchMode=yes', host, send], stdout=subprocess.PIPE) as producer:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '4080', receive], stdin=producer.stdout,
                                text=True, capture_output=True, timeout=7200)
        producer.stdout.close()
        if producer.wait() or result.returncode:
            raise RuntimeError('evidence transfer failed; retained source snapshot: ' + result.stderr[-500:])
    return json.loads(result.stdout)


def run_service():
    state = json.loads(STATE.read_text()) if STATE.exists() else {'pending': {}, 'archived': {}}
    # An idle local coordinator restart must not cause an extra archive cycle.
    # Existing pending transfers still resume immediately after a host reboot.
    delay = min(1800, max(0, state.get('last_success_at_unix', 0) + 1800 - time.time()))
    if not state['pending'] and delay:
        log(f'Preserving existing archive cadence; background helpers; startup wait {delay:.1f}s')
        time.sleep(delay)
    while True:
        for host in HOSTS:
            try:
                jobs = json.loads(remote(host, ['--list', '--origin', host]))
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                log(f'{host}: evidence listing deferred ({type(error).__name__}); training unchanged')
                continue
            for job in jobs:
                key = host + '/' + job
                try:
                    if key not in state['pending']:
                        try:
                            output = remote(host, ['--snapshot', job, '--origin', host])
                        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                            output = error.stdout or ''
                            if isinstance(output, bytes):
                                output = output.decode(errors='replace')
                            for line in output.splitlines():
                                try:
                                    started = json.loads(line).get('snapshot_stage_started')
                                except (ValueError, AttributeError):
                                    continue
                                if started:
                                    state['pending'][key] = {'stage': started, 'snapshot_incomplete': True}
                                    write_json(STATE, state)
                            raise
                        state['pending'][key] = json.loads(output.splitlines()[-1])
                        write_json(STATE, state)
                    pending = state['pending'][key]
                    if pending.get('snapshot_incomplete'):
                        raise RuntimeError('prior snapshot incomplete; preserve its staging files for inspection')
                    if 'archived_receipt' not in pending:
                        pending['archived_receipt'] = transfer(host, pending)
                        write_json(STATE, state)
                    remote(host, ['--cleanup', pending['stage'], '--origin', host])
                    state['archived'][key] = pending['archived_receipt']
                    del state['pending'][key]
                    state['last_success_at_unix'] = time.time()
                    write_json(STATE, state)
                    log(f'Archived raw evidence {key}: ' + pending['archived_receipt']['manifest'])
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    log(f'{key}: evidence archive deferred ({type(error).__name__}): {str(error)[-600:]}; training unchanged')
        log('Evidence archive cycle complete; no scientific evaluation executed')
        time.sleep(1800)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin', choices=HOSTS)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--snapshot', choices=sorted(JOBS))
    parser.add_argument('--missing', action='store_true')
    parser.add_argument('--install', type=Path)
    parser.add_argument('--cleanup')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        run_service()
    elif args.list:
        root = Path(HOSTS[args.origin]) / 'runs/shapo_p47_four_env/runs'
        print(json.dumps(sorted(p.name for p in root.iterdir() if p.name in JOBS and p.is_dir())))
    elif args.snapshot:
        print(json.dumps(snapshot(args.origin, args.snapshot)), flush=True)
    elif args.missing:
        import sys
        print(json.dumps(missing(json.load(sys.stdin))))
    elif args.install:
        print(json.dumps(install(args.install)))
    elif args.cleanup:
        cleanup(args.origin, args.cleanup)
    else:
        parser.error('an explicit archive operation is required')


if __name__ == '__main__':
    main()
