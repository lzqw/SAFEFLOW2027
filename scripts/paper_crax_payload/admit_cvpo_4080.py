"""Transfer one genuinely completed 4080 seed to the original A100 closure.

No training, evaluation, retries, local model files, or full replay transfers.
Optionally subscribe once to the original native worker's exit using pidfd.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import shutil
import subprocess
import sys
import tempfile


SOURCE_COMMIT = '7abb6ebb8e139992fc88a9bcbd2fcebc8e55227e'
SEED = 51029
MILESTONES = [2000128, 4000000, 6000128, 8000000, 10000128]
SOURCE_ROOT = '/home/carla/4T/SAFEFLOW2027_NONHARDWARE_V4/e08-fsrl/runs/cvpo-51029-10m'
TARGET_ROOT = '/storage_0/SAFEFLOW2027_lzqw_uid1007/nonhardware_v4/e08-fsrl/runs/cvpo-51029-10m'
FILES = ['run_config.json', 'receipt.json', 'progress.jsonl', 'episodes.jsonl', 'status.json'] + [
    f'{kind}_{step}.{extension}' for step in MILESTONES
    for kind, extension in [('validation', 'json'), ('policy', 'pt')]]
RESERVE = 40 * 1024 ** 3


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def inventory(root, config):
    root = Path(root)
    manifest = read_json(root / 'run_config.json')
    receipt = read_json(root / 'receipt.json')
    if (manifest['configuration'] != config or manifest['training_seed'] != SEED
            or manifest['source_commit'] != SOURCE_COMMIT
            or receipt['source_commit'] != SOURCE_COMMIT
            or receipt['training_seed'] != SEED or receipt['status'] != 'completed'
            or receipt['steps'] != MILESTONES[-1]
            or receipt['evaluated_milestones'] != MILESTONES
            or receipt.get('engineering_only')):
        raise ValueError('Require the original complete registered seed; never admit a prefix')
    result = {}
    for name in FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing original regular file: {name}')
        result[name] = dict(bytes=path.stat().st_size, sha256=file_sha(path))
    for step in MILESTONES:
        record = read_json(root / f'validation_{step}.json')
        if (record['training_seed'] != SEED or record['steps'] != step
                or record['split'] != 'common_validation'
                or record['independent_test_used'] is not False
                or record['policy_sha256'] != result[f'policy_{step}.pt']['sha256']
                or [row['seed'] for row in record['results']] != config['development_evaluation_seeds']
                or any(row['episodes'] != 50 or len(row['records']) != 50
                       for row in record['results'])):
            raise ValueError(f'Invalid original validation or policy bytes at {step}')
    return result


def wait_original_worker(root, pid):
    if (Path(root) / 'receipt.json').exists():
        return
    descriptor = os.pidfd_open(pid)
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        start = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        if ('scripts.paper_crax_payload.train_cvpo ' not in command
                or '--seed 51029 ' not in command
                or f'--output {TARGET_ROOT} ' not in command + ' '
                or SOURCE_COMMIT not in command):
            raise ValueError('PID is not the original registered seed51029 native worker')
        print(json.dumps(dict(event='waiting_for_original_worker_exit', pid=pid,
                              kernel_starttime=start, command=command)), flush=True)
        select.select([descriptor], [], [])
    finally:
        os.close(descriptor)


def require_data_disk(target, incoming_bytes):
    # Only this registered data family is writable; never fall back to home.
    if str(target) != TARGET_ROOT or not os.path.ismount('/storage_0'):
        raise ValueError('A100 original mounted data-root required')
    device = subprocess.check_output(
        ['findmnt', '-n', '-o', 'SOURCE', '--target', '/storage_0'], text=True).strip()
    if device != '/dev/sdb':
        raise ValueError(f'Unexpected A100 data filesystem: {device}')
    if shutil.disk_usage(target.parent).free - incoming_bytes < RESERVE:
        raise ValueError('Preserve at least40GiB after transfer')


def admit_stage(stage, target, config, expected):
    stage, target = Path(stage), Path(target)
    if stage.parent != target.parent or not stage.name.startswith('.cvpo-51029-incoming-'):
        raise ValueError('Admission must be an adjacent private staging directory')
    if target.exists() or target.is_symlink():
        raise FileExistsError('Preserve any existing canonical seed directory')
    actual = inventory(stage, config)
    if actual != expected:
        raise ValueError('End-to-end transferred bytes differ; preserve staging for diagnosis')
    require_data_disk(target, 0)
    lock = target.parent / '.cvpo-51029-admission-lock'
    lock.mkdir()  # No competing helper can replace the canonical output.
    try:
        if target.exists() or target.is_symlink():
            raise FileExistsError('Canonical output appeared during transfer; do not replace it')
        os.rename(stage, target)
    finally:
        lock.rmdir()
    return dict(event='admitted_completed_seed', source_commit=SOURCE_COMMIT,
                training_seed=SEED, steps=MILESTONES[-1], files=actual,
                destination=str(target), training_or_evaluation_controls=0,
                full_replay_checkpoints_transferred=False)


def remote_action(operation, payload):
    if operation == 'wait':
        wait_original_worker(payload['root'], payload['pid'])
        return dict(event='original_worker_exited_or_receipt_present')
    if operation == 'inventory':
        return inventory(payload['root'], payload['config'])
    if operation == 'stage':
        target = Path(payload['target'])
        require_data_disk(target, sum(item['bytes'] for item in payload['inventory'].values()))
        if target.exists() or target.is_symlink():
            raise FileExistsError('Existing canonical output must not be overwritten')
        return tempfile.mkdtemp(prefix='.cvpo-51029-incoming-', dir=target.parent)
    if operation == 'admit':
        return admit_stage(payload['stage'], payload['target'], payload['config'], payload['inventory'])
    raise ValueError(operation)


def ssh_command(host, command):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=60', '-o', 'ServerAliveCountMax=3', host, command]


def remote(host, operation, payload, *, stream=False):
    code = Path(__file__).read_text()
    command = shlex.join(['python3', '-u', '-c', code, '--remote', operation, json.dumps(payload)])
    if stream:
        subprocess.run(ssh_command(host, command), check=True)
        return None
    return json.loads(subprocess.check_output(ssh_command(host, command), text=True))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--remote':
        print(json.dumps(remote_action(sys.argv[2], json.loads(sys.argv[3]))), flush=True)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-host', default='4080')
    parser.add_argument('--target-host', default='tjucta-a100')
    parser.add_argument('--wait-for-pid', type=int)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2]
                        / 'configs/paper_crax_payload/cvpo_formal_v4.json')
    args = parser.parse_args()
    config = read_json(args.config)
    if args.wait_for_pid:
        remote(args.source_host, 'wait', dict(root=SOURCE_ROOT, pid=args.wait_for_pid), stream=True)
    evidence = remote(args.source_host, 'inventory', dict(root=SOURCE_ROOT, config=config))
    stage = remote(args.target_host, 'stage', dict(target=TARGET_ROOT, inventory=evidence))
    print(json.dumps(dict(event='transferring_compact_completed_seed', staging=stage,
                          total_bytes=sum(item['bytes'] for item in evidence.values()))), flush=True)
    # Models travel directly through pipes; no files are written on the workstation.
    source_command = shlex.join(['tar', '-czf', '-', '-C', SOURCE_ROOT, '--'] + FILES)
    target_command = shlex.join(['tar', '-xzf', '-', '--no-same-owner', '--no-same-permissions', '-C', stage])
    with subprocess.Popen(ssh_command(args.source_host, source_command), stdout=subprocess.PIPE) as source:
        with subprocess.Popen(ssh_command(args.target_host, target_command), stdin=source.stdout) as target:
            source.stdout.close()
            target_code = target.wait()
        source_code = source.wait()
    if source_code or target_code:
        raise RuntimeError(f'Transfer failed ({source_code}, {target_code}); preserve staging; no retry')
    receipt = remote(args.target_host, 'admit', dict(stage=stage, target=TARGET_ROOT,
                                                    config=config, inventory=evidence))
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
