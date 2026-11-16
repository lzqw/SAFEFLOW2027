"""Merge a complete native resource-resumed CVPO seed, preserving both prefixes.

Registered seed51027/51028 handoff only. No model imports, learning, evaluation, or retries.
Run after the original resume worker has genuinely completed.
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
from datetime import datetime, timezone

COMMIT = '7abb6ebb8e139992fc88a9bcbd2fcebc8e55227e'
MILESTONES = [2000128, 4000000, 6000128, 8000000, 10000128]
PHYSICAL = Path('/home/carla/4T/SAFEFLOW2027_NONHARDWARE_V4/e08-fsrl')
PREFIX = PHYSICAL / 'resume-prefix-51027-v1'
SUFFIX = PHYSICAL / 'runs/cvpo-51027-resume4080-v1'
A100 = Path('/storage_0/SAFEFLOW2027_lzqw_uid1007/nonhardware_v4/e08-fsrl')
TARGET = A100 / 'runs/cvpo-51027-10m'
ARCHIVE = A100 / 'runs/cvpo-51027-prefix-a100-v1'
LOGICAL_PARENT = str(A100 / 'resume-prefix-51027-v1/interrupted_full.pt')
SEED = 51027
PREFIX_STEPS = 4401408
PREFIX_UPDATES = 274064


def configure(seed, prefix_steps, prefix_updates):
    global SEED, PREFIX_STEPS, PREFIX_UPDATES, PREFIX, SUFFIX, TARGET, ARCHIVE, LOGICAL_PARENT
    if seed not in (51027, 51028) or not 4000000 < prefix_steps < 6000128:
        raise ValueError('Only the registered two prefixes after4M and before6M may migrate')
    SEED, PREFIX_STEPS, PREFIX_UPDATES = seed, prefix_steps, prefix_updates
    PREFIX = PHYSICAL / f'resume-prefix-{seed}-v1'
    SUFFIX = PHYSICAL / f'runs/cvpo-{seed}-resume4080-v1'
    TARGET = A100 / f'runs/cvpo-{seed}-10m'
    ARCHIVE = A100 / f'runs/cvpo-{seed}-prefix-a100-v1'
    LOGICAL_PARENT = str(A100 / f'resume-prefix-{seed}-v1/interrupted_full.pt')


def registration():
    return dict(seed=SEED, prefix_steps=PREFIX_STEPS, prefix_updates=PREFIX_UPDATES)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():
            raise ValueError('Only regular original compact files allowed')
        if path.is_file():
            result[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return result


def assemble(prefix, suffix, parent_sha, destination_parent):
    prefix, suffix = Path(prefix), Path(suffix)
    before, manifest, receipt = read(prefix / 'status.json'), read(suffix / 'run_config.json'), read(suffix / 'receipt.json')
    original = read(prefix / 'run_config.json')
    config = original['configuration']
    if (before['state'] != 'engineering_stop_at_complete_collection_boundary'
            or before['steps'] != PREFIX_STEPS or before['official_updates'] != PREFIX_UPDATES
            or before['evaluated_milestones'] != MILESTONES[:2]
            or manifest['configuration'] != config or manifest['training_seed'] != SEED
            or original['training_seed'] != SEED or original['source_commit'] != COMMIT
            or manifest['source_commit'] != COMMIT or manifest['resume_parent'] != LOGICAL_PARENT
            or receipt['status'] != 'completed' or receipt['steps'] != MILESTONES[-1]
            or receipt['training_seed'] != SEED or receipt['source_commit'] != COMMIT
            or receipt['actual_new_transitions'] != MILESTONES[-1] - before['steps']
            or receipt['official_updates'] != 623984
            or receipt['evaluated_milestones'] != MILESTONES or receipt['evaluation_controls'] != 1250000
            or receipt.get('independent_test_used') is not False
            or sha(prefix / 'interrupted_full.pt') != parent_sha):
        raise ValueError('Wrong native boundary, lineage, full budget, updates, or scientific identity')
    pairs = []
    for step in MILESTONES:
        source = prefix if step in MILESTONES[:2] else suffix
        other = suffix if source == prefix else prefix
        validation = source / f'validation_{step}.json'
        policy = source / f'policy_{step}.pt'
        if (other / validation.name).exists() or (other / policy.name).exists():
            raise ValueError('Repeated milestone; preserve original evaluations, never choose between reruns')
        record = read(validation)
        if (record['steps'] != step or record['training_seed'] != SEED
                or record['split'] != 'common_validation' or record['independent_test_used'] is not False
                or record['policy_sha256'] != sha(policy)
                or [r['seed'] for r in record['results']] != config['development_evaluation_seeds']
                or any(r['episodes'] != 50 or len(r['records']) != 50 for r in record['results'])):
            raise ValueError('Wrong native evaluation identity')
        pairs.extend([validation, policy])
    stage = Path(tempfile.mkdtemp(prefix=f'completed-resume-{SEED}-', dir=destination_parent))
    for path in [*pairs, suffix / 'run_config.json', suffix / 'receipt.json', suffix / 'status.json']:
        shutil.copyfile(path, stage / path.name)
    # Keep both original manifests/logs as well as cumulative chronological logs.
    for label, source in [('prefix_a100', prefix), ('suffix_4080', suffix)]:
        output = stage / label
        output.mkdir()
        for name in ('run_config.json', 'status.json', 'progress.jsonl', 'episodes.jsonl'):
            shutil.copyfile(source / name, output / name)
    for name in ('progress.jsonl', 'episodes.jsonl'):
        with (stage / name).open('xb') as output:
            for source in (prefix, suffix):
                with (source / name).open('rb') as stream:
                    shutil.copyfileobj(stream, output)
    lineage = dict(schema='native_fullstate_resource_resume_v1', source_commit=COMMIT,
        training_seed=SEED, prefix_steps=before['steps'], suffix_actual_new_transitions=receipt['actual_new_transitions'],
        cumulative_transitions=receipt['steps'], prefix_official_updates=before['official_updates'],
        cumulative_official_updates=receipt['official_updates'],
        parent_interrupted_checkpoint_sha256=parent_sha,
        original_prefix_manifest_sha256=sha(prefix / 'run_config.json'),
        original_prefix_status_sha256=sha(prefix / 'status.json'),
        original_prefix_progress_sha256=sha(prefix / 'progress.jsonl'),
        original_prefix_episodes_sha256=sha(prefix / 'episodes.jsonl'),
        original_prefix_large_checkpoints_preserved_on_a100=True,
        suffix_large_checkpoints_preserved_on_4080=True,
        previous_evaluations_not_repeated=True, training_budget_not_extended=True,
        receipt_time_is_cumulative_active_runtime_not_transfer_wait_wallclock=True,
        native_controls_added_by_merge=0, assembled_at_utc=datetime.now(timezone.utc).isoformat())
    write(stage / 'resume_lineage.json', lineage)
    return dict(stage=str(stage), files=inventory(stage), lineage=lineage)


def require_a100_space(size):
    device = subprocess.check_output(['findmnt', '-n', '-o', 'SOURCE', '--target', '/storage_0'], text=True).strip()
    if not os.path.ismount('/storage_0') or device != '/dev/sdb':
        raise ValueError('Require mounted original A100 data filesystem')
    if shutil.disk_usage(A100).free - size < 40 * 1024 ** 3:
        raise ValueError('Preserve40GiB reserve; never write models in home')


def admit(stage, expected):
    stage = Path(stage)
    if stage.parent != TARGET.parent or not stage.name.startswith(f'.cvpo-{SEED}-incoming-'):
        raise ValueError('Wrong canonical staging family')
    if inventory(stage) != expected or ARCHIVE.exists() or ARCHIVE.is_symlink():
        raise ValueError('Changed transfer bytes or archive already exists; do not overwrite')
    if (TARGET / 'receipt.json').exists():
        raise ValueError('Canonical seed already completed; preserve it')
    lineage = read(stage / 'resume_lineage.json')
    originals = dict(run_config='manifest', status='status', progress='progress', episodes='episodes')
    for name, key in originals.items():
        extension = 'jsonl' if name in ('progress', 'episodes') else 'json'
        if sha(TARGET / f'{name}.{extension}') != lineage[f'original_prefix_{key}_sha256']:
            raise ValueError('Original A100 prefix changed')
    if sha(TARGET / 'interrupted_full.pt') != lineage['parent_interrupted_checkpoint_sha256']:
        raise ValueError('Original complete boundary changed')
    require_a100_space(0)
    # Original closure cannot admit the prefix: it has no complete receipt.
    # Preserve it on the same data disk before exposing a full canonical result.
    os.rename(TARGET, ARCHIVE)
    try:
        os.rename(stage, TARGET)
    except BaseException:
        os.rename(ARCHIVE, TARGET)
        raise
    result = dict(event='complete_resumed_seed_atomically_admitted', training_seed=SEED,
        destination=str(TARGET), preserved_original_prefix=str(ARCHIVE),
        fullstate_parent_sha256=lineage['parent_interrupted_checkpoint_sha256'],
        steps=10000128, actual_new_transitions=10000128 - PREFIX_STEPS,
        all_transferred_files_sha_equal=True, controls_added_by_transfer=0)
    write(TARGET / 'resource_admission.json', result)
    return result


def remote_action(operation, payload):
    if 'registration' in payload:
        configure(**payload['registration'])
    if operation == 'wait':
        if not (SUFFIX / 'receipt.json').exists():
            pid = payload['pid']
            descriptor = os.pidfd_open(pid)
            try:
                command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
                if ('scripts.paper_crax_payload.train_cvpo ' not in command
                        or f'--seed {SEED} ' not in command
                        or f'--resume {LOGICAL_PARENT}' not in command
                        or f'--output {A100}/runs/cvpo-{SEED}-resume4080-v1 ' not in command):
                    raise ValueError('Not the exact native fullstate continuation worker')
                print(json.dumps(dict(event='waiting_for_native_resume_exit', pid=pid,
                                      command=command)), flush=True)
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
        return dict(event='native_resume_exited_or_receipt_present')
    if operation == 'assemble':
        return assemble(PREFIX, SUFFIX, payload['parent_sha'], PHYSICAL)
    if operation == 'stage':
        require_a100_space(sum(r['bytes'] for r in payload['files'].values()))
        if ARCHIVE.exists() or (TARGET / 'receipt.json').exists():
            raise FileExistsError('Prior admission exists; do not retry automatically')
        return tempfile.mkdtemp(prefix=f'.cvpo-{SEED}-incoming-', dir=TARGET.parent)
    if operation == 'admit':
        return admit(payload['stage'], payload['files'])
    raise ValueError(operation)


def ssh(host, command):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, command]


def remote(host, operation, payload):
    payload = dict(payload, registration=registration())
    python = '/home/lzqw/SAFEFLOW2027_PAPER/env/bin/python' if host == 'tjucta-a100' else 'python3'
    command = shlex.join([python, '-c', Path(__file__).read_text(), '--remote', operation, json.dumps(payload)])
    return json.loads(subprocess.check_output(ssh(host, command), text=True))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--remote':
        print(json.dumps(remote_action(sys.argv[2], json.loads(sys.argv[3]))), flush=True)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--wait-for-pid', type=int)
    parser.add_argument('--seed', type=int, default=51027)
    parser.add_argument('--prefix-steps', type=int, default=4401408)
    parser.add_argument('--prefix-updates', type=int, default=274064)
    args = parser.parse_args()
    configure(args.seed, args.prefix_steps, args.prefix_updates)
    if args.wait_for_pid:
        command = shlex.join(['python3', '-u', '-c', Path(__file__).read_text(), '--remote', 'wait',
                              json.dumps(dict(pid=args.wait_for_pid, registration=registration()))])
        subprocess.run(ssh('4080', command), check=True)
    combined = remote('4080', 'assemble', dict(parent_sha=args.parent_sha256))
    stage = remote('tjucta-a100', 'stage', dict(files=combined['files']))
    print(json.dumps(dict(event='transferring_complete_resumed_seed', source=combined['stage'], staging=stage)), flush=True)
    source_cmd = shlex.join(['tar', '-czf', '-', '-C', combined['stage'], '.'])
    target_cmd = shlex.join(['tar', '-xzf', '-', '--no-same-owner', '-C', stage])
    with subprocess.Popen(ssh('4080', source_cmd), stdout=subprocess.PIPE) as source:
        with subprocess.Popen(ssh('tjucta-a100', target_cmd), stdin=source.stdout) as target:
            source.stdout.close()
            target_code = target.wait()
        source_code = source.wait()
    if source_code or target_code:
        raise RuntimeError('Transfer failed; preserve staging, no automatic retries')
    result = remote('tjucta-a100', 'admit', dict(stage=stage, files=combined['files']))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(json.dumps(dict(event='resource_resume_closure_failed', exit_code=error.returncode,
                              automatic_retry=False)), file=sys.stderr, flush=True)
        raise SystemExit(1)
