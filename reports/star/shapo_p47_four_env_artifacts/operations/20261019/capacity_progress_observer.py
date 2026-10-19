#!/usr/bin/env python3
"""Observe GPU/disk capacity and logged progress without changing training."""

import csv
import json
import os
import shlex
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

OPS = Path(__file__).resolve().parent
STATE = OPS / 'gpu_capacity_state.json'
PROJECTS = {
    'tjucta-a100': '/home/lzqw/SAFEFLOW2027_A100',
    '4080': '/home/carla/SAFEFLOW2027_4080',
    '5080': '/home/autodrive/LZQW/SAFEFLOW2027',
}
ARCHIVE_STORAGE = {'4080': '/home/carla/4T/SAFEFLOW2027_4080'}
POLL_SECONDS = 1800
STABLE_SECONDS = 1800
MAX_REBALANCE_GPU_UTILIZATION_PERCENT = 85

REMOTE = r'''
import csv,json,math,shutil,subprocess,sys,time
from pathlib import Path
project=Path(sys.argv[1]).resolve()
def query(fields):
    result=subprocess.run(['nvidia-smi',fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    return list(csv.reader(result.stdout.splitlines()))
def role(pid):
    try:
        proc=Path('/proc')/str(pid)
        args=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        executable=Path(args[0]).name if args else 'unknown'
        if executable in {'Xorg','Xwayland','kwin_wayland','kwin_x11','gnome-shell',
                          'gnome-remote-desktop-daemon','plasmashell'}:
            return 'graphics',executable
        try:
            (proc/'cwd').resolve(strict=True).relative_to(project)
            return 'safeflow',executable
        except (OSError,ValueError):
            pass
        for arg in args[1:]:
            if not arg.startswith('/'):
                continue
            try:
                Path(arg).relative_to(project)
                return 'safeflow',executable
            except ValueError:
                pass
        return 'foreign_or_unknown',executable
    except OSError:
        return 'foreign_or_unknown','unknown'
gpus=[]
for row in query('--query-gpu=index,uuid,name,memory.free,memory.total,utilization.gpu'):
    row=[x.strip() for x in row]
    gpus.append({'index':int(row[0]),'uuid':row[1],'name':row[2],'free_mib':int(row[3]),'total_mib':int(row[4]),'utilization_percent':int(row[5]),'processes':[]})
for row in query('--query-compute-apps=gpu_uuid,pid,used_memory'):
    row=[x.strip() for x in row]
    if len(row)!=3:
        continue
    gpu=next((g for g in gpus if g['uuid']==row[0]),None)
    if gpu is None:
        continue
    pid=int(row[1]);kind,executable=role(pid)
    memory=int(row[2]) if row[2].isdigit() else None
    gpu['processes'].append({'pid':pid,'role':kind,'executable':executable,'memory_mib':memory})
storage=[]
locations=[('project_filesystem',project,20)]
if len(sys.argv)>2:
    locations.append(('archive_4t_filesystem',Path(sys.argv[2]),100))
for label,path,minimum_free_gib in locations:
    entry={'label':label,'path':str(path),'warning_below_free_gib':minimum_free_gib}
    try:
        usage=shutil.disk_usage(path)
        separate=path.stat().st_dev!=project.stat().st_dev
        status=('archive_not_separate_filesystem' if label=='archive_4t_filesystem' and not separate
                else 'low_space' if usage.free<minimum_free_gib*1024**3 else 'ok')
        entry.update(status=status,free_gib=round(usage.free/1024**3,2),
                     total_gib=round(usage.total/1024**3,2),
                     separate_from_project_filesystem=separate)
    except OSError as error:
        entry.update(status='unknown',error_type=type(error).__name__)
    storage.append(entry)
def last_complete_csv_row(stream):
    header_bytes=stream.readline()
    if not header_bytes.endswith(b'\n'):
        return {}
    header=next(csv.reader([header_bytes.decode('utf-8')]))
    stream.seek(0,2)
    offset=max(len(header_bytes),stream.tell()-65536)
    stream.seek(offset)
    tail=stream.read()
    if offset>len(header_bytes):
        tail=tail.partition(b'\n')[2]
    # A writer may currently be appending a row. Ignore its incomplete tail.
    for line in reversed(tail.split(b'\n')[:-1]):
        if not line.strip():
            continue
        values=next(csv.reader([line.decode('utf-8')]))
        if len(values)==len(header):
            return dict(zip(header,values))
    return {}
def finite_number(value):
    number=float(value)
    return number if math.isfinite(number) else None
training=[]
run_root=project/'runs/shapo_p47_four_env/runs'
boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args=[x.decode(errors='replace') for x in cmdline.read_bytes().split(b'\0') if x]
        if not any(x.endswith('scripts/star/train_star.py') for x in args):
            continue
        config=Path(args[args.index('--config')+1])
        if config.parent.parent!=run_root:
            continue
        fields=(cmdline.parent/'stat').read_text().rsplit(') ',1)[1].split()
        worker={'job':config.parent.name,'pid':int(cmdline.parent.name),
                'start_ticks':int(fields[19]),'state':fields[0],
                'boot_id':boot_id,'device':args[args.index('--device')+1],
                'progress_status':'no_logged_epoch_yet','steps':None,'last_epoch_fps':None}
    except (OSError,ValueError,IndexError):
        continue
    try:
        files=list(config.parent.glob('training/**/progress.csv'))
        latest=max(files,key=lambda path:path.stat().st_mtime_ns) if files else None
        if latest:
            worker.update(progress_csv=str(latest),
                          progress_age_seconds=max(0,time.time()-latest.stat().st_mtime))
            with latest.open('rb') as stream:
                row=last_complete_csv_row(stream)
            if row.get('TotalEnvSteps'):
                steps=finite_number(row['TotalEnvSteps'])
                if steps is not None and steps>=0 and steps.is_integer():
                    worker.update(steps=int(steps),progress_status='observed')
            if row.get('Time/FPS'):
                worker['last_epoch_fps']=finite_number(row['Time/FPS'])
    except (OSError,ValueError,csv.Error) as error:
        worker.update(progress_status='unknown',progress_error_type=type(error).__name__)
    training.append(worker)
print(json.dumps({'observed_at_unix':time.time(),'gpus':gpus,'storage':storage,
                  'training':training}))
'''


def log(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


def probe(host):
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
         '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', host,
         'python3 - ' + shlex.quote(PROJECTS[host]) +
         (' ' + shlex.quote(ARCHIVE_STORAGE[host]) if host in ARCHIVE_STORAGE else '')],
        input=REMOTE, text=True, capture_output=True, timeout=45, check=True,
    )
    return json.loads(result.stdout)


def load_previous():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def save(payload):
    with tempfile.NamedTemporaryFile(mode='w', dir=OPS, prefix='gpu_capacity_',
                                     suffix='.json.tmp', delete=False) as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, STATE)


def cycle(previous):
    now = time.time()
    payload = {
        'observed_at_unix': now, 'poll_seconds': POLL_SECONDS,
        'stable_seconds': STABLE_SECONDS, 'hosts': {},
        'policy': {'pause_foreign_jobs': False, 'reduce_training_steps': False,
                   'automatic_file_deletion': False, 'storage_alerts_only': True,
                   'change_training_source': False, 'matrix_cells': 20},
        'action': 'Observe and notify only. The active goal agent handles any '
                  'existing-job migration at a full checkpoint; no automatic '
                  'termination, new seeds, or scientific evaluations.',
    }
    with ThreadPoolExecutor(max_workers=len(PROJECTS)) as executor:
        futures = {host: executor.submit(probe, host) for host in PROJECTS}
        for host, future in futures.items():
            try:
                observed = future.result()
            except Exception as error:
                payload['hosts'][host] = {'status': 'unknown',
                                         'error_type': type(error).__name__}
                log(f'{host}: capacity unknown ({type(error).__name__}); no action')
                continue
            old_host = previous.get('hosts', {}).get(host, {})
            elapsed = observed['observed_at_unix'] - old_host.get('observed_at_unix', 0)
            for worker in observed.get('training', []):
                old_worker = next((item for item in old_host.get('training', [])
                                   if all(item.get(key) == worker.get(key) for key in
                                          ('job', 'pid', 'start_ticks', 'boot_id', 'progress_csv'))), {})
                if (worker.get('steps') is not None and old_worker.get('steps') is not None
                        and 0 < elapsed < 3 * POLL_SECONDS):
                    delta = worker['steps'] - old_worker['steps']
                    worker.update(logged_step_delta=delta, observation_interval_seconds=elapsed,
                                  observed_logged_steps_per_second=delta / elapsed)
                    if delta < 0:
                        log(f'{host} {worker["job"]}: logged counter decreased; '
                            'inspect evidence, no automatic restart')
            summary = ', '.join(f'{worker["job"]}={worker.get("steps")}'
                                for worker in observed.get('training', []))
            log(f'{host}: {len(observed.get("training", []))} training processes; '
                f'logged steps: {summary or "none observed"}; '
                'missing epochs alone are not a stopped-training verdict')
            for volume in observed.get('storage', []):
                log(f'{host} {volume["label"]}: {volume["status"]}; '
                    f'free={volume.get("free_gib", "unknown")} GiB; '
                    'alerts only, no files deleted or training stopped')
            for gpu in observed['gpus']:
                old = next((g for g in old_host.get('gpus', [])
                            if g['uuid'] == gpu['uuid']), {})
                foreign = sorted(p['pid'] for p in gpu['processes']
                                 if p['role'] == 'foreign_or_unknown')
                own = sum(p['role'] == 'safeflow' for p in gpu['processes'])
                reserve = 8192 if gpu['total_mib'] <= 24576 else 16384
                eligible = (not foreign and gpu['free_mib'] >= reserve
                            and gpu['utilization_percent'] <= MAX_REBALANCE_GPU_UTILIZATION_PERCENT)
                recent = now - old_host.get('observed_at_unix', 0) < 3 * POLL_SECONDS
                since = old.get('candidate_since_unix') if recent else None
                since = (since if since is not None else now) if eligible else None
                status = ('available_for_rebalance' if eligible and now-since >= STABLE_SECONDS
                          else 'candidate_waiting_stability' if eligible
                          else 'occupied_or_insufficient_headroom')
                gpu.update(status=status, foreign_pids=foreign,
                           safeflow_gpu_processes=own, reserve_mib=reserve,
                           max_rebalance_utilization_percent=MAX_REBALANCE_GPU_UTILIZATION_PERCENT,
                           candidate_since_unix=since)
                signature = [status, foreign, own]
                if signature != old.get('signature'):
                    log(f'{host} GPU {gpu["index"]}: {status}; '
                        f'free={gpu["free_mib"]} MiB, foreign={len(foreign)}, '
                        f'SAFEFLOW={own}; other projects unchanged')
                gpu['signature'] = signature
            observed['status'] = 'observed'
            payload['hosts'][host] = observed
    save(payload)
    return payload


def main():
    previous = load_previous()
    heartbeat = 0
    # A monitor-only restart must not add an out-of-schedule SSH sampling wave.
    next_due = previous.get('observed_at_unix', 0) + POLL_SECONDS
    if next_due > time.time():
        time.sleep(min(POLL_SECONDS, next_due - time.time()))
    while True:
        previous = cycle(previous)
        if time.time() - heartbeat >= 1800:
            log('Capacity observation complete; full budgets retained; no foreign jobs stopped')
            heartbeat = time.time()
        time.sleep(POLL_SECONDS)


if __name__ == '__main__':
    main()
