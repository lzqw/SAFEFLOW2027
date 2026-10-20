#!/usr/bin/env python3
"""Read only metadata for the20 current matrix containers; no model/GPU work."""
import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOTS = {'tjucta-a100': '/home/lzqw/SAFEFLOW2027_A100',
         '4080': '/home/carla/SAFEFLOW2027_4080',
         '5080': '/home/autodrive/LZQW/SAFEFLOW2027'}


def probe(host, jobs):
    script = r'''
import json,shutil,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);jobs=json.loads(sys.argv[2])
def du(path,apparent):
 args=['du','-s','-B1']+(['--apparent-size'] if apparent else [])+['--',str(path)]
 return int(subprocess.check_output(args,text=True,timeout=45).split()[0])
rows=[]
for job in jobs:
 d=root/'runs/shapo_p47_four_env/runs'/job
 if not d.is_dir():rows.append({'job':job,'status':'missing_no_assumed_zero'});continue
 checkpoints=list(d.glob('training/**/checkpoints/latest_full.pt'))
 finals=list(d.glob('training/**/checkpoints/final_full.pt'))
 rows.append({'job':job,'container':str(d),'status':'metadata_observed',
  'physical_bytes':du(d,False),'apparent_bytes':du(d,True),
  'latest_full_files':len(checkpoints),'largest_current_latest_full_bytes':max((p.stat().st_size for p in checkpoints),default=None),
  'final_full_files':len(finals),'container_is_symlink':d.is_symlink()})
backups=[]
if sys.argv[3]=='include_backup_top_level_metadata':
 p=root/'recovery_backups'
 if p.is_dir():
  for d in sorted(p.iterdir()):
   if d.is_dir() and not d.is_symlink():backups.append({'path':str(d),'physical_bytes':du(d,False),'apparent_bytes':du(d,True),'offload_authorized_by_this_inventory':False})
available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
shm=shutil.disk_usage('/dev/shm')
print(json.dumps({'host_observed_at_unix':time.time(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 'filesystem_free_bytes':shutil.disk_usage(root).free,'memory_available_bytes':available,
 'shm_total_bytes':shm.total,'shm_free_bytes':shm.free,'current_matrix_jobs':rows,'backup_top_level':backups,
 'only_filesystem_metadata_read':True,'GPU_queries_or_model_loads_or_file_changes':0}))
'''
    prefix = 'taskset -c 16-27 nice -n 10 ' if host == '4080' else 'nice -n 10 '
    args = [ROOTS[host], json.dumps(jobs),
            'include_backup_top_level_metadata' if host == 'tjucta-a100' else 'skip_backups']
    command = prefix + 'python3 -c ' + shlex.quote(script) + ' ' + ' '.join(map(shlex.quote, args))
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', host, command],
                       text=True, capture_output=True, check=True, timeout=180)
    return json.loads(r.stdout)


def main():
    native = json.loads((BASE / 'capacity_progress_observation_20261021_0652.json').read_text())
    jobs = {h: [w['job'] for w in d['training']] for h, d in native['hosts'].items()}
    assert sum(map(len, jobs.values())) == 18
    jobs['tjucta-a100'] += ['ant-2027', 'walker2d-2027']
    assert len({j for values in jobs.values() for j in values}) == 20
    with ThreadPoolExecutor(max_workers=3) as executor:
        tasks = {h: executor.submit(probe, h, names) for h, names in jobs.items()}
        observations = {h: task.result() for h, task in tasks.items()}
    assert all(r['status'] == 'metadata_observed' for d in observations.values() for r in d['current_matrix_jobs'])
    incoming = sum(r['apparent_bytes'] for h, d in observations.items() if h != 'tjucta-a100' for r in d['current_matrix_jobs'])
    free = observations['tjucta-a100']['filesystem_free_bytes']
    result = {'kind': 'read_only_current_matrix_storage_inventory_not_final_snapshot_or_audit',
              'placement_reference_frame': 'ordinary_observation_20261021_0652.json', 'hosts': observations,
              'A100_import_capacity_context': {'current_remote_15_container_apparent_bytes': incoming,
                  'current_A100_free_bytes': free, 'required_training_reserve_bytes': 2*1024**3,
                  'margin_after_current_remote_bytes_and_reserve': free-incoming-2*1024**3,
                  'future_final_checkpoint_and_log_growth_not_in_this_margin': True,
                  'per_transfer_existing_collector_free_space_guard_unchanged': True},
              'notice': 'Read-only size metadata, not a sealed final manifest or scientific audit. Live logs/checkpoints can change during and after observation. No files were moved, linked, deleted, or modified, and no learner/source/config/GPU state was changed.'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
