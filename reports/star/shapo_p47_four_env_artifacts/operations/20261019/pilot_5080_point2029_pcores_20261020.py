#!/usr/bin/env python3
"""Reversible main-thread P-core affinity pilot; never restart or modify trainers."""
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

OPS = Path(__file__).resolve().parent
RECEIPT = OPS / 'point2029_pcore_affinity_pilot_20261020.json'
ROOT = '/home/autodrive/LZQW/SAFEFLOW2027'
BOOT = '5f46bb0c-b4ec-4802-8f6e-01ac611c0f2f'
WORKERS = [('point-goal-2029', 28355, 1928788), ('point-button-2029', 31275, 1959385)]


def rpc(code):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                             '5080', 'python3 -c ' + shlex.quote(code)],
                            capture_output=True, text=True, timeout=60, check=True)
    return json.loads(result.stdout)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--observe'
    assert mode in ('--apply', '--observe', '--restore')
    if mode == '--apply':
        assert not RECEIPT.exists(), 'Prior pilot exists: observe or restore; never repeat blindly'
        saved = {}
    else:
        saved = json.loads(RECEIPT.read_text())
    code = '''
import csv,json,os,time
from pathlib import Path
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
def identity(job,pid,ticks):
 p=Path('/proc')/str(pid)
 a=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\\0') if v]
 assert int((p/'stat').read_text().rsplit(') ',1)[1].split()[19])==ticks
 assert 'scripts/star/train_star.py' in a
 assert a[a.index('--config')+1]==ROOT+'/runs/shapo_p47_four_env/runs/'+job+'/resolved_formal_config.yaml'
 assert a[a.index('--device')+1]=='cuda:0'
 return p
def progress(p):
 files=[]
 for fd in (p/'fd').iterdir():
  try:
   f=Path(os.readlink(fd))
   if f.name=='progress.csv' and str(f).startswith(ROOT+'/runs/') and f.is_file(): files.append(f)
  except OSError: pass
 files=list(set(files)); assert len(files)==1
 with files[0].open('rb') as f:
  head=f.readline(); f.seek(0,2); offset=max(len(head),f.tell()-65536); f.seek(offset); tail=f.read()
 if offset>len(head): tail=tail.partition(b'\\n')[2]
 keys=next(csv.reader([head.decode()])); rows=[]
 for raw in tail.split(b'\\n')[:-1]:
  if not raw.strip():continue
  vals=next(csv.reader([raw.decode()]))
  if len(vals)!=len(keys):continue
  r=dict(zip(keys,vals))
  if r.get('TotalEnvSteps'):
   rows.append({k:r[k] for k in ['TotalEnvSteps','Time/FPS','Time/Total','Time/Epoch','Train/CumulativeCost'] if k in r})
 return {'csv':str(files[0]),'mtime_unix':files[0].stat().st_mtime,'recent_rows':rows[-4:]}
target=list(range(16))
assert set(target)<=os.sched_getaffinity(0)
topology=[]
for cpu in target:
 base=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))
 siblings=(base/'topology/thread_siblings_list').read_text().strip()
 frequency=int((base/'cpufreq/cpuinfo_max_freq').read_text())
 assert frequency>=5500000, 'Topology changed: target is not a high-frequency P core'
 topology.append({'cpu':cpu,'thread_siblings':siblings,'max_frequency_khz':frequency})
before=[]
for job,pid,ticks in WORKERS:
 p=identity(job,pid,ticks)
 before.append({'job':job,'pid':pid,'start_ticks':ticks,'old_affinity':sorted(os.sched_getaffinity(pid)),
                'progress':progress(p)})
changed=[]
try:
 for w in before:
  identity(w['job'],w['pid'],w['start_ticks'])
  if MODE=='--apply': desired=target
  elif MODE=='--restore': desired=next(x['old_affinity'] for x in SAVED['before'] if x['pid']==w['pid'])
  else: continue
  os.sched_setaffinity(w['pid'],desired); changed.append(w)
  assert sorted(os.sched_getaffinity(w['pid']))==sorted(desired)
except BaseException:
 for w in changed:
  identity(w['job'],w['pid'],w['start_ticks']);os.sched_setaffinity(w['pid'],w['old_affinity'])
 raise
after=[]
for w in before:
 p=identity(w['job'],w['pid'],w['start_ticks'])
 after.append({'job':w['job'],'pid':w['pid'],'start_ticks':w['start_ticks'],
               'affinity':sorted(os.sched_getaffinity(w['pid'])),'progress':progress(p)})
print(json.dumps({'observed_at_unix':time.time(),'boot_id':BOOT,'mode':MODE,
                  'main_thread_only':True,'topology':topology,'before':before,'after':after,
                  'trainers_restarted':False,'scientific_parameters_changed':False,
                  'extra_environment_steps':0,'other_projects_modified':False}))
'''
    prefix = '\n'.join(f'{k}={v!r}' for k, v in {
        'ROOT': ROOT, 'BOOT': BOOT, 'WORKERS': WORKERS, 'MODE': mode, 'SAVED': saved}.items())
    result = rpc(prefix + '\n' + code)
    if mode == '--apply':
        RECEIPT.write_text(json.dumps(result, indent=2) + '\n')
    else:
        path = OPS / ('point2029_pcore_affinity_pilot_' + mode[2:] + '_' + str(int(time.time())) + '.json')
        path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
