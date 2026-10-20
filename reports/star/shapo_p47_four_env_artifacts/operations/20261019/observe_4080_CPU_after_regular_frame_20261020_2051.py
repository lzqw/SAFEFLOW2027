#!/usr/bin/env python3
"""One read-only CPU-placement preflight after the existing20:51 native frame."""
import json
import shlex
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATE = BASE / '4080_CPU_placement_preflight_20261020_2051.json'
ROOT = '/home/carla/SAFEFLOW2027_4080'
BOOT = '3b362c03-4ed3-4e03-aea4-cf7339356026'
MINIMUM_FRAME = 1792500706.5858595


def save(data):
    data['updated_at_unix'] = time.time()
    part = STATE.with_suffix('.tmp')
    part.write_text(json.dumps(data, indent=2) + '\n')
    part.replace(STATE)
    print(json.dumps({'phase': data['phase'], 'updated_at_unix': data['updated_at_unix']}), flush=True)


def main():
    assert not STATE.exists(), 'Inspect the original one-shot probe; never duplicate'
    data = {'phase': 'waiting_for_existing_regular_2051_frame',
            'minimum_frame_unix': MINIMUM_FRAME, 'read_only_remote_probe': True,
            'training_or_foreign_process_signals': 0, 'new_training_cells': 0,
            'scientific_evaluation_rollouts': 0}
    save(data)
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        raw = json.loads((BASE / 'gpu_capacity_state.json').read_text())
        host = raw.get('hosts', {}).get('4080', {})
        frame = host.get('observed_at_unix', 0)
        if frame >= MINIMUM_FRAME and 0 <= time.time() - frame <= 2100 and host.get('status') == 'observed':
            break
        time.sleep(3)
    else:
        data['phase'] = 'no_fresh_regular_frame_no_remote_probe_performed'
        save(data)
        return
    expected = host.get('training', [])
    sources = [w for w in raw.get('hosts', {}).get('5080', {}).get('training', [])
               if w['job'] in ('ant-2028', 'ant-2029')]
    code = f'''
import importlib.metadata,json,os,shutil,subprocess,time
from pathlib import Path
root=Path({ROOT!r})
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={BOOT!r}
expected=json.loads({json.dumps(expected)!r})
models=[]
for line in Path('/proc/cpuinfo').read_text().splitlines():
 if line.startswith('model name'):
  model=line.split(':',1)[1].strip()
  if model not in models:models.append(model)
cores=len(os.sched_getaffinity(0));load=os.getloadavg()[0]
available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
def cpu_sample():
 values=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
 return sum(values),values[3]
a=cpu_sample();started=time.monotonic();time.sleep(1);b=cpu_sample()
idle=(b[1]-a[1])/(b[0]-a[0]);seconds=time.monotonic()-started
tmux=int(subprocess.check_output(['tmux','display-message','-p','#{{pid}}'],text=True).strip())
quota_root=Path('/sys/fs/cgroup').resolve();quotas=[];checked=[]
for pid in [os.getpid(),tmux]+[w['pid'] for w in expected]:
 p=Path('/proc')/str(pid)
 if pid in [w['pid'] for w in expected]:
  w=next(w for w in expected if w['pid']==pid)
  assert int((p/'stat').read_text().rsplit(') ',1)[1].split()[19])==w['start_ticks']
  args=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(bytes([0])) if v]
  assert args[args.index('--config')+1]==str(root/'runs/shapo_p47_four_env/runs'/w['job']/'resolved_formal_config.yaml')
 entry=next((v for v in (p/'cgroup').read_text().splitlines() if v.startswith('0::')),None)
 assert entry is not None,'CPU quota policy unproven'
 q=(quota_root/entry.split('::',1)[1].lstrip('/')).resolve()
 assert q==quota_root or quota_root in q.parents
 while True:
  file=q/'cpu.max'
  if file.exists():
   values=file.read_text().split()
   if values[0]!='max':quotas.append(int(values[0])/int(values[1]))
  if q==quota_root:break
  q=q.parent
 checked.append(pid)
quota=min(quotas) if quotas else None
free=shutil.disk_usage(root).free
targets={{}}
for job in ['ant-2028','ant-2029']:
 d=root/'runs/shapo_p47_four_env/runs'/job
 targets[job]={{'canonical_container_exists':d.exists(),'canonical_container_is_symlink':d.is_symlink()}}
source_commit=subprocess.check_output(['git','-C',str(root/'code'),'rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C',str(root/'code'),'status','--porcelain'],text=True)
eligible=(cores>=24 and load<cores*.5 and idle>=.4 and available>=8*1024**3
          and free>=5*1024**3 and (quota is None or quota>=8)
          and source_commit=='62ca7399357c10645a955dcbead7ed2153a82c69' and not dirty)
print(json.dumps({{'observed_at_unix':time.time(),'boot_id':{BOOT!r},'cpu_models':models,
 'affinity_logical_CPUs':cores,'load_1m':load,'actual_cpu_idle_fraction':idle,
 'CPU_idle_sample_seconds':seconds,'memory_available_bytes':available,'project_free_bytes':free,
 'minimum_cpu_quota_cores_or_null':quota,'quota_checked_pids':checked,
 'quota_scope':'observer_tmux_existing_target_learners_and_ancestors',
 'runtime_versions':{{p:importlib.metadata.version(p) for p in ['torch','gymnasium','safety-gymnasium','mujoco','numpy']}},
 'frozen_source_commit':source_commit,'frozen_source_dirty':bool(dirty),
 'target_containers':targets,'CPU_headroom_eligible_under_existing_conservative_policy':eligible,
 'does_not_authorize_or_start_a_handoff':True,'GPU_resources_not_claimed':True}}))
'''
    command = ('env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '
               + ROOT + '/env/bin/python -c ' + shlex.quote(code))
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '4080', command],
                                text=True, capture_output=True, check=True, timeout=90)
        proof = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        data.update(phase='read_only_CPU_probe_unproven_no_training_action',
                    error=str(error), stderr_tail=getattr(error, 'stderr', '')[-1500:])
        save(data)
        return
    data.update(phase='read_only_CPU_placement_preflight_completed',
                regular_native_frame_unix=frame, source_existing_CPU_workers=sources,
                target_existing_workers=expected, target_CPU_proof=proof,
                next_action='Inspect evidence before preparing any new immutable handoff/accounting hook; no task has moved')
    save(data)


if __name__ == '__main__':
    main()
