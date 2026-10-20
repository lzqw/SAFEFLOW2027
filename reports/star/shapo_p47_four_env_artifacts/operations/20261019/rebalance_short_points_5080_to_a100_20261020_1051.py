#!/usr/bin/env python3
"""Rebalance two shorter Point cells into newly stable A100 headroom.

Existing10M budgets and frozen source remain unchanged. One source at a time is
held in memory until a new target epoch. No MPS or unrelated-process actions.
"""
import hashlib
import importlib.util
import json
import shlex
import subprocess
import time
from pathlib import Path

BASE=Path(__file__).resolve().parent
SRC,DST='5080','tjucta-a100'
ROOTS={SRC:'/home/autodrive/LZQW/SAFEFLOW2027',DST:'/home/lzqw/SAFEFLOW2027_A100'}
CODES={SRC:ROOTS[SRC]+'/code',DST:ROOTS[DST]+'/code/training-62ca739'}
SOURCE='62ca7399357c10645a955dcbead7ed2153a82c69'
SIDE='a100_return_balance_20261020_1051'
HELPER=BASE/'rebalance_goal2030_a100_to_5080_20261020_0621.py'
HELPER_SHA='690fb3391274fe30f08aabb1b91f8df246276547bac9b3bd21a3f3a2c4158323'
CONTROLLER=BASE/(SIDE+'_controller_state.json')
SPECS=[('point-button-2030',2030,0),('point-button-2031',2031,1)]


def save(path,value):
    value['updated_at_unix']=time.time()
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
    print(json.dumps({'file':path.name,'phase':value.get('phase'),'time':value['updated_at_unix']}),flush=True)


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def helpers(job):
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest()==HELPER_SHA
    h=load_module(HELPER,'verified_short_Point_RPC_'+job.replace('-','_'))
    h.SRC,h.DST,h.JOB=SRC,DST,job;h.ROOTS=ROOTS;h.CODES=CODES
    return h


def target_capacity(h,plan,copy_complete=False):
    return json.loads(h.py(DST,f'''
import csv,importlib.metadata,json,os,shutil,subprocess,time
from pathlib import Path
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={plan['target_boot_id']!r}
assert importlib.metadata.version('torch')=='2.7.1+cu118'
def query(flag):
 return list(csv.reader(subprocess.check_output(['nvidia-smi',flag,'--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()))
row=next([v.strip() for v in r] for r in query('--query-gpu=index,uuid,memory.free,utilization.gpu') if int(r[0])=={plan['target_gpu_index']})
assert row[1]=={plan['target_gpu_uuid']!r} and int(row[2])>=16384 and int(row[3])<=85
pids=[]
for uuid,pid,_ in query('--query-compute-apps=gpu_uuid,pid,used_memory'):
 if uuid.strip()!=row[1]:continue
 pid=int(pid.strip());a=[v.decode(errors='replace') for v in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\\0') if v]
 assert 'scripts/star/train_star.py' in a,'Foreign GPU process; keep source running'
 assert a[a.index('--config')+1].startswith({ROOTS[DST]+'/runs/'!r})
 pids.append(pid)
assert sorted(pids)=={plan['expected_target_owned_pids']!r},'Target GPU cohort changed'
free=shutil.disk_usage({ROOTS[DST]!r}).free
assert free>={0 if copy_complete else plan['source_container_bytes']}+20*1024**3,'Preserve20GiB A100 root headroom'
cores=len(os.sched_getaffinity(0));load=os.getloadavg()[0]
available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
assert cores>=16 and load<cores*.75 and available>=8*1024**3
def sample():
 v=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]));return sum(v),v[3]
a=sample();time.sleep(1);b=sample();idle=(b[1]-a[1])/(b[0]-a[0]);assert idle>=.25
tmux=shutil.which('tmux');assert tmux
tmux_pid=int(subprocess.check_output([tmux,'display-message','-p','#{{pid}}'],text=True).strip())
# This A100 uses hybrid cgroups: CPU quotas are enforced by v1, not cpu.max.
roots={{p.resolve() for p in [Path('/sys/fs/cgroup/cpu,cpuacct'),Path('/sys/fs/cgroup/cpu')] if (p/'cpu.cfs_quota_us').is_file()}}
assert len(roots)==1,'Unknown CPU quota hierarchy'
quota_root=roots.pop();limits=[];checked=[]
for pid in [os.getpid(),tmux_pid]+pids:
 entry=next(line for line in Path('/proc',str(pid),'cgroup').read_text().splitlines() if 'cpu' in line.split(':')[1].split(','))
 p=(quota_root/entry.split(':',2)[2].lstrip('/')).resolve()
 assert p==quota_root or quota_root in p.parents
 while True:
  q=p/'cpu.cfs_quota_us';period=p/'cpu.cfs_period_us'
  assert q.is_file() and period.is_file(),'CPU quota scope unproven'
  quota=int(q.read_text());interval=int(period.read_text());assert interval>0
  checked.append({{'pid':pid,'path':str(p),'quota':quota,'period':interval}})
  if quota>=0:limits.append(quota/interval)
  if p==quota_root:break
  p=p.parent
minimum=min(limits) if limits else None;assert minimum is None or minimum>=8
print(json.dumps({{'physical_gpu_index':{plan['target_gpu_index']},'gpu_uuid':row[1],
 'free_mib':int(row[2]),'utilization_percent':int(row[3]),'foreign_gpu_pids':[],
 'owned_gpu_pids':sorted(pids),'project_free_bytes':free,'reserved_root_bytes':20*1024**3,
 'incoming_container_bytes':{plan['source_container_bytes']},'copy_complete':{copy_complete!r},'cpu_affinity_count':cores,
 'load_1m':load,'cpu_idle_fraction':idle,'memory_available_bytes':available,
 'minimum_cpu_quota_cores_or_null':minimum,'cpu_quota_hierarchy':'hybrid_v1_cpu',
 'cpu_quota_scope':'observer_tmux_existing_GPU_workers_and_all_ancestors',
 'cpu_quota_records':checked,'tmux_server_pid':tmux_pid,'tmux_executable':tmux}}))
''',True))


def source_capacity(h,expected):
    return json.loads(h.py(SRC,f'''
import csv,json,subprocess
from pathlib import Path
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={expected['boot_id']!r}
owned=[]
for uuid,pid in csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()):
 if uuid.strip()!='GPU-ac315b09-0bb7-2072-efd1-cdfc5fae96a5':continue
 a=[v.decode(errors='replace') for v in Path('/proc',pid.strip(),'cmdline').read_bytes().split(b'\\0') if v]
 assert 'scripts/star/train_star.py' in a
 assert a[a.index('--config')+1].startswith({ROOTS[SRC]+'/runs/'!r})
 owned.append(int(pid.strip()))
assert {expected['pid']} in owned
print(json.dumps({{'gpu_uuid':'GPU-ac315b09-0bb7-2072-efd1-cdfc5fae96a5','owned_gpu_pids':owned,'foreign_gpu_pids':[]}}))
'''))


def run_cell(job,seed,index,before,now):
    h=helpers(job);prefix=job+'_'+SIDE
    state_path=BASE/(prefix+'_state.json');plan_path=BASE/(prefix+'_plan.json')
    assert not state_path.exists() and not plan_path.exists(),'Never duplicate an existing transition'
    old_gpu=next(g for g in before['hosts'][DST]['gpus'] if g['index']==index)
    gpu=next(g for g in now['hosts'][DST]['gpus'] if g['index']==index)
    assert gpu['uuid']==old_gpu['uuid'] and not old_gpu['foreign_pids'] and not gpu['foreign_pids']
    assert gpu['status']=='available_for_rebalance'
    assert now['observed_at_unix']-before['observed_at_unix']>=1800
    assert 0<=time.time()-now['observed_at_unix']<7200
    expected=next(w for w in now['hosts'][SRC]['training'] if w['job']==job)
    observed=h.status(SRC);assert len(observed['workers'])==1
    source={**observed['workers'][0],'boot_id':observed['boot_id']}
    assert all(source[k]==expected[k] for k in ('pid','start_ticks','boot_id','device'))
    assert source['device']=='cuda:0' and source['cuda_visible_devices']=='0'
    for flag,value in {'--seed':str(seed),'--steps':'10000000','--vector-envs':'1',
                       '--batch-size':'256','--replay-size':'400000','--steps-per-epoch':'5000'}.items():
        assert source['args'][source['args'].index(flag)+1]==value
    assert not h.status(DST)['workers']
    src,dst=h.container(SRC),h.container(DST)
    h.ssh(DST,'test ! -e '+shlex.quote(dst))
    for host in (SRC,DST):
        assert h.ssh(host,'git -C '+shlex.quote(CODES[host])+' rev-parse HEAD').strip()==SOURCE
        assert not h.ssh(host,'git -C '+shlex.quote(CODES[host])+' status --porcelain').strip()
    size=int(h.ssh(SRC,'du -sb '+shlex.quote(src)).split()[0])
    target_owned=sorted(p['pid'] for p in gpu['processes'] if p['role']=='safeflow')
    assert len(target_owned)==1
    plan={'job':job,'seed':seed,'source_host':SRC,'target_host':DST,'source_commit':SOURCE,
          'source_identity':source,'source_container_bytes':size,'total_budget_steps':10000000,
          'source_ordinary_observation':'capacity_progress_observation_20261020_1051.json',
          'previous_target_observation':'capacity_progress_observation_20261020_1021.json',
          'target_gpu_index':index,'target_gpu_uuid':gpu['uuid'],
          'target_boot_id':now['hosts'][DST]['training'][0]['boot_id'],
          'expected_target_owned_pids':target_owned,'two_ordinary_frames_qualified':True,
          'new_experiment_cells':0,'scientific_parameters_changed':False,'other_projects_paused':False,
          'choice':'shorter_remaining_budget_Point_cell_to_reduce5080_long_tail_contention'}
    plan['source_capacity_before_pause']=source_capacity(h,source)
    plan['target_capacity_before_pause']=target_capacity(h,plan)
    save(plan_path,plan)
    state={'job':job,'seed':seed,'phase':'preparing','plan':plan,'source_identity':source,
           'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'source_commit':SOURCE,'total_budget_steps':10000000,'scientific_parameters_changed':False,
           'other_projects_paused':False,'source_paused':False,'source_retired':False,
           'runtime_change':{'from_torch':'2.7.1+cu128','to_torch':'2.7.1+cu118',
                             'source_physical_GPU':'RTX5080','target_physical_GPU':'A100',
                             'target_physical_gpu_index':index,'bitwise_equivalence_claimed':False}}
    save(state_path,state);paused=False;launched=False;session='safeflow_'+prefix.replace('-','_')
    try:
        h.signal_source(source,'SIGSTOP');paused=True
        last=next(r for r in h.status(SRC)['progress'] if r['csv']==expected['progress_csv'])
        full=json.loads(h.py(SRC,f'''
import hashlib,json,time,torch
from pathlib import Path
assert str(torch.__version__)=='2.7.1+cu128'
for _ in range(50):
 if Path('/proc/{source['pid']}/status').read_text().split('State:',1)[1].split()[0]=='T':break
 time.sleep(.1)
else:raise RuntimeError('Original pause unproven')
d=Path({src!r});p=Path({last['csv']!r}).parent/'checkpoints/latest_full.pt'
p.resolve().relative_to(d.resolve());x=torch.load(p,map_location='cpu',weights_only=False)
assert x.get('rng') and x.get('replay') is not None and x.get('optimizers')
assert all(torch.isfinite(v).all() for v in x['model'].values() if torch.is_tensor(v))
n=int(x['counters']['env_steps']);c=float(x['counters']['cumulative_training_cost'])
assert 0<n<10000000 and 0<={last['steps']}-n<50000 and {last['cost']}>=c
print(json.dumps({{'source_checkpoint':str(p),'relative_checkpoint':str(p.relative_to(d)),
 'checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'resume_steps':n,
 'resume_cumulative_training_cost':c,'full_replay_rng_optimizers_finite_model':True}}))
''',True))
        state.update(full,phase='copying_paused_source',source_paused=True,
            last_logged_source_steps=last['steps'],last_logged_source_cost=last['cost'],
            known_repeated_steps_lower_bound=last['steps']-full['resume_steps'],
            known_extra_training_cost_lower_bound=last['cost']-full['resume_cumulative_training_cost'],
            unlogged_source_tail_unknown_not_zero=True)
        save(state_path,state)
        receive=('set -e; test ! -e '+shlex.quote(dst)+'; point_stage=$(mktemp -d '
                 +shlex.quote(ROOTS[DST]+'/runs/.a100-return-point.XXXXXX')+'); '
                 'tar -xzf - --no-same-owner -C "$point_stage"; test ! -e '+shlex.quote(dst)
                 +'; mv "$point_stage/'+job+'" '+shlex.quote(dst)+'; rmdir "$point_stage"')
        send='tar --format=posix -czf - -C '+shlex.quote(str(Path(src).parent))+' -- '+shlex.quote(job)
        with subprocess.Popen([*h.OPTIONS,SRC,send],stdout=subprocess.PIPE) as producer:
            try:
                receiver=subprocess.run([*h.OPTIONS,DST,receive],stdin=producer.stdout,timeout=3600)
            finally:
                producer.stdout.close()
                try:producer.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    # This is our transfer-only SSH child, never a trainer.
                    producer.terminate();producer.wait(timeout=30)
            assert producer.returncode==0 and receiver.returncode==0
        target_checkpoint=dst+'/'+full['relative_checkpoint']
        relocation={'format':'shapo-artifact-relocation-v1','job':job,'original_container':src,
                    'original_host':SRC,'target_host':DST,'scientific_protocol_changed':False,
                    'original_evidence_modified':False}
        h.py(DST,f'''
import hashlib,time
from pathlib import Path
d=Path({dst!r});p=Path({target_checkpoint!r})
assert hashlib.sha256(p.read_bytes()).hexdigest()=={full['checkpoint_sha256']!r}
for name in ['exit_code.txt','receipt.json','artifact_relocation.json']:
 p=d/name
 if p.exists():p.rename(d/(name+'.before_{SIDE}_'+str(time.time_ns())))
(d/'artifact_relocation.json').write_text({json.dumps(relocation,indent=2)+chr(10)!r})
''')
        prior={r['csv'] for r in h.status(DST)['progress']}
        shell=f'''#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES={index} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={CODES[DST]}
cd {CODES[DST]}
set +e
{ROOTS[DST]}/env/bin/python scripts/star/train_star.py --config {dst}/resolved_formal_config.yaml --seed {seed} --steps 10000000 --device cuda:0 --vector-envs 1 --batch-size 256 --replay-size 400000 --steps-per-epoch 5000 --log-dir {dst}/training --result {dst}/receipt.json --no-tensorboard --resume {shlex.quote(target_checkpoint)} >{dst}/train_{SIDE}.log 2>&1
status=$?
printf '%s\\n' "$status" >{dst}/exit_code.txt
exit "$status"
'''
        launcher=ROOTS[DST]+'/deploy/resume_'+prefix+'.sh'
        h.py(DST,f'from pathlib import Path; Path({launcher!r}).write_text({shell!r})')
        state['target_capacity_at_launch']=target_capacity(h,plan,copy_complete=True)
        launched=True;state.update(phase='target_starting_source_paused',target_launch_requested=True,
                                   target_checkpoint=target_checkpoint,target_session=session)
        save(state_path,state)
        h.ssh(DST,'tmux new-session -d -s '+shlex.quote(session)+' '+shlex.quote('bash '+launcher))
        deadline=time.monotonic()+3600
        while time.monotonic()<deadline:
            target=h.status(DST)
            new=[r for r in target['progress'] if r['csv'] not in prior and r['steps']>=full['resume_steps']+5000]
            if new and len(target['workers'])==1:break
            marker=h.ssh(DST,'test ! -e '+shlex.quote(dst+'/exit_code.txt')+' || cat '+shlex.quote(dst+'/exit_code.txt')).strip()
            if not target['workers'] and marker:raise RuntimeError('Target terminal before first new epoch')
            time.sleep(30)
        else:raise RuntimeError('No new epoch yet; inspect same live target, never duplicate')
        target_id={**target['workers'][0],'boot_id':target['boot_id']};first=new[-1]
        assert target_id['device']=='cuda:0' and target_id['cuda_visible_devices']==str(index)
        proof=json.loads(h.py(DST,f'''
import csv,json,subprocess
rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())
assert any(uuid.strip()=={plan['target_gpu_uuid']!r} and int(pid.strip())=={target_id['pid']} for uuid,pid in rows)
print(json.dumps({{'pid':{target_id['pid']},'gpu_uuid':{plan['target_gpu_uuid']!r},'verified':True}}))
'''))
        state.update(phase='first_target_epoch_verified_retiring_source',target_identity=target_id,
                     first_target_epoch=first,target_GPU_identity_proof=proof)
        save(state_path,state)
        h.signal_source(source,'SIGTERM');h.signal_source(source,'SIGCONT');paused=False
        for _ in range(60):
            if not h.status(SRC)['workers']:break
            time.sleep(1)
        else:raise RuntimeError('Original source retirement unproven')
        state['source_retired']=True
        archive=ROOTS[SRC]+'/recovery_backups/retired_gpu_workers/'+prefix+'_'+str(time.time_ns())
        h.py(SRC,f'''
import time
from pathlib import Path
d=Path({src!r});archive=Path({archive!r})
for _ in range(120):
 if (d/'exit_code.txt').is_file():break
 time.sleep(.5)
else:raise RuntimeError('Original source wrapper marker not observed')
archive.parent.mkdir(parents=True,exist_ok=True)
assert not archive.exists() and d.stat().st_dev==archive.parent.stat().st_dev
d.rename(archive)
''')
        observer=load_module(BASE/'watch_gpu_capacity.py','A100_return_epoch_observer')
        native=observer.probe(DST);worker=next(w for w in native['training'] if w['pid']==target_id['pid'])
        assert worker['steps']>=first['steps'] and worker['progress_csv']==first['csv']
        native_path=BASE/(prefix+'_first_target_native_observation.json');save(native_path,native)
        state.update(phase='completed_first_target_epoch_source_retired',source_paused=False,
                     retired_source_archive=archive,source_canonical_path_vacated=True,
                     first_target_native_observation=native_path.name)
        save(state_path,state)
        h.py(DST,f'from pathlib import Path; Path({dst+"/handoff_"+SIDE+"_state.json"!r}).write_text({json.dumps(state,indent=2)+chr(10)!r})')
        return state
    except Exception as error:
        absent=not launched
        try:
            absent=not h.status(DST)['workers'] and h.ssh(DST,'tmux has-session -t '+shlex.quote(session)+' >/dev/null 2>&1 && echo live || echo absent').strip()=='absent'
        except (OSError,ValueError,subprocess.SubprocessError):pass
        resumed=False
        if paused and absent:h.signal_source(source,'SIGCONT');paused=False;resumed=True
        state.update(phase='needs_inspection_after_A100_return_error',source_paused=paused,
                     target_launch_requested=launched,target_absence_confirmed=absent,
                     original_source_resumed_in_memory=resumed,error=str(error),automatic_retry=False)
        save(state_path,state)
        raise


def main():
    assert not CONTROLLER.exists(),'Inspect existing controller; never duplicate'
    accounting=load_module(BASE/'a100_rebalance_accounting_20261020.py','A100_rebalance_accounting_readiness')
    assert accounting.verify_A100_return_balances(BASE)==[]
    before=json.loads((BASE/'capacity_progress_observation_20261020_1021.json').read_text())
    now=json.loads((BASE/'capacity_progress_observation_20261020_1051.json').read_text())
    state={'phase':'starting_serial_short_Point_rebalance','completed_jobs':[],'source_host':SRC,'target_host':DST}
    save(CONTROLLER,state)
    for job,seed,index in SPECS:
        run_cell(job,seed,index,before,now)
        state['completed_jobs'].append(job);save(CONTROLLER,state)
    state['phase']='two_short_Point_cells_rebalanced_to_A100';save(CONTROLLER,state)


if __name__=='__main__':main()
