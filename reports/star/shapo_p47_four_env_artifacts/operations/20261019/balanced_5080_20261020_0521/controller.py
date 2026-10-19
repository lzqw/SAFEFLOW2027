#!/usr/bin/env python3
"""Balance one existing10M Point cell within our own5080 GPU cohort; no new experiment."""
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
JOB = 'point-button-2028'
SIDE = 'balanced_5080_20261020_0521'
SRC, DST = 'tjucta-a100', '5080'
ROOTS = {SRC: '/home/lzqw/SAFEFLOW2027_A100', DST: '/home/autodrive/LZQW/SAFEFLOW2027'}
CODES = {SRC: ROOTS[SRC]+'/code/training-62ca739', DST: ROOTS[DST]+'/code'}
PLAN = BASE/'point_button2028_5080_balanced_20261020_0521.json'
STATE = BASE/(JOB+'_'+SIDE+'_state.json')
RECEIPT = BASE/(JOB+'_'+SIDE+'_receipt.json')
SESSION = 'safeflow_point_button_2028_balanced_5080_20261020_0521'
OPTIONS = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',
           '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']


def ssh(host, command, timeout=120):
    try:
        return subprocess.run([*OPTIONS,host,command],capture_output=True,text=True,
                              check=True,timeout=timeout).stdout
    except subprocess.CalledProcessError as error:
        print(host+': '+(error.stderr or '')[-4000:],flush=True)
        raise


def py(host, code, environment=False):
    python = ROOTS[host]+'/env/bin/python' if environment else 'python3'
    return ssh(host, 'env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
               'MKL_NUM_THREADS=1 '+python+' -c '+shlex.quote(code))


def save(path, data):
    data['updated_at_unix'] = time.time()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def container(host):
    return ROOTS[host]+'/runs/shapo_p47_four_env/runs/'+JOB


def status(host):
    return json.loads(py(host,f'''
import csv,json
from pathlib import Path
d=Path({container(host)!r});workers=[];progress=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=[x.decode(errors='replace') for x in p.read_bytes().split(b'\\0') if x]
  if 'scripts/star/train_star.py' not in a or a[a.index('--config')+1]!=str(d/'resolved_formal_config.yaml'):continue
  env=dict(x.split(b'=',1) for x in (p.parent/'environ').read_bytes().split(b'\\0') if b'=' in x)
  fields=(p.parent/'stat').read_text().rsplit(') ',1)[1].split()
  workers.append({{'pid':int(p.parent.name),'start_ticks':int(fields[19]),'state':fields[0],
    'device':a[a.index('--device')+1],'cuda_visible_devices':env.get(b'CUDA_VISIBLE_DEVICES',b'').decode(),
    'args':a}})
 except(OSError,ValueError,IndexError):pass
for p in d.glob('training/**/progress.csv'):
 row=None
 for row in csv.DictReader(p.open()):pass
 if row and row.get('TotalEnvSteps'):
  progress.append({{'csv':str(p),'mtime_ns':p.stat().st_mtime_ns,'steps':int(float(row['TotalEnvSteps'])),
   'cost':float(row['Train/CumulativeCost']),'fps':float(row['Time/FPS'])}})
print(json.dumps({{'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 'workers':workers,'progress':sorted(progress,key=lambda r:r['mtime_ns'])}}))
'''))


def signal_source(identity, name):
    py(SRC,f'''
import os,signal
from pathlib import Path
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={identity['boot_id']!r}
p=Path('/proc/{identity['pid']}')
a=p.joinpath('cmdline').read_bytes().split(b'\\0')
assert {str(container(SRC)+'/resolved_formal_config.yaml').encode()!r} in a
assert int(p.joinpath('stat').read_text().rsplit(') ',1)[1].split()[19])=={identity['start_ticks']}
os.kill({identity['pid']},signal.{name})
''')


def source_shared_capacity(plan, identity):
    return json.loads(py(SRC,f'''
import csv,json,subprocess
from pathlib import Path
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={identity['boot_id']!r}
rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())
owned=[];foreign=[]
for uuid,pid in rows:
 if uuid.strip()!={plan['source_gpu_uuid']!r}:continue
 pid=int(pid.strip())
 try:
  a=[v.decode(errors='replace') for v in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\\0') if v]
  own=('scripts/star/train_star.py' in a and a[a.index('--config')+1].startswith({ROOTS[SRC]+'/runs/'!r}))
 except(OSError,ValueError,IndexError):own=False
 (owned if own else foreign).append(pid)
assert {identity['pid']} in owned,'Source GPU identity changed; leave source untouched'
assert foreign,'Source GPU became free of other projects; leave source running there'
print(json.dumps({{'uuid':{plan['source_gpu_uuid']!r},'owned_gpu_pids':owned,'foreign_gpu_pids':foreign}}))
'''))


def capacity(plan):
    return json.loads(py(DST,f'''
import csv,importlib.metadata,json,os,shutil,subprocess
from pathlib import Path
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={plan['expected_target_boot_id']!r}
assert importlib.metadata.version('torch')=='2.7.1+cu128'
def query(flag):
 return list(csv.reader(subprocess.check_output(['nvidia-smi',flag,'--format=csv,noheader,nounits'],text=True,timeout=20).splitlines()))
row=next([v.strip() for v in r] for r in query('--query-gpu=index,uuid,memory.free,utilization.gpu') if int(r[0])==0)
assert row[1]=={plan['rebalance_qualification']['uuid']!r}
assert int(row[2])>=8192 and int(row[3])<=100,'GPU memory capacity changed; do not pause source'
pids=[]
for uuid,pid,_ in query('--query-compute-apps=gpu_uuid,pid,used_memory'):
 if uuid.strip()!=row[1]:continue
 args=[x.decode(errors='replace') for x in Path('/proc',pid.strip(),'cmdline').read_bytes().split(b'\\0') if x]
 assert 'scripts/star/train_star.py' in args,'Foreign process on targetGPU'
 Path(args[args.index('--config')+1]).resolve().relative_to(Path({ROOTS[DST]!r}).resolve())
 pids.append(int(pid.strip()))
assert sorted(pids)==sorted({plan['rebalance_qualification']['owned_gpu_pids']!r}),'Owned target cohort changed; do not pause source'
free=shutil.disk_usage({ROOTS[DST]!r}).free
cores=len(os.sched_getaffinity(0));load=os.getloadavg()[0]
available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
assert cores>=16 and load<cores*.35 and available>=8*1024**3,'Target CPU/memory headroom insufficient'
assert free>=5*1024**3
print(json.dumps({{'physical_gpu_index':0,'uuid':row[1],'free_mib':int(row[2]),
 'utilization_percent':int(row[3]),'foreign_gpu_processes':0,'owned_gpu_pids':pids,
 'project_free_bytes':free,'cpu_affinity_count':cores,'load_1m':load,
 'memory_available_bytes':available}}))
''',True))


def run():
    assert not STATE.exists(), 'Inspect existing transaction, never duplicate'
    plan=json.loads(PLAN.read_text());q=plan['rebalance_qualification']
    assert plan['job']==JOB and plan['source_commit']==SOURCE and plan['total_training_budget']==10000000
    assert q['kind']=='existing_cohort_makespan_balance' and not q['foreign_pids']
    assert q['observed_at_unix']-q['previous_observed_at_unix']>=1800
    assert len(q['owned_gpu_pids'])==5
    assert q['source_last_epoch_fps']*2<=q['target_minimum_last_epoch_fps']
    assert plan['additional_experiment_cells']==0
    assert 0<=time.time()-q['observed_at_unix']<2100
    src,dst=container(SRC),container(DST)
    observed=status(SRC);expected=plan['expected_source_identity']
    assert len(observed['workers'])==1
    identity={**observed['workers'][0],'boot_id':observed['boot_id']}
    assert all(identity[key]==expected[key] for key in ('pid','start_ticks','boot_id','device'))
    assert identity['device']=='cuda:0' and identity['cuda_visible_devices']=='0'
    args=identity['args']
    for flag,value in {'--seed':'2028','--steps':'10000000','--vector-envs':'1',
                       '--batch-size':'256','--replay-size':'400000','--steps-per-epoch':'5000'}.items():
        assert args[args.index(flag)+1]==value
    assert not status(DST)['workers']
    ssh(DST,'test ! -e '+shlex.quote(dst))
    for host in (SRC,DST):
        assert ssh(host,'git -C '+shlex.quote(CODES[host])+' rev-parse HEAD').strip()==SOURCE
        assert not ssh(host,'git -C '+shlex.quote(CODES[host])+' status --porcelain').strip()
    initial_source_capacity=source_shared_capacity(plan,identity)
    initial_capacity=capacity(plan)
    state={'job':JOB,'seed':2028,'phase':'preparing','source_host':SRC,'target_host':DST,
           'source_identity':identity,'source_commit':SOURCE,'plan':plan,
           'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'scientific_parameters_changed':False,'additional_evaluation':False,
           'total_training_budget':10000000,'target_capacity_at_launch':initial_capacity,
           'source_shared_capacity_before_pause':initial_source_capacity,
           'runtime_change':{'from_torch':'2.7.1+cu118','to_torch':'2.7.1+cu128',
                             'target_physical_gpu_index':0,'cuda_build_changed':True,
                             'bitwise_equivalence_claimed':False}}
    save(STATE,state)
    paused=retired=launch_requested=False
    try:
        signal_source(identity,'SIGSTOP');paused=True
        last=status(SRC)['progress'][-1]
        checkpoint=json.loads(py(SRC,f'''
import hashlib,json,time,torch
from pathlib import Path
torch.set_num_threads(1)
assert str(torch.__version__)=='2.7.1+cu118'
for _ in range(50):
 if Path('/proc/{identity['pid']}/status').read_text().split('State:',1)[1].split()[0]=='T':break
 time.sleep(.1)
else:raise RuntimeError('Owned source pause not confirmed')
d=Path({src!r});p=max(d.glob('training/**/checkpoints/latest_full.pt'),key=lambda p:p.stat().st_mtime_ns)
p.resolve().relative_to(d.resolve())
x=torch.load(p,map_location='cpu',weights_only=False)
assert 'rng' in x and 'replay' in x
n=int(x['counters']['env_steps']);c=float(x['counters']['cumulative_training_cost'])
assert 0< n <10000000 and 0<={last['steps']}-n<50000 and {last['cost']}>=c
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
print(json.dumps({{'source_checkpoint':str(p),'relative_checkpoint':str(p.relative_to(d)),
 'checkpoint_sha256':h.hexdigest(),'resume_steps':n,'resume_cumulative_training_cost':c,
 'full_replay_and_rng_present':True}}))
''',True))
        state.update(checkpoint,phase='copying_paused_source',source_paused=True,
            last_logged_source_steps=last['steps'],last_logged_source_cumulative_training_cost=last['cost'],
            known_repeated_environment_steps_lower_bound=last['steps']-checkpoint['resume_steps'],
            known_additional_training_cost_lower_bound=last['cost']-checkpoint['resume_cumulative_training_cost'],
            physical_repeated_steps_and_cost_unknown_not_zero=True,
            unlogged_source_tail_may_exist=True,original_evidence_preserved=True)
        save(STATE,state)
        receive=('set -e; test ! -e '+shlex.quote(dst)+'; handoff_stage=$(mktemp -d '
                 +shlex.quote(ROOTS[DST]+'/runs/.button2028-handoff.XXXXXX')+'); '
                 'tar -xzf - --no-same-owner -C "$handoff_stage"; test ! -e '+shlex.quote(dst)
                 +'; mv "$handoff_stage/'+JOB+'" '+shlex.quote(dst)+'; rmdir "$handoff_stage"')
        send='tar --format=posix -czf - -C '+shlex.quote(str(Path(src).parent))+' -- '+shlex.quote(JOB)
        with subprocess.Popen([*OPTIONS,SRC,send],stdout=subprocess.PIPE) as producer:
            try:
                result=subprocess.run([*OPTIONS,DST,receive],stdin=producer.stdout,timeout=3600)
            finally:
                producer.stdout.close()
                try:producer.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    producer.terminate()
                    try:producer.wait(timeout=15)
                    except subprocess.TimeoutExpired:producer.kill();producer.wait()
            assert producer.returncode==0 and result.returncode==0
        receipt={**state,'target_checkpoint':dst+'/'+checkpoint['relative_checkpoint']}
        save(RECEIPT,receipt)
        relocation={'format':'shapo-artifact-relocation-v1','job':JOB,'original_container':src,
                    'original_host':SRC,'target_host':DST,'scientific_protocol_changed':False,
                    'original_evidence_modified':False}
        py(DST,f'''
import hashlib,json,time
from pathlib import Path
d=Path({dst!r});p=d/{checkpoint['relative_checkpoint']!r};h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
assert h.hexdigest()=={checkpoint['checkpoint_sha256']!r}
for name in ['artifact_relocation.json','exit_code.txt','receipt.json']:
 path=d/name
 if path.exists():path.rename(d/(name+'.before_{SIDE}_'+str(time.time_ns())))
(d/'artifact_relocation.json').write_text({json.dumps(relocation,indent=2)+chr(10)!r})
(d/'handoff_{SIDE}_receipt.json').write_text({json.dumps(receipt,indent=2)+chr(10)!r})
''')
        prior={r['csv'] for r in status(DST)['progress']}
        root=ROOTS[DST];code=CODES[DST]
        shell=f'''#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={shlex.quote(code)}
cd {shlex.quote(code)}
set +e
{root}/env/bin/python scripts/star/train_star.py --config {dst}/resolved_formal_config.yaml --seed 2028 --steps 10000000 --device cuda:0 --vector-envs 1 --batch-size 256 --replay-size 400000 --steps-per-epoch 5000 --log-dir {dst}/training --result {dst}/receipt.json --no-tensorboard --resume {shlex.quote(receipt['target_checkpoint'])} >{dst}/train_{SIDE}.log 2>&1
status=$?
printf '%s\\n' "$status" > {dst}/exit_code.txt
exit "$status"
'''
        launcher=root+'/deploy/resume_'+JOB+'_'+SIDE+'.sh'
        py(DST,f'from pathlib import Path; Path({launcher!r}).write_text({shell!r})')
        state['target_capacity_at_launch']=capacity(plan)
        launch_requested=True
        ssh(DST,ROOTS[DST]+'/env/bin/tmux new-session -d -s '+shlex.quote(SESSION)+' '+shlex.quote('bash '+launcher))
        state.update(phase='target_starting_source_paused',target_session=SESSION);save(STATE,state)
        deadline=time.monotonic()+3600
        while time.monotonic()<deadline:
            target=status(DST)
            new=[r for r in target['progress'] if r['csv'] not in prior
                 and r['steps']>=checkpoint['resume_steps']+5000]
            if new and len(target['workers'])==1:break
            if not target['workers'] and ssh(DST,'test ! -e '+shlex.quote(dst+'/exit_code.txt')+' || cat '+shlex.quote(dst+'/exit_code.txt')).strip():
                raise RuntimeError('Target terminal before first epoch')
            time.sleep(30)
        else:raise RuntimeError('No first target epoch yet; inspect live target, never duplicate')
        identity_target=target['workers'][0]
        assert identity_target['device']=='cuda:0' and identity_target['cuda_visible_devices']=='0'
        py(DST,f'''
import csv,subprocess
rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())
assert any(uuid.strip()=={q['uuid']!r} and int(pid.strip())=={identity_target['pid']} for uuid,pid in rows)
''')
        state.update(phase='first_target_epoch_verified_retiring_source',target_identity=identity_target,
            first_verified_target_steps=new[-1]['steps'],first_verified_target_fps=new[-1]['fps'],
            first_verified_target_csv=new[-1]['csv'],first_verified_target_cost=new[-1]['cost'],
            target_gpu_uuid_verified=q['uuid'])
        save(STATE,state)
        signal_source(identity,'SIGTERM');signal_source(identity,'SIGCONT');paused=False
        for _ in range(30):
            if not status(SRC)['workers']:break
            time.sleep(1)
        else:raise RuntimeError('Source retirement not confirmed')
        retired=True
        archive=ROOTS[SRC]+'/recovery_backups/retired_gpu_workers/'+JOB+'_'+SIDE+'_'+str(time.time_ns())
        py(SRC,f'''
import time
from pathlib import Path
d=Path({src!r})
for _ in range(120):
 if (d/'exit_code.txt').is_file():break
 time.sleep(.5)
else:raise RuntimeError('Retired source wrapper marker not observed')
archive=Path({archive!r});archive.parent.mkdir(parents=True,exist_ok=True)
assert not archive.exists()
d.rename(archive)
''')
        state.update(phase='handoff_completed_first_target_epoch_verified',source_retired=True,
            source_paused=False,source_canonical_path_vacated_for_final_collection=True,
            retired_source_archive=archive)
        save(STATE,state)
        py(DST,f'from pathlib import Path; Path({dst+"/handoff_"+SIDE+"_state.json"!r}).write_text({json.dumps(state,indent=2)+chr(10)!r})')
        print(JOB+': first new target epoch and source retirement verified',flush=True)
    except Exception as error:
        live=None;absent=not launch_requested
        try:
            live=status(DST)['workers']
            session_absent=ssh(DST,ROOTS[DST]+'/env/bin/tmux has-session -t '+shlex.quote(SESSION)+' >/dev/null 2>&1 && echo live || echo absent').strip()=='absent'
            absent=not live and session_absent
        except (OSError,ValueError,subprocess.SubprocessError):pass
        if paused and absent:
            try:signal_source(identity,'SIGCONT');paused=False
            except Exception as resume_error:state['source_resume_error']=str(resume_error)
        state.update(phase='needs_inspection_after_handoff_error',source_paused=paused,
            source_retired=retired,target_workers=live,target_launch_requested=launch_requested,
            target_absence_confirmed=absent,error=str(error))
        save(STATE,state)
        raise


if __name__=='__main__':
    run()
