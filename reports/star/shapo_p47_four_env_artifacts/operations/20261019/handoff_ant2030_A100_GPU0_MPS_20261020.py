#!/usr/bin/env python3
"""Existing Ant-2030 CPU cell: fresh full checkpoint -> private A100 GPU0 MPS.

Never edits the learner/config, deletes old segments, changes GPU compute mode,
or signals unrelated processes. Source remains paused until a new target epoch.
"""
import hashlib
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = '/home/lzqw/SAFEFLOW2027_A100'
CODE = ROOT + '/code/training-62ca739'
HOST = 'tjucta-a100'
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
BOOT = '6928f333-1036-473c-92d5-4c482d5fef59'
GPU = "GPU-7b6a829b-7b56-af3e-cf56-1af815c62dbe"
SIDE = 'cpu_to_mps_a100_gpu0_20261020'
PIPE = ROOT + '/operations/mps_formal_a100_gpu0_20261020'
LOG = ROOT + '/operations/mps_formal_a100_gpu0_20261020_logs'
BOOTSTRAP = ROOT + '/operations/mps_training_bootstrap_20261020.py'
HELPER = BASE/'rebalance_goal2030_a100_to_5080_20261020_0621.py'
HELPER_SHA = '690fb3391274fe30f08aabb1b91f8df246276547bac9b3bd21a3f3a2c4158323'
STATE = BASE/'ant2030_CPU_to_MPS_A100_GPU0_20261020_controller_state.json'


def save(path, value):
    value['updated_at_unix'] = time.time()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)
    print(json.dumps({'file':path.name,'phase':value.get('phase'),'time':value['updated_at_unix']}), flush=True)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def helpers(job):
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
    h = module(HELPER, 'verified_mps_rpc_'+job.replace('-','_'))
    h.SRC = h.DST = HOST
    h.ROOTS = {HOST:ROOT}
    h.CODES = {HOST:CODE}
    h.JOB = job
    return h


def identity(h, expected):
    current = h.status(HOST)
    assert current['boot_id'] == BOOT
    source = next(w for w in current['workers'] if w['pid']==expected['pid'])
    assert source['start_ticks'] == expected['start_ticks'] and source['device']=='cpu'
    assert source['cuda_visible_devices']=='-1'
    return {**source, 'boot_id':BOOT}, current


def capacity(h):
    return json.loads(h.py(HOST, f'''
import csv,json,os,shutil,subprocess
from pathlib import Path
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={BOOT!r}
rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,compute_mode,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()))
r=next(r for r in rows if r[0].strip()=={GPU!r});assert int(r[1])>=16384 and r[2].strip()=='Default'
rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True).splitlines())
owned=[];foreign=[]
for uuid,pid,executable in rows:
 if uuid.strip()!={GPU!r}:continue
 p=Path('/proc')/pid.strip();a=[]
 try:a=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(bytes([0])) if v]
 except OSError:pass
 if 'nvidia-cuda-mps-server' in executable or any(Path(v).name=='nvidia-cuda-mps-server' for v in a):
  env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY={PIPE!r})
  server=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\\n',env=env,text=True,capture_output=True,check=True,timeout=10).stdout
  assert int(pid.strip()) in [int(v) for v in server.split() if v.isdigit()]
 else:
  if 'scripts/star/train_star.py' not in a or '--config' not in a or not a[a.index('--config')+1].startswith({ROOT+'/runs/'!r}):
   assert executable.strip() not in ('','N/A','Unknown'),'GPU process type unproven'
   foreign.append(int(pid.strip()));continue
 owned.append(int(pid.strip()))
assert int(r[3])<=(60 if foreign else 85),'GPU0 currently too busy for shared admission'
assert shutil.disk_usage({ROOT!r}).free>=8*1024**3
available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
assert available>=8*1024**3
print(json.dumps({{'gpu_uuid':{GPU!r},'owned_GPU_OS_pids':owned,'foreign_pids':foreign,
 'memory_free_MiB':int(r[1]),'compute_mode':r[2].strip(),'utilization_percent':int(r[3]),
 'no_foreign_process_actions':True,'shared_load_ceiling_percent':60,'no_speed_gain_guaranteed':True}}))
'''))


def checkpoint(h, csv, snapshot_copy=None):
    return json.loads(h.py(HOST, f'''
import hashlib,io,json,math,shutil
from pathlib import Path
import torch
p=Path({csv!r}).parent/'checkpoints/latest_full.pt'
blob=p.read_bytes();x=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
assert x.get('format')=='star-checkpoint' and x.get('replay') is not None
assert x.get('rng') and x.get('optimizers')
assert all(torch.isfinite(v).all() for v in x['model'].values() if torch.is_tensor(v))
n=int(x['counters']['env_steps']);c=float(x['counters']['cumulative_training_cost'])
assert 0<n<5000000 and math.isfinite(c) and c>=0
copy={snapshot_copy!r}
if copy:
 target=Path(copy);target.parent.mkdir(parents=True,exist_ok=True)
 assert not target.exists();target.write_bytes(blob)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(blob).hexdigest()
print(json.dumps({{'source_checkpoint':str(p),'checkpoint_sha256':hashlib.sha256(blob).hexdigest(),
 'resume_steps':n,'resume_cumulative_training_cost':c,'full_replay_rng_optimizers_finite_model':True,
 'immutable_input_copy':copy}}))
''',True))


def mps_proof(h, pid):
    return json.loads(h.py(HOST, f'''
import csv,json,os,subprocess
from pathlib import Path
p=Path('/proc/{pid}')
envs=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\\0') if b'=' in v)
assert envs.get(b'CUDA_MPS_PIPE_DIRECTORY')=={PIPE.encode()!r}
env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY={PIPE!r})
def q(command):
 return subprocess.run(['nvidia-cuda-mps-control'],input=command+'\\n',env=env,text=True,capture_output=True,check=True,timeout=10).stdout
servers=[int(v) for v in q('get_server_list').split() if v.isdigit()];assert len(servers)==1
clients=[int(v) for v in q('get_client_list '+str(servers[0])).split() if v.isdigit()];assert {pid} in clients
rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())
assert any(uuid.strip()=={GPU!r} and int(pid.strip())==servers[0] for uuid,pid in rows)
print(json.dumps({{'client_pid':{pid},'server_pid':servers[0],'client_pids':clients,
 'pipe_directory':{PIPE!r},'gpu_uuid':{GPU!r},'connection_verified':True}}))
'''))


def preserve_input(h, job, full):
    """Keep verified immutable copies on the host and4080's separate4T disk."""
    source=full['source_checkpoint']
    assert re.fullmatch(r'[A-Za-z0-9_./{}-]+',source)
    directory=Path('/home/lzqw/SAFEFLOW2027_artifacts/recovered_a100_4080/mps_runtime_transitions')/job/SIDE/'input'
    directory.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(directory).free>=2*1024**3
    target=directory/'latest_full.pt';partial=directory/'latest_full.pt.partial'
    assert not target.exists() and not partial.exists()
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=12',HOST+':'+source,str(partial)],
                   check=True,capture_output=True,text=True,timeout=3600)
    assert hashlib.sha256(partial.read_bytes()).hexdigest()==full['checkpoint_sha256']
    partial.rename(target)
    archive='/home/carla/4T/SAFEFLOW2027_4080/mps_runtime_transitions/'+job+'/'+SIDE+'/input/latest_full.pt'
    prepare=f'''
from pathlib import Path
import shutil
p=Path({archive!r});p.parent.mkdir(parents=True,exist_ok=True)
assert p.parent.stat().st_dev!=Path('/home/carla/SAFEFLOW2027_4080').stat().st_dev
assert shutil.disk_usage(p.parent).free>=2*1024**3
assert not p.exists() and not Path(str(p)+'.partial').exists()
'''
    h.ssh('4080','python3 -c '+shlex.quote(prepare))
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=12',str(target),'4080:'+archive+'.partial'],
                   check=True,capture_output=True,text=True,timeout=3600)
    finish=f'''
import hashlib,json
from pathlib import Path
p=Path({archive!r});part=Path(str(p)+'.partial')
assert hashlib.sha256(part.read_bytes()).hexdigest()=={full['checkpoint_sha256']!r}
assert not p.exists();part.rename(p)
print(json.dumps({{'archive_input_path':str(p),'archive_input_sha256':{full['checkpoint_sha256']!r},
 'archive_separate_4T_filesystem_verified':True}}))
'''
    archive_proof=json.loads(h.ssh('4080','python3 -c '+shlex.quote(finish)))
    return {'host_input_path':str(target),'host_input_sha256':full['checkpoint_sha256'],**archive_proof}


def start_daemon(h, controller):
    if controller.get('private_daemon_started'):
        return
    native = json.loads(h.py(HOST, f'''
import json,os,subprocess
from pathlib import Path
pipe=Path({PIPE!r});log=Path({LOG!r})
assert not pipe.exists() and not log.exists(),'Inspect existing private runtime; never duplicate daemon'
assert len(str(pipe/'control').encode())<100
pipe.mkdir(parents=True);log.mkdir(parents=True)
env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(log),CUDA_VISIBLE_DEVICES={GPU!r})
subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True,cwd={CODE!r},timeout=20)
print(json.dumps({{'pipe_directory':str(pipe),'log_directory':str(log),'compute_mode_unchanged':True}}))
'''))
    controller.update(private_daemon_started=True, daemon=native)
    save(STATE, controller)


def transition(h, expected, controller):
    job = expected['job']; prefix=job+'_'+SIDE
    path=BASE/(prefix+'_state.json')
    assert not path.exists(), 'Never duplicate an existing per-cell transition'
    source, current = identity(h, expected)
    assert len(current['workers'])==1
    for flag,value in {'--seed':'2030','--steps':'5000000','--device':'cpu',
                       '--vector-envs':'1','--batch-size':'256','--replay-size':'400000',
                       '--steps-per-epoch':'5000'}.items():
        assert source['args'][source['args'].index(flag)+1]==value
    old_csv = expected['progress_csv']
    last = next(r for r in current['progress'] if r['csv']==old_csv)
    # Full snapshots occur every50k steps. Avoid repeatedly deserializing400k replay
    # while the latest logged epoch is nowhere near a fresh snapshot boundary.
    if last['steps'] % 50000 > 5000:
        return False
    full = checkpoint(h, old_csv)
    if not 0<=last['steps']-full['resume_steps']<=5000:
        return False
    root=h.container(HOST); side=root+'/'+SIDE
    state={'job':job,'phase':'preparing_fresh_full_checkpoint','source_identity':source,
           'source_commit':SOURCE,'total_budget_steps':5000000,'source_progress_csv':old_csv,
           'source_runtime_backend':'CPU','target_runtime_backend':'private_MPS',
           'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'bootstrap_sha256':hashlib.sha256((BASE/'mps_training_bootstrap_20261020.py').read_bytes()).hexdigest(),
           'source_paused':False,'source_retired':False,'target_launch_requested':False,
           'original_training_segments_preserved_at_original_paths':True,
           'scientific_parameters_changed':False,'other_projects_modified':False}
    save(path,state)
    paused=False; launched=False
    session='safeflow_'+prefix.replace('-','_')
    try:
        assert h.ssh(HOST,'git -C '+CODE+' rev-parse HEAD').strip()==SOURCE
        assert not h.ssh(HOST,'git -C '+CODE+' status --porcelain').strip()
        state['capacity_before_pause']=capacity(h)
        identity(h, expected)
        h.py(HOST,f'''from pathlib import Path
assert b'CUDA_MPS_PIPE_DIRECTORY=' not in Path('/proc/{source['pid']}/environ').read_bytes()
''')
        state['source_CPU_backend_verified']=True
        h.signal_source(source,'SIGSTOP');paused=True
        source_now,current=identity(h,expected)
        assert source_now['state'] in ('T','t')
        last=next(r for r in current['progress'] if r['csv']==old_csv)
        full=checkpoint(h,old_csv,side+'/input/latest_full.pt')
        assert 0<=last['steps']-full['resume_steps']<=5000
        state.update(full,phase='source_paused_full_input_preserved',source_paused=True,
            last_logged_source_steps=last['steps'],last_logged_source_cost=last['cost'],
            known_repeated_steps_lower_bound=last['steps']-full['resume_steps'],
            known_extra_training_cost_lower_bound=last['cost']-full['resume_cumulative_training_cost'],
            unlogged_source_tail_unknown_not_zero=True)
        assert state['known_extra_training_cost_lower_bound']>=0
        save(path,state)
        state['input_preservation']=preserve_input(h,job,full)
        save(path,state)
        h.py(HOST,f'''from pathlib import Path
d=Path({root!r});side=Path({side!r})
if (d/'exit_code.txt').exists():
 archive=side/'preexisting_exit_code.txt';assert not archive.exists()
 (d/'exit_code.txt').rename(archive)
''')
        start_daemon(h,controller)
        prior={r['csv'] for r in current['progress']}
        shell=f'''#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY={shlex.quote(PIPE)}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={CODE}
cd {CODE}
set +e
{ROOT}/env/bin/python {BOOTSTRAP} scripts/star/train_star.py --config {root}/resolved_formal_config.yaml --seed 2030 --steps 5000000 --device cuda:0 --vector-envs 1 --batch-size 256 --replay-size 400000 --steps-per-epoch 5000 --log-dir {root}/training --result {root}/receipt.json --no-tensorboard --resume {shlex.quote(full['source_checkpoint'])} >{side}/target.stdout 2>&1
status=$?
printf '%s\\n' "$status" >{side}/target_exit_code.txt
if test -f {side}/target_promoted; then printf '%s\\n' "$status" >{root}/exit_code.txt; fi
exit "$status"
'''
        launcher=ROOT+'/deploy/resume_'+prefix+'.sh'
        h.py(HOST,f'from pathlib import Path; Path({launcher!r}).write_text({shell!r})')
        capacity(h)
        launched=True;state.update(phase='MPS_target_starting_source_paused',target_launch_requested=True,
            target_session=session,target_stdout=side+'/target.stdout',target_exit_code=side+'/target_exit_code.txt')
        save(path,state)
        h.ssh(HOST,'/usr/bin/tmux new-session -d -s '+shlex.quote(session)+' '+shlex.quote('bash '+launcher))
        deadline=time.monotonic()+3600
        while time.monotonic()<deadline:
            target=h.status(HOST)
            workers=[w for w in target['workers'] if w['pid']!=source['pid']]
            new=[r for r in target['progress'] if r['csv'] not in prior and r['steps']>=full['resume_steps']+5000]
            if len(workers)==1 and new:
                break
            marker=h.ssh(HOST,'test ! -e '+shlex.quote(side+'/target_exit_code.txt')+' || cat '+shlex.quote(side+'/target_exit_code.txt')).strip()
            if not workers and marker:
                raise RuntimeError('MPS target terminal before first epoch; preserve failed attempt and resume original only after absence proof')
            time.sleep(30)
        else:
            raise RuntimeError('No first MPS epoch yet; inspect same target, never duplicate')
        target_id={**workers[0],'boot_id':target['boot_id']}
        assert target_id['device']=='cuda:0' and target_id['boot_id']==BOOT
        proof=mps_proof(h,target_id['pid'])
        first=new[-1]
        state.update(phase='first_MPS_epoch_verified_source_still_paused',target_identity=target_id,
            first_target_epoch=first,MPS_connection_proof=proof)
        save(path,state)
        h.signal_source(source,'SIGTERM');h.signal_source(source,'SIGCONT');paused=False
        for _ in range(60):
            if not any(w['pid']==source['pid'] for w in h.status(HOST)['workers']):break
            time.sleep(1)
        else:raise RuntimeError('Original CPU worker retirement not proven')
        state['source_retired']=True
        marker=json.loads(h.py(HOST,f'''
import hashlib,json,time
from pathlib import Path
d=Path({root!r});side=Path({side!r})
for _ in range(120):
 if (d/'exit_code.txt').is_file():break
 time.sleep(.5)
else:raise RuntimeError('Original wrapper exit marker not observed')
raw=(d/'exit_code.txt').read_bytes();text=raw.decode().strip()
literal_newline=text.endswith(chr(92)+'n')
if literal_newline:text=text[:-2]
exit_code=int(text);assert exit_code!=0
archive=side/'original_wrapper_exit_code.txt';assert not archive.exists()
(d/'exit_code.txt').rename(archive)
assert hashlib.sha256(Path({full['source_checkpoint']!r}).read_bytes()).hexdigest()=={full['checkpoint_sha256']!r}
(side/'target_promoted').touch()
print(json.dumps({{'original_wrapper_exit_code':exit_code,'archived_exit_marker':str(archive),
 'original_exit_marker_sha256':hashlib.sha256(raw).hexdigest(),
 'original_exit_marker_encoding':'literal_backslash_n' if literal_newline else 'normal_newline',
 'old_full_input_still_at_original_path_and_hash':True,'target_marker_promoted':True}}))
'''))
        observer=module(BASE/'watch_gpu_capacity.py','mps_first_epoch_native_observer')
        native=observer.probe(HOST)
        target_native=next(w for w in native['training'] if w['pid']==target_id['pid'])
        assert target_native.get('mps_gpu_uuid_verified')==GPU
        assert not any(w['pid']==source['pid'] for w in native['training'])
        native_path=BASE/(prefix+'_first_epoch_native_observation.json')
        save(native_path,native)
        state.update(phase='completed_first_MPS_epoch_source_retired',source_paused=False,
            target_first_epoch_native_observation=native_path.name,wrapper_retirement=marker)
        save(path,state)
        h.py(HOST,f'from pathlib import Path; Path({side+"/completion_state.json"!r}).write_text({json.dumps(state,indent=2)+chr(10)!r})')
        return True
    except Exception as error:
        absent=False
        try:
            now=h.status(HOST)
            others=[w for w in now['workers'] if w['pid']!=source['pid']]
            session_absent=h.ssh(HOST,'/usr/bin/tmux has-session -t '+shlex.quote(session)+' >/dev/null 2>&1 && echo live || echo absent').strip()=='absent'
            absent=not others and (session_absent or not launched)
        except (OSError,ValueError,subprocess.SubprocessError):pass
        resumed=False
        if paused and absent:
            h.signal_source(source,'SIGCONT');paused=False;resumed=True
        state.update(phase='needs_inspection_after_MPS_handoff_error',source_paused=paused,
            target_launch_requested=launched,target_absence_confirmed=absent,
            original_source_resumed_in_memory=resumed,error=str(error),automatic_retry=False)
        save(path,state)
        raise


def main():
    assert not STATE.exists(),'Inspect existing A100 controller; never duplicate it'
    readiness=json.loads((BASE/'cpu_to_mps_ant2030_a100_gpu0_20261020_preparation.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==readiness['controller_sha256']
    launch=json.loads((BASE/'cpu_to_mps_ant2030_a100_gpu0_20261020_launch_ready.json').read_text())
    assert launch['GPU0_shared_headroom_qualified'] is True
    assert launch['accounting_hook_integrated_before_any_transition'] is True
    accounting=module(BASE/'cpu_to_mps_a100_recovery_accounting_20261020.py','A100_MPS_accounting')
    assert accounting.verify_A100_CPU_to_MPS_transitions(BASE)==[]
    from host_full_input_cache_20261020 import ensure_cache
    ensure_cache(BASE)
    bootstrap=BASE/'mps_training_bootstrap_20261020.py'
    assert hashlib.sha256(bootstrap.read_bytes()).hexdigest()==readiness['prepared_bootstrap']['sha256']
    assert shutil.disk_usage('/home/lzqw/SAFEFLOW2027_artifacts/recovered_a100_4080').free>=2*1024**3+400*1024**2
    pending=readiness['workers_at_ordinary_observation'][:]
    assert {w['job'] for w in pending}=={'ant-2030'}
    h=helpers(pending[0]['job'])
    assert h.ssh(HOST,'git -C '+CODE+' rev-parse HEAD').strip()==SOURCE
    assert not h.ssh(HOST,'git -C '+CODE+' status --porcelain').strip()
    assert h.ssh(HOST,'sha256sum '+BOOTSTRAP).split()[0]==readiness['prepared_bootstrap']['sha256']
    capacity(h)
    controller={'phase':'waiting_for_fresh_A100_GPU0_full_checkpoints',
        'private_daemon_started':False,'completed_jobs':[],
        'Goal2027_and_Button2027_retained_on_original_GPUs':True,
        'GPU0_shared_foreign_processes_not_modified':True,
        'new_training_cells':0,'scientific_parameters_changed':False,
        'other_projects_modified':False,'performance_gain_not_guaranteed':True}
    save(STATE,controller)
    while pending:
        for expected in pending[:]:
            if transition(helpers(expected['job']),expected,controller):
                pending.remove(expected);controller['completed_jobs'].append(expected['job'])
                save(STATE,controller)
        if pending:time.sleep(300)
    controller['phase']='existing_Ant2030_running_on_A100_GPU0_MPS'
    save(STATE,controller)


if __name__=='__main__':
    main()
