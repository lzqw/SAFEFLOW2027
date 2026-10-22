"""Move only two existing short cells to newly idle A100 capacity.

No new experiment, budget change, model change, scientific evaluation, foreign
process action, global GPU setting, or forced MPS-client kill is permitted.
The live source is held until a new target epoch is observed. A timeout is not
target failure; --finish inspects the same target rather than relaunching it.
"""
import argparse
import hashlib
import importlib.util
import json
import shlex
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
COMMIT = '62ca7399357c10645a955dcbead7ed2153a82c69'
HOST = 'tjucta-a100'
ROOTS = {
    HOST: '/home/lzqw/SAFEFLOW2027_A100',
    '4080': '/home/carla/SAFEFLOW2027_4080',
    '5080': '/home/autodrive/LZQW/SAFEFLOW2027',
}
CODES = {h: r + ('/code/training-62ca739' if h == HOST else '/code') for h, r in ROOTS.items()}
SPECS = {'point-goal-2027': ('5080', 10000000, 0), 'walker2d-2028': ('4080', 5000000, 1)}
PIPES = {
    0: ROOTS[HOST] + '/operations/mps_formal_a100_gpu0_20261020',
    1: ROOTS[HOST] + '/operations/mps_formal_a100_gpu1_20261020',
}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def save(path, state):
    state['updated_at_unix'] = time.time()
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(path)
    print(state['phase'], flush=True)


def capacity(h, index, incoming_bytes):
    return json.loads(h.py(HOST, f'''
import csv,json,os,shutil,subprocess
from pathlib import Path
def query(flag):
 return list(csv.reader(subprocess.check_output(['nvidia-smi',flag,'--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()))
r=next([v.strip() for v in r] for r in query('--query-gpu=index,uuid,memory.free,utilization.gpu') if int(r[0])=={index})
assert int(r[2])>=16384 and int(r[3])<=60,'Target no longer has admission headroom'
env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY={PIPES[index]!r})
def mps(command):
 return subprocess.run(['nvidia-cuda-mps-control'],input=command+'\\n',env=env,text=True,capture_output=True,check=True,timeout=10).stdout
servers=[int(v) for v in mps('get_server_list').split() if v.isdigit()]
assert len(servers)==1,'Use existing MPS server only'
clients=[int(v) for v in mps('get_client_list '+str(servers[0])).split() if v.isdigit()]
seen_server=False
for uuid,pid,executable in query('--query-compute-apps=gpu_uuid,pid,process_name'):
 if uuid.strip()!=r[1]:continue
 pid=int(pid.strip())
 if pid==servers[0]:
  assert 'nvidia-cuda-mps-server' in executable;seen_server=True;continue
 a=[v.decode(errors='replace') for v in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\\0') if v]
 assert pid in clients and 'scripts/star/train_star.py' in a,'Foreign/unknown GPU process; source untouched'
 assert a[a.index('--config')+1].startswith({ROOTS[HOST]+'/runs/'!r})
assert seen_server
free=shutil.disk_usage({ROOTS[HOST]!r}).free
assert free>={incoming_bytes}+20*1024**3,'Keep20GiB root headroom'
assert len(os.sched_getaffinity(0))>=16 and os.getloadavg()[0]<8
mem=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
assert mem>=8*1024**3
print(json.dumps(dict(index={index},uuid=r[1],free_mib=int(r[2]),utilization_percent=int(r[3]),
 server_pid=servers[0],existing_clients=clients,free_bytes=free,load=os.getloadavg(),memory_available_bytes=mem,
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())))
'''))


def finish(h, observer, path, state):
    target = h.status(HOST)
    live = [w for w in target['workers'] if w['pid'] == state['target_pid']]
    assert len(live) == 1, 'Target identity missing; inspect, do not relaunch'
    assert target['boot_id'] == state['target_capacity']['boot_id']
    new = [r for r in target['progress'] if Path(r['csv']).parent.name not in state['prior_logger_names']
           and r['steps'] >= state['resume_steps'] + 5000]
    if not new:
        return False
    native = observer.probe(HOST)
    worker = next(w for w in native['training'] if w['pid'] == state['target_pid'])
    assert worker['cuda_execution_backend'] == 'mps_verified'
    assert worker['mps_gpu_uuid_verified'] == state['target_capacity']['uuid']
    state.update(phase='first_target_epoch_verified', target_identity=live[0],
                 first_target_epoch=new[-1], first_target_worker=worker)
    save(path, state)
    if not state['source_retired']:
        current = h.status(h.SRC)
        source = [w for w in current['workers'] if w['pid'] == state['source_identity']['pid']]
        if source:
            assert source[0]['state'] == 'T'
            h.signal_source(state['source_identity'], 'SIGTERM')
            h.signal_source(state['source_identity'], 'SIGCONT')
            for _ in range(60):
                if not h.status(h.SRC)['workers']:
                    break
                time.sleep(1)
            else:
                raise RuntimeError('Source cleanup not finished; never force-kill an MPS client')
        assert not h.status(h.SRC)['workers']
        archive = state['retired_source_archive']
        h.py(h.SRC, f'''
from pathlib import Path
p=Path({h.container(h.SRC)!r});a=Path({archive!r})
assert p.is_dir() and not p.is_symlink() and not a.exists()
a.parent.mkdir(parents=True,exist_ok=True)
assert p.stat().st_dev==a.parent.stat().st_dev
p.rename(a)
''')
    state.update(phase='completed_first_target_epoch_source_retired', source_retired=True, source_paused=False)
    save(path, state)
    h.py(HOST, f'from pathlib import Path; Path({h.container(HOST)+"/handoff_"+state["side"]+"_state.json"!r}).write_text({json.dumps(state,indent=2)+chr(10)!r})')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job', choices=SPECS)
    parser.add_argument('--finish', action='store_true')
    args = parser.parse_args()
    job = args.job
    source_host, budget, index = SPECS[job]
    side = 'short_to_idle_a100_' + job + '_20261022'
    path = BASE / (side + '_state.json')
    helper = BASE / 'rebalance_goal2030_a100_to_5080_20261020_0621.py'
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == '690fb3391274fe30f08aabb1b91f8df246276547bac9b3bd21a3f3a2c4158323'
    h = module(helper, 'short_cell_rpc')
    h.SRC, h.DST, h.JOB, h.ROOTS, h.CODES = source_host, HOST, job, ROOTS, CODES
    observer = module(BASE / 'watch_gpu_capacity.py', 'short_cell_observer')
    if args.finish:
        state = json.loads(path.read_text())
        assert state['job'] == job and state['target_launch_requested']
        if state['phase'] == 'completed_first_target_epoch_source_retired':
            print('Already completed; no action', flush=True)
        else:
            print('finished' if finish(h, observer, path, state) else 'Same target live; waiting for first epoch', flush=True)
        return
    assert not path.exists(), 'Existing transaction must be inspected with --finish, never duplicate'
    snapshot = json.loads((BASE / 'gpu_capacity_state.json').read_text())
    gpu = next(g for g in snapshot['hosts'][HOST]['gpus'] if g['index'] == index)
    assert gpu['status'] == 'available_for_rebalance' and not gpu['foreign_pids']
    assert snapshot['observed_at_unix'] - gpu['candidate_since_unix'] >= 1800
    assert 0 <= time.time() - snapshot['observed_at_unix'] < 2100
    expected = next(w for w in snapshot['hosts'][source_host]['training'] if w['job'] == job)
    original = h.status(source_host)
    assert len(original['workers']) == 1 and not h.status(HOST)['workers']
    identity = dict(original['workers'][0], boot_id=original['boot_id'])
    assert all(identity[k] == expected[k] for k in ('pid', 'start_ticks', 'boot_id'))
    assert identity['device'] == 'cuda:0'
    for flag, value in {'--seed': job.rsplit('-', 1)[1], '--steps': str(budget),
                        '--batch-size': '256', '--replay-size': '400000',
                        '--vector-envs': '1', '--steps-per-epoch': '5000'}.items():
        assert identity['args'][identity['args'].index(flag) + 1] == value
    src, dst = h.container(source_host), h.container(HOST)
    h.ssh(HOST, 'test ! -e ' + shlex.quote(dst))
    for host in (source_host, HOST):
        assert h.ssh(host, 'git -C ' + shlex.quote(CODES[host]) + ' rev-parse HEAD').strip() == COMMIT
        assert not h.ssh(host, 'git -C ' + shlex.quote(CODES[host]) + ' status --porcelain').strip()
    size = int(h.ssh(source_host, 'du -sb ' + shlex.quote(src)).split()[0])
    target_capacity = capacity(h, index, size + 512 * 1024**2)
    state = dict(job=job, side=side, source_host=source_host, target_host=HOST,
        source_identity=identity, source_commit=COMMIT, total_budget_steps=budget,
        target_capacity=target_capacity, qualification_snapshot_at=snapshot['observed_at_unix'],
        target_gpu_index=index, source_paused=False, source_retired=False, target_launch_requested=False,
        scientific_parameters_changed=False, other_projects_paused=False, additional_evaluation_rollouts=0,
        bitwise_equivalence_claimed=False, new_experiment_cells=0, phase='waiting_for_fresh_full_checkpoint',
        prior_logger_names=[Path(r['csv']).parent.name for r in original['progress']],
        retired_source_archive=ROOTS[source_host]+'/recovery_backups/retired_gpu_workers/'+job+'_'+side,
        selection_reason='Short remaining budget fits A100 while releasing fast-card capacity for long cells; not selected by reward or cost',
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(path, state)
    paused = False
    try:
        # Wait for a normal full-checkpoint boundary; do not discard a long tail.
        csv_path = expected['progress_csv']
        deadline = time.monotonic() + 3600
        while True:
            n = int(h.py(source_host, f'''
import csv
from pathlib import Path
p=Path('/proc/{identity['pid']}')
assert int(p.joinpath('stat').read_text().rsplit(') ',1)[1].split()[19])=={identity['start_ticks']}
with Path({csv_path!r}).open() as f:r=list(csv.DictReader(f))[-1]
print(int(float(r['TotalEnvSteps'])))
''').strip())
            if n % 50000 <= 5000:
                break
            if time.monotonic() >= deadline:
                state.update(phase='fresh_checkpoint_wait_timeout_source_untouched');save(path, state);return
            time.sleep(30)
        state['target_capacity'] = capacity(h, index, size + 512 * 1024**2)
        # Recheck the actual state at the moment of the hold, not an earlier
        # /proc snapshot. Never take ownership of an already-stopped worker.
        h.py(source_host, f'''
import os,signal
from pathlib import Path
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={identity['boot_id']!r}
p=Path('/proc/{identity['pid']}')
f=p.joinpath('stat').read_text().rsplit(') ',1)[1].split()
assert int(f[19])=={identity['start_ticks']} and f[0] not in ('T','t','Z','X')
assert {str(src+'/resolved_formal_config.yaml').encode()!r} in p.joinpath('cmdline').read_bytes().split(b'\\0')
os.kill({identity['pid']},signal.SIGSTOP)
''');paused = True
        state.update(phase='source_held_copying_full_container', source_paused=True);save(path, state)
        full = json.loads(h.py(source_host, f'''
import csv,hashlib,json,time,torch
from pathlib import Path
for _ in range(50):
 if Path('/proc/{identity['pid']}/status').read_text().split('State:',1)[1].split()[0]=='T':break
 time.sleep(.1)
else:raise RuntimeError('Source hold unproven')
d=Path({src!r});p=Path({csv_path!r}).parent/'checkpoints/latest_full.pt'
blob=p.read_bytes();x=torch.load(p,map_location='cpu',weights_only=False)
assert x.get('rng') and x.get('optimizers') and x.get('replay') is not None
with Path({csv_path!r}).open() as f:r=list(csv.DictReader(f))[-1]
n=int(x['counters']['env_steps']);logged=int(float(r['TotalEnvSteps']))
assert 0<=logged-n<=5000 and 0<n<{budget}
q=d/'resume_inputs'/{side!r}/'latest_full.pt';q.parent.mkdir(parents=True,exist_ok=True)
assert not q.exists();q.write_bytes(blob)
sha=hashlib.sha256(blob).hexdigest()
print(json.dumps(dict(relative_checkpoint=str(q.relative_to(d)),checkpoint_sha256=sha,
 resume_steps=n,counters=x['counters'],resume_cumulative_training_cost=x['counters']['cumulative_training_cost'],
 last_logged_source_steps=logged,last_logged_source_cost=float(r['Train/CumulativeCost']),
 full_replay_rng_optimizer_present=True,source_torch_version=str(torch.__version__))))
''', True))
        state.update(full, known_repeated_steps_lower_bound=full['last_logged_source_steps']-full['resume_steps'],
            known_extra_training_cost_lower_bound=full['last_logged_source_cost']-full['resume_cumulative_training_cost'],
            unlogged_source_tail_unknown_not_zero=True)
        save(path, state)
        send = 'tar -cf - -C ' + shlex.quote(str(Path(src).parent)) + ' -- ' + shlex.quote(job)
        receive = ('set -e; test ! -e ' + shlex.quote(dst) + '; stage_dir=$(mktemp -d '
                   + shlex.quote(ROOTS[HOST]+'/runs/.short-cell-handoff.XXXXXX')
                   + '); tar -xf - --no-same-owner -C "$stage_dir"; test ! -e '+shlex.quote(dst)
                   + '; mv "$stage_dir/'+job+'" '+shlex.quote(dst)+'; rmdir "$stage_dir"')
        with subprocess.Popen([*h.OPTIONS, source_host, send], stdout=subprocess.PIPE) as producer:
            try:
                copied = subprocess.run([*h.OPTIONS, HOST, receive], stdin=producer.stdout, timeout=900)
            finally:
                producer.stdout.close()
                producer.wait(timeout=30)
            assert producer.returncode == copied.returncode == 0
        checkpoint = dst + '/' + full['relative_checkpoint']
        state['target_checkpoint'] = checkpoint
        relocation = dict(format='shapo-artifact-relocation-v1',job=job,original_host=source_host,
            target_host=HOST,original_container=src,scientific_protocol_changed=False,original_evidence_modified=False)
        result = json.loads(h.py(HOST, f'''
import hashlib,json
from pathlib import Path
import torch
d=Path({dst!r});p=Path({checkpoint!r})
assert hashlib.sha256(p.read_bytes()).hexdigest()=={full['checkpoint_sha256']!r}
for name in ['artifact_relocation.json','receipt.json','exit_code.txt']:
 p=d/name
 if p.exists():p.rename(d/(name+'.before_{side}'))
(d/'artifact_relocation.json').write_text({json.dumps(relocation,indent=2)+chr(10)!r})
print(json.dumps(dict(target_torch_version=str(torch.__version__))))
''', True))
        state.update(result)
        # Pin the exact resume input on the separate4T filesystem as well.
        archive = '/home/carla/4T/SAFEFLOW2027_4080/runtime_resume_inputs_20261022/short_to_a100/'+job+'/latest_full.pt'
        h.py('4080', f'from pathlib import Path; p=Path({archive!r}); p.parent.mkdir(parents=True,exist_ok=True); assert not p.exists() and not Path(str(p)+".partial").exists(); assert p.parent.stat().st_dev!=Path({ROOTS["4080"]!r}).stat().st_dev')
        subprocess.run(['scp','-3','-o','BatchMode=yes','-o','ConnectTimeout=12',HOST+':'+checkpoint,'4080:'+archive+'.partial'],check=True,timeout=600)
        h.py('4080', f'import hashlib; from pathlib import Path; p=Path({archive!r}); q=Path(str(p)+".partial"); assert hashlib.sha256(q.read_bytes()).hexdigest()=={full["checkpoint_sha256"]!r}; q.rename(p)')
        state.update(phase='target_launch_requested', target_launch_requested=True, archive_resume_input=archive)
        save(path, state)
        state['target_pid'] = int(h.py(HOST, f'''
import os,subprocess
from pathlib import Path
d=Path({dst!r});root=Path({ROOTS[HOST]!r})
env=dict(os.environ,CUDA_VISIBLE_DEVICES={state['target_capacity']['uuid']!r},
 CUDA_MPS_PIPE_DIRECTORY={PIPES[index]!r},OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
 PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH={CODES[HOST]!r})
a=['nice','-n','15',str(root/'env/bin/python'),str(root/'operations/mps_training_bootstrap_20261020.py'),
 'scripts/star/train_star.py','--config',str(d/'resolved_formal_config.yaml'),
 '--seed',{job.rsplit('-',1)[1]!r},'--steps',{str(budget)!r},'--device','cuda:0','--vector-envs','1',
 '--batch-size','256','--replay-size','400000','--steps-per-epoch','5000','--log-dir',str(d/'training'),
 '--result',str(d/'receipt.json'),'--no-tensorboard','--resume',{checkpoint!r}]
with (d/'train_{side}.log').open('xb') as log:
 p=subprocess.Popen(a,cwd={CODES[HOST]!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(p.pid)
''').strip())
        state.update(phase='same_target_running_source_held');save(path, state)
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            if finish(h, observer, path, state):
                return
            time.sleep(45)
        state.update(phase='waiting_for_target_epoch_live_target_not_failure');save(path, state)
    except Exception as error:
        absent = not state.get('target_launch_requested')
        if state.get('target_launch_requested'):
            try:
                absent = not h.status(HOST)['workers']
            except Exception:
                absent = False
        if paused and absent:
            h.signal_source(identity, 'SIGCONT')
            state['source_paused'] = False
        state.update(phase='needs_inspection_no_automatic_relaunch',error=str(error),
                     target_absence_confirmed=absent)
        save(path, state)
        raise


if __name__ == '__main__':
    main()
