#!/usr/bin/env python3
"""Move one existing Point MPS worker to newly stable4080 GPU capacity.

Never replay a per-cell transaction. Preserve full inputs, old directories, and
paused source until a genuine target5k epoch; no scientific selection/evaluation.
"""
import hashlib
import importlib.util
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
SIDE = 'point_5080_to_4080_MPS_20261021'
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
ROOTS = {'5080': '/home/autodrive/LZQW/SAFEFLOW2027', '4080': '/home/carla/SAFEFLOW2027_4080'}
BOOTS = {'5080': '5f46bb0c-b4ec-4802-8f6e-01ac611c0f2f', '4080': '3b362c03-4ed3-4e03-aea4-cf7339356026'}
GPUS = {'5080': 'GPU-ac315b09-0bb7-2072-efd1-cdfc5fae96a5', '4080': 'GPU-66ed7e12-a0c3-9273-463d-56a5e5f0a4f7'}
PIPES = {'5080': ROOTS['5080'] + '/operations/mps_formal_20261020',
         '4080': ROOTS['4080'] + '/operations/mps_formal_4080_20261021'}
BOOTSTRAP_SHA = 'e2152f91622afd2a12ce7ba8555ef70e523d92cefce6ad8a53a5525eaec9e6c0'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, BASE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    value['updated_at_unix'] = time.time()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps({'file': path.name, 'phase': value.get('phase')}), flush=True)


def helpers(job):
    filename = 'rebalance_goal2030_a100_to_5080_20261020_0621.py'
    assert digest(BASE / filename) == '690fb3391274fe30f08aabb1b91f8df246276547bac9b3bd21a3f3a2c4158323'
    h = load_module(filename, 'Point_GPU_RPC_' + job.replace('-', '_'))
    h.JOB, h.SRC, h.DST = job, '5080', '4080'
    h.ROOTS, h.CODES = ROOTS, {host: root + '/code' for host, root in ROOTS.items()}
    return h


def mps_helper(host):
    name = 'handoff_point2029_MPS_5080_20261020.py'
    assert digest(BASE / name) == 'ef42dd61a2c58bcb4a221af08bbaf5f4552fdae6064714e6ce4d8920c5010a02'
    m = load_module(name, 'Point_GPU_MPS_' + host)
    m.HOST, m.ROOT, m.BOOT, m.GPU, m.PIPE, m.SIDE = host, ROOTS[host], BOOTS[host], GPUS[host], PIPES[host], SIDE
    return m


def capacity(h):
    # Reuse only the OS CPU/memory/quota probe; never call a historical controller.
    cpu = load_module('handoff_Walker_CPU_5080_to_4080_20261021.py', 'Point_GPU_CPU_headroom').capacity(h, BOOTS['4080'])
    gpu = json.loads(h.py('4080', f'''
import csv,json,os,subprocess
from pathlib import Path
def query(flag):return list(csv.reader(subprocess.check_output(['nvidia-smi',flag,'--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()))
r=next([v.strip() for v in r] for r in query('--query-gpu=index,uuid,memory.free,utilization.gpu,compute_mode') if int(r[0])==0)
assert r[1]=={GPUS['4080']!r} and int(r[2])>=8192 and int(r[3])<=85 and r[4]=='Default','GPU no longer admissible'
env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY={PIPES['4080']!r})
servers=[];clients=[]
if Path({PIPES['4080']!r}).exists():
 def q(command):return subprocess.run(['nvidia-cuda-mps-control'],input=command+chr(10),env=env,text=True,capture_output=True,check=True,timeout=10).stdout
 servers=[int(v) for v in q('get_server_list').split() if v.isdigit()]
 assert len(servers)<=1
 for server in servers:clients.extend(int(v) for v in q('get_client_list '+str(server)).split() if v.isdigit())
for uuid,pid,_ in query('--query-compute-apps=gpu_uuid,pid,used_memory'):
 if uuid.strip()!={GPUS['4080']!r}:continue
 pid=int(pid.strip())
 if pid in servers+clients:continue
 a=[v.decode(errors='replace') for v in Path('/proc',str(pid),'cmdline').read_bytes().split(bytes([0])) if v]
 assert a and Path(a[0]).name in ('Xorg','Xwayland','gnome-shell','gnome-remote-desktop-daemon','kwin_wayland','kwin_x11','plasmashell'),'Foreign or unproven GPU process'
for pid in clients:
 a=[v.decode(errors='replace') for v in Path('/proc',str(pid),'cmdline').read_bytes().split(bytes([0])) if v]
 assert 'scripts/star/train_star.py' in a and a[a.index('--config')+1].startswith({ROOTS['4080']+'/runs/shapo_p47_four_env/runs/point-'!r})
print(json.dumps({{'uuid':r[1],'free_mib':int(r[2]),'utilization_percent':int(r[3]),'compute_mode':r[4],
 'foreign_gpu_processes':0,'MPS_server_pids':servers,'MPS_client_pids':clients}}))
'''))
    return {'CPU': cpu, 'GPU': gpu}


def run(job):
    prefix = job + '_' + SIDE
    state_path, receipt_path, plan_path = (BASE / (prefix + suffix) for suffix in ('_state.json', '_receipt.json', '_plan.json'))
    assert not state_path.exists() and not receipt_path.exists(), 'Never repeat a per-cell transaction'
    plan = json.loads(plan_path.read_text())
    assert plan['controller_sha256'] == digest(__file__)
    assert plan['accounting_module_sha256'] == digest(BASE / 'point_5080_to_4080_MPS_accounting_20261021.py')
    assert plan['accounting_hook_integrated_before_any_transition'] is True
    assert 'verify_Point_5080_to_4080_MPS(BASE)' in (BASE / 'publish_final.py').read_text()
    assert plan['job'] == job and plan['regular_gpu_status'] == 'available_for_rebalance'
    assert plan['regular_frame_at_unix'] - plan['candidate_since_unix'] >= 1800
    assert 0 <= time.time() - plan['regular_frame_at_unix'] <= 2100
    assert job in {slug + '-' + str(seed) for slug in ('point-goal', 'point-button') for seed in range(2027, 2032)}
    seed = int(job.rsplit('-', 1)[1])
    h = helpers(job)
    src, dst = h.container('5080'), h.container('4080')
    current = h.status('5080')
    assert len(current['workers']) == 1 and current['boot_id'] == BOOTS['5080']
    identity = {**current['workers'][0], 'boot_id': current['boot_id']}
    expected = plan['expected_source_identity']
    assert all(identity[k] == expected[k] for k in ('pid', 'start_ticks', 'device', 'boot_id'))
    for flag, value in {'--seed': str(seed), '--steps': '10000000', '--device': 'cuda:0', '--vector-envs': '1',
                        '--batch-size': '256', '--replay-size': '400000', '--steps-per-epoch': '5000'}.items():
        assert identity['args'][identity['args'].index(flag) + 1] == value
    assert identity['cuda_visible_devices'] == '0'
    source_bootstrap = ROOTS['5080'] + '/operations/mps_training_bootstrap_20261020.py'
    assert source_bootstrap in identity['args'], 'Only retire an existing synchronize-on-exit MPS worker'
    assert h.ssh('5080', 'sha256sum ' + source_bootstrap).split()[0] == BOOTSTRAP_SHA
    last = next(r for r in current['progress'] if r['csv'] == expected['progress_csv'])
    if last['steps'] % 50000 > 5000:
        return False
    ready = json.loads(h.py('5080', f"import json,torch;from pathlib import Path;p=Path({last['csv']!r}).parent/'checkpoints/latest_full.pt';x=torch.load(p,map_location='cpu',weights_only=False);print(json.dumps({{'steps':int(x['counters']['env_steps']),'bytes':p.stat().st_size}}))", True))
    if not 0 <= last['steps'] - ready['steps'] <= 5000:
        return False
    assert shutil.disk_usage('/home/lzqw/SAFEFLOW2027_artifacts').free >= 2*1024**3 + ready['bytes'] + 300*1024**2
    assert not h.status('4080')['workers']
    for host in ROOTS:
        assert h.ssh(host, 'git -C ' + h.CODES[host] + ' rev-parse HEAD').strip() == SOURCE
        assert not h.ssh(host, 'git -C ' + h.CODES[host] + ' status --porcelain').strip()
    source_proof = mps_helper('5080').mps_proof(h, identity['pid'])
    initial = capacity(h)
    assert digest(BASE / 'mps_training_bootstrap_20261020.py') == BOOTSTRAP_SHA
    bootstrap = ROOTS['4080'] + '/operations/mps_training_bootstrap_20261020.py'
    state = {**plan, 'plan': plan, 'job': job, 'seed': seed, 'source_commit': SOURCE,
             'source_host': '5080', 'target_host': '4080', 'total_training_budget': 10000000,
             'source_identity': identity, 'source_MPS_proof': source_proof,
             'target_capacity_before_pause': initial, 'phase': 'preparing_existing_Point_GPU_move',
             'scientific_parameters_changed': False, 'additional_evaluation': False,
             'additional_experiment_cells': 0, 'other_projects_modified': False,
             'runtime_change': {'from_torch': '2.7.1+cu128', 'to_torch': '2.7.1+cu118',
                                'from_backend': 'private_MPS', 'to_backend': 'private_MPS',
                                'cuda_build_changed': True, 'bitwise_equivalence_claimed': False}}
    save(state_path, state)
    paused = retired = launched = False
    session = 'safeflow_' + prefix.replace('-', '_')
    try:
        old_target = json.loads(h.py('4080', f'''
import json,time
from pathlib import Path
d=Path({dst!r});archive=None
if d.exists():
 assert d.is_dir() and not d.is_symlink()
 archive=Path({ROOTS['4080']!r})/'recovery_backups/retired_target_containers'/({job!r}+'_{SIDE}_'+str(time.time_ns()))
 archive.parent.mkdir(parents=True,exist_ok=True)
 assert d.stat().st_dev==archive.parent.stat().st_dev and not archive.exists()
 d.rename(archive)
print(json.dumps({{'previous_target_container_preserved_at':str(archive) if archive else None,'no_previous_target_files_deleted':True}}))
'''))
        state.update(old_target)
        save(state_path, state)
        h.signal_source(identity, 'SIGSTOP'); paused = True
        last = next(r for r in h.status('5080')['progress'] if r['csv'] == expected['progress_csv'])
        full = json.loads(h.py('5080', f'''
import hashlib,io,json,math,time,torch
from pathlib import Path
torch.set_num_threads(1);assert str(torch.__version__)=='2.7.1+cu128'
for _ in range(50):
 if Path('/proc/{identity['pid']}/status').read_text().split('State:',1)[1].split()[0]=='T':break
 time.sleep(.1)
else:raise RuntimeError('Source pause unproven')
p=Path({last['csv']!r}).parent/'checkpoints/latest_full.pt';p.resolve().relative_to(Path({src!r}).resolve())
blob=p.read_bytes();x=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
assert x.get('format')=='star-checkpoint' and x.get('rng') and x.get('optimizers') and x.get('replay') is not None
def finite(v):
 if torch.is_tensor(v):return bool(torch.isfinite(v).all())
 if isinstance(v,dict):return all(finite(w) for w in v.values())
 if isinstance(v,(list,tuple)):return all(finite(w) for w in v)
 return True
assert finite(x['model']) and finite(x['optimizers']) and x['extra']['seed']=={seed}
assert x['extra']['environment']=={'SafetyPointGoal1-v0' if job.startswith('point-goal-') else 'SafetyPointButton1-v0'!r}
n=int(x['counters']['env_steps']);c=float(x['counters']['cumulative_training_cost'])
assert 0<n<10000000 and 0<={last['steps']}-n<=5000 and math.isfinite(c) and 0<=c<={last['cost']}
print(json.dumps({{'source_checkpoint':str(p),'relative_checkpoint':str(p.relative_to(Path({src!r}))),
 'checkpoint_sha256':hashlib.sha256(blob).hexdigest(),'checkpoint_bytes':len(blob),'resume_steps':n,
 'resume_cumulative_training_cost':c,'full_replay_rng_optimizers_and_finite_state_verified':True}}))
''', True))
        state.update(full, phase='copying_paused_MPS_source', source_paused=True, source_retired=False,
                     original_evidence_preserved=True, last_logged_source_steps=last['steps'], last_logged_source_cost=last['cost'],
                     known_repeated_environment_steps_lower_bound=last['steps'] - full['resume_steps'],
                     known_additional_training_cost_lower_bound=last['cost'] - full['resume_cumulative_training_cost'],
                     unknown_unlogged_source_tail_not_zero=True)
        save(state_path, state)
        state['input_preservation'] = mps_helper('5080').preserve_input(h, job, full)
        save(state_path, state)
        h.py('5080', f"import time;from pathlib import Path;p=Path({src!r})/'exit_code.txt';p.rename(p.with_name('exit_code.before_{SIDE}_'+str(time.time_ns())+'.txt')) if p.exists() else None")
        receive = ('set -e; test ! -e ' + shlex.quote(dst) + '; point_move_stage=$(mktemp -d '
                   + shlex.quote(ROOTS['4080'] + '/runs/.point-gpu-handoff.XXXXXX')
                   + '); tar -xzf - --no-same-owner -C "$point_move_stage"; test ! -e ' + shlex.quote(dst)
                   + '; mv "$point_move_stage/' + job + '" ' + shlex.quote(dst) + '; rmdir "$point_move_stage"')
        send = 'tar --format=posix -czf - -C ' + shlex.quote(str(Path(src).parent)) + ' -- ' + shlex.quote(job)
        with subprocess.Popen([*h.OPTIONS, '5080', send], stdout=subprocess.PIPE) as producer:
            try:
                result = subprocess.run([*h.OPTIONS, '4080', receive], stdin=producer.stdout, timeout=3600)
            finally:
                producer.stdout.close()
                try: producer.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    producer.terminate()
                    try: producer.wait(timeout=15)
                    except subprocess.TimeoutExpired: producer.kill(); producer.wait()
            assert producer.returncode == 0 and result.returncode == 0
        receipt = {**state, 'target_checkpoint': dst + '/' + full['relative_checkpoint']}
        save(receipt_path, receipt)
        relocation = {'format': 'shapo-artifact-relocation-v1', 'job': job, 'original_container': src,
                      'original_host': '5080', 'target_host': '4080', 'scientific_protocol_changed': False,
                      'original_evidence_modified': False}
        h.py('4080', f'''
import hashlib,json,time
from pathlib import Path
d=Path({dst!r});p=d/{full['relative_checkpoint']!r}
assert hashlib.sha256(p.read_bytes()).hexdigest()=={full['checkpoint_sha256']!r}
for name in ['artifact_relocation.json','exit_code.txt','receipt.json']:
 p=d/name
 if p.exists():p.rename(d/(name+'.before_{SIDE}_'+str(time.time_ns())))
(d/'artifact_relocation.json').write_text({json.dumps(relocation,indent=2)+chr(10)!r})
(d/'handoff_{SIDE}_receipt.json').write_text({json.dumps(receipt,indent=2)+chr(10)!r})
p=Path({bootstrap!r})
p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()=={BOOTSTRAP_SHA!r}
else:p.write_text({(BASE / 'mps_training_bootstrap_20261020.py').read_text()!r})
''')
        # New private daemon only; never modify compute mode or any foreign daemon.
        state['target_capacity_at_launch'] = capacity(h)
        daemon = json.loads(h.py('4080', f'''
import json,os,subprocess
from pathlib import Path
pipe=Path({PIPES['4080']!r});logs=Path(str(pipe)+'_logs');created=False
if not pipe.exists():
 assert not logs.exists() and len(str(pipe/'control').encode())<100
 pipe.mkdir();logs.mkdir();created=True
 env=dict(os.environ,CUDA_VISIBLE_DEVICES={GPUS['4080']!r},CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(logs))
 subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True,timeout=20,cwd={ROOTS['4080']+'/code'!r})
print(json.dumps({{'pipe':str(pipe),'created_new_private_daemon':created,'compute_mode_unchanged':True}}))
'''))
        state['target_daemon'] = daemon
        prior = {r['csv'] for r in h.status('4080')['progress']}
        root = ROOTS['4080']
        shell = f'''#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY={PIPES['4080']} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={root}/code
cd {root}/code
set +e
taskset -c 0-15 {root}/env/bin/python {bootstrap} scripts/star/train_star.py --config {dst}/resolved_formal_config.yaml --seed {seed} --steps 10000000 --device cuda:0 --vector-envs 1 --batch-size 256 --replay-size 400000 --steps-per-epoch 5000 --log-dir {dst}/training --result {dst}/receipt.json --no-tensorboard --resume {shlex.quote(receipt['target_checkpoint'])} >{dst}/train_{SIDE}.log 2>&1
status=$?
printf '%s\\n' "$status" >{dst}/exit_code.txt
exit "$status"
'''
        launcher = root + '/deploy/resume_' + prefix + '.sh'
        h.py('4080', f'from pathlib import Path;Path({launcher!r}).write_text({shell!r})')
        state['target_capacity_immediately_before_client_launch'] = capacity(h)
        state.update(phase='target_MPS_starting_source_paused', target_session=session)
        save(state_path, state)
        launched = True
        h.ssh('4080', 'tmux new-session -d -s ' + shlex.quote(session) + ' ' + shlex.quote('bash ' + launcher))
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            target = h.status('4080')
            new = [r for r in target['progress'] if r['csv'] not in prior and r['steps'] >= full['resume_steps'] + 5000]
            if new and len(target['workers']) == 1: break
            if not target['workers'] and h.ssh('4080', 'test ! -e ' + shlex.quote(dst + '/exit_code.txt') + ' || cat ' + shlex.quote(dst + '/exit_code.txt')).strip():
                raise RuntimeError('Target terminal before first epoch; preserve failed evidence')
            time.sleep(30)
        else: raise RuntimeError('No first target epoch yet; inspect same live target, never duplicate')
        target_id = {**target['workers'][0], 'boot_id': target['boot_id']}
        assert target_id['boot_id'] == BOOTS['4080'] and target_id['device'] == 'cuda:0'
        proof = mps_helper('4080').mps_proof(h, target_id['pid'])
        native = load_module('watch_gpu_capacity.py', 'Point_GPU_first_epoch').probe('4080')
        w = next(w for w in native['training'] if w['pid'] == target_id['pid'])
        assert w['mps_gpu_uuid_verified'] == GPUS['4080'] and w['steps'] >= full['resume_steps'] + 5000
        native_path = BASE / (prefix + '_first_epoch_native_observation.json')
        save(native_path, native)
        state.update(phase='first_target_epoch_verified_retiring_source', target_identity=target_id, target_MPS_proof=proof,
                     target_first_epoch_native_observation=native_path.name, first_verified_target_steps=new[-1]['steps'],
                     first_verified_target_fps=new[-1]['fps'], first_verified_target_cost=new[-1]['cost'])
        save(state_path, state)
        # Source already uses the signal-safe synchronize-on-exit bootstrap.
        h.signal_source(identity, 'SIGTERM'); h.signal_source(identity, 'SIGCONT'); paused = False
        for _ in range(60):
            if not h.status('5080')['workers']: break
            time.sleep(1)
        else: raise RuntimeError('Source MPS retirement not proven; no force kill')
        retired = True
        archive = ROOTS['5080'] + '/recovery_backups/retired_gpu_workers/' + prefix + '_' + str(time.time_ns())
        retirement = json.loads(h.py('5080', f'''
import hashlib,json,time
from pathlib import Path
d=Path({src!r})
for _ in range(120):
 if (d/'exit_code.txt').exists():break
 time.sleep(.5)
else:raise RuntimeError('Original wrapper exit marker missing')
text=(d/'exit_code.txt').read_text().strip()
if text.endswith(chr(92)+'n'):text=text[:-2]
code=int(text);assert code!=0
a=Path({archive!r});a.parent.mkdir(parents=True,exist_ok=True)
assert not a.exists() and a.parent.stat().st_dev==d.stat().st_dev
d.rename(a)
assert hashlib.sha256((a/{full['relative_checkpoint']!r}).read_bytes()).hexdigest()=={full['checkpoint_sha256']!r}
print(json.dumps({{'source_wrapper_exit_code':code,'retired_source_archive':str(a),
 'retired_source_full_checkpoint':str(a/{full['relative_checkpoint']!r}),'retired_source_full_checkpoint_SHA_verified':True}}))
'''))
        state.update(retirement, phase='completed_first_MPS_epoch_source_retired', source_paused=False,
                     source_retired=True, source_canonical_path_vacated=True)
        save(state_path, state)
        h.py('4080', f'from pathlib import Path;Path({dst+"/handoff_"+SIDE+"_state.json"!r}).write_text({json.dumps(state,indent=2)+chr(10)!r})')
        return True
    except Exception as error:
        absent = False
        try:
            absent = not h.status('4080')['workers'] and h.ssh('4080', 'tmux has-session -t ' + shlex.quote(session) + ' >/dev/null 2>&1 && echo live || echo absent').strip() == 'absent'
        except (OSError, ValueError, subprocess.SubprocessError): pass
        if paused and absent:
            h.signal_source(identity, 'SIGCONT'); paused = False
        state.update(phase='needs_inspection_no_automatic_retry', source_paused=paused, source_retired=retired,
                     target_launch_requested=launched, target_absence_confirmed=absent, error=str(error))
        save(state_path, state)
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    run(parser.parse_args().job)
