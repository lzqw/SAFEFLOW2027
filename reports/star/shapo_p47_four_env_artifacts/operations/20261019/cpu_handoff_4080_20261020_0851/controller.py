#!/usr/bin/env python3
"""Relocate an existing CPU locomotion cell to available4080-host CPU capacity."""
import hashlib
import importlib.util
import json
import shlex
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
JOB, SEED, ENV = 'ant-2031', 2031, 'Ant-v4'
SIDE = 'cpu_handoff_4080_20261020_0851'
PREFIX = JOB + '_' + SIDE
STATE, RECEIPT, PLAN = (BASE / (PREFIX + suffix) for suffix in ('_state.json', '_receipt.json', '_plan.json'))
SESSION = 'safeflow_ant_2031_' + SIDE
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
HELPER = BASE / 'rebalance_goal2030_a100_to_5080_20261020_0621.py'
HELPER_SHA = '690fb3391274fe30f08aabb1b91f8df246276547bac9b3bd21a3f3a2c4158323'


def save(path, value):
    value['updated_at_unix'] = time.time()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps({'file': path.name, 'phase': value.get('phase'),
                      'updated_at_unix': value['updated_at_unix']}), flush=True)


def helpers():
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
    spec = importlib.util.spec_from_file_location('verified_cpu_handoff_rpc_helpers', HELPER)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    h.JOB, h.SRC, h.DST = JOB, 'tjucta-a100', '4080'
    h.ROOTS = {'tjucta-a100': '/home/lzqw/SAFEFLOW2027_A100', '4080': '/home/carla/SAFEFLOW2027_4080'}
    h.CODES = {'tjucta-a100': h.ROOTS[h.SRC] + '/code/training-62ca739',
               '4080': h.ROOTS[h.DST] + '/code'}
    return h


def capacity(h, boot):
    return json.loads(h.py(h.DST, f'''
import importlib.metadata,json,os,shutil,subprocess,time
from pathlib import Path
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={boot!r}
assert importlib.metadata.version('torch')=='2.7.1+cu118'
cores=len(os.sched_getaffinity(0));load=os.getloadavg()[0]
available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
free=shutil.disk_usage({h.ROOTS[h.DST]!r}).free
assert cores>=24 and load<cores*.5 and available>=8*1024**3 and free>=5*1024**3,'CPU/memory/disk headroom changed'
def sample():
 values=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]));return sum(values),values[3]
a=sample();started=time.monotonic();time.sleep(1);b=sample();seconds=time.monotonic()-started
idle=(b[1]-a[1])/(b[0]-a[0]);assert idle>=.4,'Actual CPU idle headroom insufficient'
tmux_pid=int(subprocess.check_output(['tmux','display-message','-p','#{{pid}}'],text=True).strip())
quotas=[];root=Path('/sys/fs/cgroup').resolve()
for pid in [os.getpid(),tmux_pid]:
 entry=next((x for x in Path('/proc',str(pid),'cgroup').read_text().splitlines() if x.startswith('0::')),None)
 assert entry is not None,'Unknown CPU quota policy'
 p=(root/entry.split('::',1)[1].lstrip('/')).resolve();assert p==root or root in p.parents
 while True:
  q=p/'cpu.max'
  if q.exists():
   fields=q.read_text().split()
   if fields[0]!='max':quotas.append(int(fields[0])/int(fields[1]))
  if p==root:break
  p=p.parent
quota=min(quotas) if quotas else None
assert quota is None or quota>=8,'Target tmux/observer CPU quota is too restrictive'
print(json.dumps({{'boot_id':{boot!r},'cpu_affinity_count':cores,'load_1m':load,
 'maximum_load_affinity_fraction':.5,'actual_cpu_idle_fraction':idle,'cpu_idle_sample_seconds':seconds,
 'memory_available_bytes':available,'project_free_bytes':free,
 'minimum_cpu_quota_cores_or_null':quota,'cpu_quota_scope':'observer_and_target_tmux_server_with_ancestors',
 'cpu_quota_checked_tmux_server_pid':tmux_pid,'target_device':'cpu','target_cuda_visible_devices':'-1',
 'GPU_capacity_not_claimed_or_used':True}}))
''', True))


def main():
    assert not STATE.exists() and not RECEIPT.exists() and not PLAN.exists(), 'Inspect prior transaction; never duplicate'
    h = helpers()
    before_file, current_file = 'capacity_progress_observation_20261020_0821.json', 'capacity_progress_observation_20261020_0851.json'
    before = json.loads((BASE / before_file).read_text())
    now = json.loads((BASE / current_file).read_text())
    expected = next(w for w in now['hosts'][h.SRC]['training'] if w['job'] == JOB)
    prior = next(w for w in before['hosts'][h.SRC]['training'] if w['job'] == JOB)
    assert all(expected[k] == prior[k] for k in ('pid', 'start_ticks', 'boot_id', 'device', 'progress_csv'))
    assert expected['device'] == 'cpu' and expected['pid'] == 14549 and expected['steps'] >= 4050000
    assert now['observed_at_unix'] - before['observed_at_unix'] >= 1800
    assert not now['hosts'][h.DST]['training']
    boot = '3b362c03-4ed3-4e03-aea4-cf7339356026'
    src, dst = h.container(h.SRC), h.container(h.DST)
    observed = h.status(h.SRC)
    assert len(observed['workers']) == 1 and observed['boot_id'] == expected['boot_id']
    identity = {**observed['workers'][0], 'boot_id': observed['boot_id']}
    assert all(identity[k] == expected[k] for k in ('pid', 'start_ticks', 'device', 'boot_id'))
    assert identity['cuda_visible_devices'] == '-1'
    for flag, value in {'--seed':'2031', '--steps':'5000000', '--device':'cpu', '--vector-envs':'1',
                        '--batch-size':'256', '--replay-size':'400000', '--steps-per-epoch':'5000'}.items():
        assert identity['args'][identity['args'].index(flag) + 1] == value
    assert not h.status(h.DST)['workers']
    h.ssh(h.DST, 'test ! -e ' + shlex.quote(dst))
    for host in (h.SRC, h.DST):
        assert h.ssh(host, 'git -C ' + shlex.quote(h.CODES[host]) + ' rev-parse HEAD').strip() == SOURCE
        assert not h.ssh(host, 'git -C ' + shlex.quote(h.CODES[host]) + ' status --porcelain').strip()
    initial = capacity(h, boot)
    plan = {'job': JOB, 'seed': SEED, 'environment': ENV, 'source_host': h.SRC, 'target_host': h.DST,
            'source_commit': SOURCE, 'total_training_budget': 5000000,
            'source_device': 'cpu', 'target_device': 'cpu', 'source_cuda_visible_devices': '-1',
            'target_cuda_visible_devices': '-1', 'expected_source_identity': expected,
            'expected_target_boot_id': boot, 'previous_observation_file': before_file,
            'current_observation_file': current_file, 'scientific_parameters_changed': False,
            'additional_evaluation': False, 'additional_experiment_cells': 0,
            'no_other_projects_stopped': True, 'GPU_capacity_not_claimed_or_used': True,
            'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rpc_helper_filename': HELPER.name, 'rpc_helper_sha256': HELPER_SHA,
            'requires_first_new_CPU_epoch_before_source_retirement': True}
    save(PLAN, plan)
    state = {**plan, 'plan': plan, 'source_identity': identity,
             'phase': 'preparing_CPU_to_CPU_handoff', 'target_capacity_before_pause': initial,
             'runtime_change': {'from_device':'cpu', 'to_device':'cpu', 'from_torch':'2.7.1+cu118',
                 'to_torch':'2.7.1+cu118', 'cuda_build_changed':False, 'bitwise_equivalence_claimed':False}}
    save(STATE, state)
    paused = retired = launched = False
    try:
        h.signal_source(identity, 'SIGSTOP'); paused = True
        last = next(r for r in h.status(h.SRC)['progress'] if r['csv'] == expected['progress_csv'])
        checkpoint = json.loads(h.py(h.SRC, f'''
import hashlib,json,math,time,torch
from pathlib import Path
torch.set_num_threads(1);assert str(torch.__version__)=='2.7.1+cu118'
for _ in range(50):
 if Path('/proc/{identity['pid']}/status').read_text().split('State:',1)[1].split()[0]=='T':break
 time.sleep(.1)
else:raise RuntimeError('Owned source CPU pause not confirmed')
d=Path({src!r});p=Path({expected['progress_csv']!r}).parent/'checkpoints/latest_full.pt'
p.resolve().relative_to(d.resolve());x=torch.load(p,map_location='cpu',weights_only=False)
assert x['replay'] is not None and x['rng'] is not None
assert x['extra']['seed']==2031 and x['extra']['environment']=='Ant-v4'
assert all(bool(torch.isfinite(v).all()) for v in x['model'].values() if torch.is_floating_point(v))
n=int(x['counters']['env_steps']);cost=float(x['counters']['cumulative_training_cost'])
assert 0<n<5000000 and 0<={last['steps']}-n<50000 and math.isfinite(cost) and 0<=cost<={last['cost']}
digest=hashlib.sha256()
with p.open('rb') as f:
 for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
print(json.dumps({{'source_checkpoint':str(p),'relative_checkpoint':str(p.relative_to(d)),
 'checkpoint_sha256':digest.hexdigest(),'checkpoint_bytes':p.stat().st_size,
 'resume_steps':n,'resume_cumulative_training_cost':cost,
 'full_replay_and_rng_present':True,'model_parameter_finiteness_verified':True}}))
''', True))
        state.update(checkpoint, phase='copying_paused_CPU_source', source_paused=True,
                     source_retired=False, original_evidence_preserved=True,
                     last_logged_source_steps=last['steps'], last_logged_source_cumulative_training_cost=last['cost'],
                     known_repeated_environment_steps_lower_bound=last['steps']-checkpoint['resume_steps'],
                     known_additional_training_cost_lower_bound=last['cost']-checkpoint['resume_cumulative_training_cost'],
                     unknown_unlogged_source_tail_not_zero=True)
        save(STATE, state)
        receive = ('set -e; test ! -e ' + shlex.quote(dst) + '; cpu_handoff_stage=$(mktemp -d '
                   + shlex.quote(h.ROOTS[h.DST] + '/runs/.ant2031-cpu-handoff.XXXXXX')
                   + '); tar -xzf - --no-same-owner -C "$cpu_handoff_stage"; test ! -e '
                   + shlex.quote(dst) + '; mv "$cpu_handoff_stage/' + JOB + '" '
                   + shlex.quote(dst) + '; rmdir "$cpu_handoff_stage"')
        send = 'tar --format=posix -czf - -C ' + shlex.quote(str(Path(src).parent)) + ' -- ' + shlex.quote(JOB)
        with subprocess.Popen([*h.OPTIONS, h.SRC, send], stdout=subprocess.PIPE) as producer:
            try: result = subprocess.run([*h.OPTIONS, h.DST, receive], stdin=producer.stdout, timeout=3600)
            finally:
                producer.stdout.close()
                try: producer.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    producer.terminate()
                    try: producer.wait(timeout=15)
                    except subprocess.TimeoutExpired: producer.kill(); producer.wait()
            assert producer.returncode == 0 and result.returncode == 0
        receipt = {**state, 'target_checkpoint': dst + '/' + checkpoint['relative_checkpoint']}
        save(RECEIPT, receipt)
        relocation = {'format':'shapo-artifact-relocation-v1', 'job':JOB, 'original_container':src,
                      'original_host':h.SRC, 'target_host':h.DST, 'scientific_protocol_changed':False,
                      'original_evidence_modified':False}
        h.py(h.DST, f'''
import hashlib,json,time
from pathlib import Path
d=Path({dst!r});p=d/{checkpoint['relative_checkpoint']!r};digest=hashlib.sha256()
with p.open('rb') as f:
 for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
assert digest.hexdigest()=={checkpoint['checkpoint_sha256']!r}
for name in ['artifact_relocation.json','exit_code.txt','receipt.json']:
 p=d/name
 if p.exists():p.rename(d/(name+'.before_{SIDE}_'+str(time.time_ns())))
(d/'artifact_relocation.json').write_text({json.dumps(relocation,indent=2)+chr(10)!r})
(d/'handoff_{SIDE}_receipt.json').write_text({json.dumps(receipt,indent=2)+chr(10)!r})
''')
        prior_csvs = {r['csv'] for r in h.status(h.DST)['progress']}
        root, code = h.ROOTS[h.DST], h.CODES[h.DST]
        shell = f'''#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={shlex.quote(code)}
cd {shlex.quote(code)}
set +e
{root}/env/bin/python scripts/star/train_star.py --config {dst}/resolved_formal_config.yaml --seed 2031 --steps 5000000 --device cpu --vector-envs 1 --batch-size 256 --replay-size 400000 --steps-per-epoch 5000 --log-dir {dst}/training --result {dst}/receipt.json --no-tensorboard --resume {shlex.quote(receipt['target_checkpoint'])} >{dst}/train_{SIDE}.log 2>&1
status=$?
printf '%s\\n' "$status" >{dst}/exit_code.txt
exit "$status"
'''
        launcher = root + '/deploy/resume_' + PREFIX + '.sh'
        h.py(h.DST, f'from pathlib import Path; Path({launcher!r}).write_text({shell!r})')
        state['target_capacity_at_launch'] = capacity(h, boot)
        launched = True
        h.ssh(h.DST, 'tmux new-session -d -s ' + shlex.quote(SESSION) + ' ' + shlex.quote('bash ' + launcher))
        state.update(phase='target_CPU_starting_source_CPU_paused', target_session=SESSION)
        save(STATE, state)
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            target = h.status(h.DST)
            new = [r for r in target['progress'] if r['csv'] not in prior_csvs
                   and r['steps'] >= checkpoint['resume_steps'] + 5000]
            if new and len(target['workers']) == 1: break
            if not target['workers'] and h.ssh(h.DST, 'test ! -e ' + shlex.quote(dst + '/exit_code.txt')
                    + ' || cat ' + shlex.quote(dst + '/exit_code.txt')).strip():
                raise RuntimeError('Target CPU became terminal before first epoch')
            time.sleep(30)
        else: raise RuntimeError('No new CPU epoch yet; inspect live target, do not duplicate')
        target_identity = target['workers'][0]
        assert target['boot_id'] == boot and target_identity['device'] == 'cpu'
        assert target_identity['cuda_visible_devices'] == '-1'
        state.update(phase='first_target_CPU_epoch_verified_retiring_source', target_identity=target_identity,
                     first_verified_target_steps=new[-1]['steps'], first_verified_target_fps=new[-1]['fps'],
                     first_verified_target_cost=new[-1]['cost'], first_verified_target_csv=new[-1]['csv'])
        save(STATE, state)
        h.signal_source(identity, 'SIGTERM'); h.signal_source(identity, 'SIGCONT'); paused = False
        for _ in range(30):
            if not h.status(h.SRC)['workers']: break
            time.sleep(1)
        else: raise RuntimeError('Source CPU retirement not confirmed')
        retired = True
        archive = h.ROOTS[h.SRC] + '/recovery_backups/retired_cpu_workers/' + PREFIX + '_' + str(time.time_ns())
        h.py(h.SRC, f'''
import time
from pathlib import Path
d=Path({src!r})
for _ in range(120):
 if (d/'exit_code.txt').is_file():break
 time.sleep(.5)
else:raise RuntimeError('Source wrapper exit marker not observed')
archive=Path({archive!r});archive.parent.mkdir(parents=True,exist_ok=True)
assert not archive.exists() and d.stat().st_dev==archive.parent.stat().st_dev
d.rename(archive)
''')
        state.update(phase='handoff_completed_first_CPU_epoch_verified', source_retired=True,
                     source_paused=False, source_canonical_path_vacated=True, retired_source_archive=archive)
        save(STATE, state)
        h.py(h.DST, f'from pathlib import Path; Path({dst+"/handoff_"+SIDE+"_state.json"!r}).write_text({json.dumps(state,indent=2)+chr(10)!r})')
    except Exception as error:
        live = None
        absent = not launched
        try:
            live = h.status(h.DST)['workers']
            absent = not live and h.ssh(h.DST, 'tmux has-session -t ' + shlex.quote(SESSION)
                    + ' >/dev/null 2>&1 && echo live || echo absent').strip() == 'absent'
        except (OSError, ValueError, subprocess.SubprocessError): pass
        if paused and absent:
            try: h.signal_source(identity, 'SIGCONT'); paused = False
            except Exception as resume_error: state['source_resume_error'] = str(resume_error)
        state.update(phase='needs_inspection_after_CPU_to_CPU_handoff_error', source_paused=paused,
                     source_retired=retired, target_workers=live, target_absence_confirmed=absent,
                     target_launch_requested=launched, error=str(error), automatic_retry=False)
        save(STATE, state)
        raise


if __name__ == '__main__':
    main()
