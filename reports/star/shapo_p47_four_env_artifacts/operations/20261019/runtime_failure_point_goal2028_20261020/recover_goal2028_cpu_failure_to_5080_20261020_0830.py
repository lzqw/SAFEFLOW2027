#!/usr/bin/env python3
"""Recover one terminal CPU numerical failure on our existing GPU cohort."""
import hashlib
import importlib.util
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
JOB = 'point-goal-2028'
SIDE = 'cpu_failure_gpu_recovery_20261020_0830'
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
PREFIX = JOB + '_' + SIDE
STATE, RECEIPT = (BASE / (PREFIX + suffix) for suffix in ('_state.json', '_receipt.json'))
PLAN = BASE / (PREFIX + '_plan.json')
SESSION = 'safeflow_point_goal_2028_' + SIDE
HELPER = BASE / 'rebalance_goal2030_a100_to_5080_20261020_0621.py'
HELPER_SHA = '690fb3391274fe30f08aabb1b91f8df246276547bac9b3bd21a3f3a2c4158323'
HOST = Path('/home/lzqw/SAFEFLOW2027_artifacts/recovered_a100_4080/failed_runtime_attempts') / JOB / SIDE
FAILED_RUN = 'training/STAR-{SafetyPointGoal1-v0}/seed-2028-2026-10-19-23-39-24'


def save(path, value):
    value['updated_at_unix'] = time.time()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps({'file': path.name, 'phase': value.get('phase'),
                      'updated_at_unix': value['updated_at_unix']}), flush=True)


def load_helper():
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
    spec = importlib.util.spec_from_file_location('verified_gpu_capacity_helpers', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.JOB, helper.SRC, helper.DST = JOB, '4080', '5080'
    helper.ROOTS = {'4080': '/home/carla/SAFEFLOW2027_4080',
                    '5080': '/home/autodrive/LZQW/SAFEFLOW2027'}
    helper.CODES = {host: root + '/code' for host, root in helper.ROOTS.items()}
    return helper


def main():
    assert not STATE.exists() and not RECEIPT.exists() and not PLAN.exists(), 'Inspect prior attempt, never duplicate'
    assert not HOST.exists(), 'Preserved evidence already exists; inspect instead of overwriting'
    assert shutil.disk_usage(HOST.parent.parent).free > 1024**3
    h = load_helper()
    src, dst = h.container(h.SRC), h.container(h.DST)
    before = json.loads((BASE / 'capacity_progress_observation_20261020_0751.json').read_text())
    now = json.loads((BASE / 'capacity_progress_observation_20261020_0821.json').read_text())
    failed_identity = next(w for w in before['hosts']['4080']['training'] if w['job'] == JOB)
    assert failed_identity['pid'] == 667308 and failed_identity['device'] == 'cpu'
    assert not any(w['job'] == JOB for w in now['hosts']['4080']['training'])
    assert now['observed_at_unix'] - before['observed_at_unix'] >= 1800
    target = now['hosts']['5080']
    cohort = sorted(w['pid'] for w in target['training'] if w['device'] == 'cuda:0')
    assert len(cohort) == 7
    assert cohort == sorted(w['pid'] for w in before['hosts']['5080']['training'] if w['device'] == 'cuda:0')
    gpu = next(g for g in target['gpus'] if g['index'] == 0)
    assert not gpu['foreign_pids'] and gpu['free_mib'] >= 8192
    boot = {w['boot_id'] for w in target['training']}
    assert len(boot) == 1
    plan = {
        'job': JOB, 'seed': 2028, 'source_host': h.SRC, 'target_host': h.DST,
        'source_commit': SOURCE, 'total_training_budget': 10000000,
        'kind': 'terminal_cpu_numerical_failure_recovery_within_owned_gpu_cohort',
        'failed_source_identity': failed_identity, 'failed_source_run_relative': FAILED_RUN,
        'expected_target_boot_id': boot.pop(), 'target_physical_gpu_index': 0,
        'rebalance_qualification': {'uuid': gpu['uuid'], 'owned_gpu_pids': cohort,
            'previous_observation_file': 'capacity_progress_observation_20261020_0751.json',
            'current_observation_file': 'capacity_progress_observation_20261020_0821.json',
            'observed_at_unix': now['observed_at_unix'], 'foreign_pids': [],
            'spare_compute_claimed': False, 'expected_concurrency_after_recovery': 8},
        'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'capacity_helper_filename': HELPER.name, 'capacity_helper_sha256': HELPER_SHA,
        'scientific_parameters_changed': False, 'additional_evaluation': False,
        'additional_experiment_cells': 0, 'no_other_projects_stopped': True,
        'existing_target_cells_may_slow_down': True,
        'requires_first_new_target_epoch_before_archiving_failed_source': True,
    }
    assert not h.status(h.SRC)['workers'] and not h.status(h.DST)['workers']
    h.ssh(h.DST, 'test ! -e ' + shlex.quote(dst))
    for host in (h.SRC, h.DST):
        assert h.ssh(host, 'git -C ' + shlex.quote(h.CODES[host]) + ' rev-parse HEAD').strip() == SOURCE
        assert not h.ssh(host, 'git -C ' + shlex.quote(h.CODES[host]) + ' status --porcelain').strip()
    capacity = h.capacity(plan)
    save(PLAN, plan)
    state = {'job': JOB, 'seed': 2028, 'source_host': h.SRC, 'target_host': h.DST,
             'source_commit': SOURCE, 'plan': plan, 'controller_sha256': plan['controller_sha256'],
             'phase': 'verifying_terminal_source_and_full_input',
             'scientific_parameters_changed': False, 'additional_evaluation': False,
             'total_training_budget': 10000000, 'target_capacity_before_copy': capacity,
             'runtime_change': {'from_device': 'cpu', 'to_device': 'cuda:0',
                 'from_torch': '2.7.1+cu118', 'to_torch': '2.7.1+cu128',
                 'cuda_build_changed': True, 'bitwise_equivalence_claimed': False}}
    save(STATE, state)
    launched = False
    try:
        proof = json.loads(h.py(h.SRC, f'''
import csv,hashlib,json,math,torch
from pathlib import Path
torch.set_num_threads(1)
assert str(torch.__version__)=='2.7.1+cu118'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={failed_identity['boot_id']!r}
d=Path({src!r});run=d/{FAILED_RUN!r}
assert (d/'exit_code.txt').read_text().strip()=='1'
stop=json.loads((run/'artifacts/engineering_hard_stop.json').read_text())
summary=json.loads((run/'artifacts/training_summary.json').read_text())
assert stop['category']=='nan_or_inf' and stop['reason']=='non-finite actor gradient norm'
assert summary['status']=='hard_stopped' and summary['seed']==2028
assert summary['environment']=='SafetyPointGoal1-v0'
assert summary['env_steps']==stop['env_steps']==3138661
checkpoint=run/'checkpoints/latest_full.pt'
x=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert x['format']=='star-checkpoint' and x['replay'] is not None and x['rng'] is not None
assert x['extra']['seed']==2028 and x['extra']['environment']=='SafetyPointGoal1-v0'
assert all(bool(torch.isfinite(v).all()) for v in x['model'].values() if torch.is_floating_point(v))
n=int(x['counters']['env_steps']);cost=float(x['counters']['cumulative_training_cost'])
assert 0<n<summary['env_steps']<10000000 and summary['env_steps']-n<50000
assert math.isfinite(cost) and 0<=cost<=summary['cumulative_training_cost']
with (run/'progress.csv').open() as f:
 for row in csv.DictReader(f):pass
assert int(float(row['TotalEnvSteps']))==3135000
files=[checkpoint,run/'artifacts/engineering_hard_stop.json',run/'artifacts/training_summary.json',
 run/'progress.csv',run/'config.json',run/'artifacts/run_manifest.json',
 d/'train_4080_gpu_cpu_rebalance_20261019_2330.log',d/'exit_code.txt']
artifacts=[]
for p in files:
 digest=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
 artifacts.append({{'source_path':str(p),'relative_path':str(p.relative_to(d)),
  'sha256':digest.hexdigest(),'bytes':p.stat().st_size}})
print(json.dumps({{'native_hard_stop':stop,'native_training_summary':summary,
 'source_exit_code':1,'source_workers_absent':True,
 'source_checkpoint':str(checkpoint),'relative_checkpoint':str(checkpoint.relative_to(d)),
 'checkpoint_sha256':artifacts[0]['sha256'],'resume_steps':n,
 'resume_cumulative_training_cost':cost,'full_replay_and_rng_present':True,
 'model_parameter_finiteness_verified':True,'failed_at_steps':summary['env_steps'],
 'failed_summary_cumulative_training_cost':summary['cumulative_training_cost'],
 'known_repeated_environment_steps':summary['env_steps']-n,
 'known_additional_training_cost':summary['cumulative_training_cost']-cost,
 'source_last_logged_steps':3135000,'source_artifacts':artifacts,
 'this_failure_tail_has_native_exact_counters':True,
 'older_unknown_historical_tails_not_assumed_zero':True}}))
''', True))
        assert not h.status(h.SRC)['workers']
        state.update(proof, phase='copying_terminal_source', original_evidence_preserved=True)
        save(STATE, state)
        receive = ('set -e; test ! -e ' + shlex.quote(dst) + '; recovery_stage=$(mktemp -d '
                   + shlex.quote(h.ROOTS[h.DST] + '/runs/.goal2028-recovery.XXXXXX')
                   + '); tar -xzf - --no-same-owner -C "$recovery_stage"; test ! -e '
                   + shlex.quote(dst) + '; mv "$recovery_stage/' + JOB + '" '
                   + shlex.quote(dst) + '; rmdir "$recovery_stage"')
        send = 'tar --format=posix -czf - -C ' + shlex.quote(str(Path(src).parent)) + ' -- ' + shlex.quote(JOB)
        with subprocess.Popen([*h.OPTIONS, h.SRC, send], stdout=subprocess.PIPE) as producer:
            try:
                result = subprocess.run([*h.OPTIONS, h.DST, receive], stdin=producer.stdout, timeout=3600)
            finally:
                producer.stdout.close()
                try: producer.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    producer.terminate()
                    try: producer.wait(timeout=15)
                    except subprocess.TimeoutExpired: producer.kill(); producer.wait()
            assert producer.returncode == 0 and result.returncode == 0
        receipt = {**state, 'target_checkpoint': dst + '/' + proof['relative_checkpoint']}
        save(RECEIPT, receipt)
        relocation = {'format': 'shapo-artifact-relocation-v1', 'job': JOB,
                      'original_container': src, 'original_host': h.SRC, 'target_host': h.DST,
                      'scientific_protocol_changed': False, 'original_evidence_modified': False}
        h.py(h.DST, f'''
import hashlib,json,time
from pathlib import Path
d=Path({dst!r});p=d/{proof['relative_checkpoint']!r};digest=hashlib.sha256()
with p.open('rb') as f:
 for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
assert digest.hexdigest()=={proof['checkpoint_sha256']!r}
for name in ['artifact_relocation.json','exit_code.txt','receipt.json']:
 p=d/name
 if p.exists():p.rename(d/(name+'.before_{SIDE}_'+str(time.time_ns())))
(d/'artifact_relocation.json').write_text({json.dumps(relocation,indent=2)+chr(10)!r})
(d/'handoff_{SIDE}_receipt.json').write_text({json.dumps(receipt,indent=2)+chr(10)!r})
''')
        prior = {r['csv'] for r in h.status(h.DST)['progress']}
        root, code = h.ROOTS[h.DST], h.CODES[h.DST]
        shell = f'''#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={shlex.quote(code)}
cd {shlex.quote(code)}
set +e
{root}/env/bin/python scripts/star/train_star.py --config {dst}/resolved_formal_config.yaml --seed 2028 --steps 10000000 --device cuda:0 --vector-envs 1 --batch-size 256 --replay-size 400000 --steps-per-epoch 5000 --log-dir {dst}/training --result {dst}/receipt.json --no-tensorboard --resume {shlex.quote(receipt['target_checkpoint'])} >{dst}/train_{SIDE}.log 2>&1
status=$?
printf '%s\\n' "$status" >{dst}/exit_code.txt
exit "$status"
'''
        launcher = root + '/deploy/resume_' + PREFIX + '.sh'
        h.py(h.DST, f'from pathlib import Path; Path({launcher!r}).write_text({shell!r})')
        assert not h.status(h.SRC)['workers'] and not h.status(h.DST)['workers']
        state['target_capacity_at_launch'] = h.capacity(plan)
        launched = True
        h.ssh(h.DST, root + '/env/bin/tmux new-session -d -s ' + shlex.quote(SESSION) + ' ' + shlex.quote('bash ' + launcher))
        state.update(phase='gpu_target_starting_failed_cpu_not_restarted', target_session=SESSION)
        save(STATE, state)
        HOST.mkdir(parents=True)
        copied = []
        for artifact in proof['source_artifacts']:
            destination = HOST / artifact['relative_path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                            h.SRC + ':' + shlex.quote(artifact['source_path']), str(destination)],
                           check=True, timeout=1800)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            assert digest == artifact['sha256'] and destination.stat().st_size == artifact['bytes']
            copied.append({**artifact, 'host_path': str(destination)})
        manifest = {'job': JOB, 'source_input_sha256': proof['checkpoint_sha256'],
                    'files': copied, 'raw_native_bytes_preserved': True}
        save(HOST / 'host_artifact_manifest.json', manifest)
        save(BASE / (PREFIX + '_host_manifest.json'), manifest)
        state['host_native_evidence_and_full_input_preserved'] = True
        save(STATE, state)
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            live = h.status(h.DST)
            new = [r for r in live['progress'] if r['csv'] not in prior
                   and r['steps'] >= proof['resume_steps'] + 5000]
            if new and len(live['workers']) == 1: break
            if not live['workers'] and h.ssh(h.DST, 'test ! -e ' + shlex.quote(dst + '/exit_code.txt')
                    + ' || cat ' + shlex.quote(dst + '/exit_code.txt')).strip():
                raise RuntimeError('GPU target terminal before first new epoch; inspect, do not duplicate')
            time.sleep(30)
        else: raise RuntimeError('First GPU epoch not observed; inspect live worker, never duplicate')
        identity = live['workers'][0]
        assert identity['device'] == 'cuda:0' and identity['cuda_visible_devices'] == '0'
        assert not h.status(h.SRC)['workers']
        h.py(h.DST, f'''
import csv,subprocess
rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())
assert any(uuid.strip()=={gpu['uuid']!r} and int(pid.strip())=={identity['pid']} for uuid,pid in rows)
''')
        archive = h.ROOTS[h.SRC] + '/recovery_backups/retired_cpu_workers/' + PREFIX + '_' + str(time.time_ns())
        h.py(h.SRC, f'''
from pathlib import Path
d=Path({src!r});assert (d/'exit_code.txt').read_text().strip()=='1'
archive=Path({archive!r});archive.parent.mkdir(parents=True,exist_ok=True)
assert not archive.exists();d.rename(archive)
''')
        state.update(phase='recovery_completed_first_gpu_epoch_verified',
                     target_identity=identity, first_verified_target_steps=new[-1]['steps'],
                     first_verified_target_fps=new[-1]['fps'], first_verified_target_cost=new[-1]['cost'],
                     first_verified_target_csv=new[-1]['csv'], target_gpu_uuid_verified=gpu['uuid'],
                     failed_source_remains_terminal=True, failed_cpu_not_restarted=True,
                     failed_source_archive=archive, source_canonical_path_vacated=True)
        save(STATE, state)
        h.py(h.DST, f'from pathlib import Path; Path({dst+"/handoff_"+SIDE+"_state.json"!r}).write_text({json.dumps(state,indent=2)+chr(10)!r})')
    except Exception as error:
        state.update(phase='needs_inspection_after_cpu_failure_gpu_recovery_error',
                     error=str(error), target_launch_requested=launched,
                     failed_cpu_not_restarted=True, automatic_retry=False)
        save(STATE, state)
        raise


if __name__ == '__main__':
    main()
