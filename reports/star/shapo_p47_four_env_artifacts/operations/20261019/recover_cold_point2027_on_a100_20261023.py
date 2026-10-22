"""Continue the stopped PointGoal2027/10M cell on A100, never on5080.

No source trainer is signalled, no new scientific cell/evaluation is added,
and a new target epoch plus CUDA/MPS membership is required for confirmation.
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
JOB = 'point-goal-2027'
SIDE = 'cold_point2027_5080_to_a100_20261023'
STATE = BASE / (SIDE + '_state.json')
INVENTORY = 'incident_5080_2334_20261022_checkpoint_inventory.json'
COMMIT = '62ca7399357c10645a955dcbead7ed2153a82c69'
SRC, DST = '5080', 'tjucta-a100'
ROOTS = {SRC: '/home/autodrive/LZQW/SAFEFLOW2027', DST: '/home/lzqw/SAFEFLOW2027_A100',
         '4080': '/home/carla/SAFEFLOW2027_4080'}
CODES = {h: r + ('/code/training-62ca739' if h == DST else '/code') for h, r in ROOTS.items()}
PIPE = ROOTS[DST] + '/operations/mps_formal_a100_gpu0_20261020'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def save(state):
    state['updated_at_unix'] = time.time()
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(STATE)
    print(state['phase'], flush=True)


def cold_capacity(h):
    """Memory-bounded admission of one stopped cell to an already shared GPU.

    Utilization is evidence, not an exclusive-occupancy requirement. Unlike a
    live-worker migration this cannot slow an existing source continuation:
    there is no source worker. Keep16GiB GPU reserve plus2GiB for this client.
    """
    return json.loads(h.py(DST, f'''
import csv,json,os,shutil,subprocess
from pathlib import Path
def query(flag):
 return list(csv.reader(subprocess.check_output(['nvidia-smi',flag,'--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()))
r=next([v.strip() for v in r] for r in query('--query-gpu=index,uuid,memory.free,utilization.gpu,compute_mode') if int(r[0])==0)
assert int(r[2])>=18432 and r[4]=='Default'
env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY={PIPE!r})
def mps(command):
 return subprocess.run(['nvidia-cuda-mps-control'],input=command+'\\n',env=env,text=True,capture_output=True,check=True,timeout=10).stdout
servers=[int(v) for v in mps('get_server_list').split() if v.isdigit()]
assert len(servers)==1
clients=[int(v) for v in mps('get_client_list '+str(servers[0])).split() if v.isdigit()]
assert not clients,'Only one new SAFEFLOW client is admitted on GPU0'
foreign=[];seen=False
for uuid,pid,exe in query('--query-compute-apps=gpu_uuid,pid,process_name'):
 if uuid.strip()!=r[1]:continue
 pid=int(pid.strip())
 if pid==servers[0]:assert 'nvidia-cuda-mps-server' in exe;seen=True;continue
 assert exe.strip() not in ('','N/A','Unknown')
 try:a=[v.decode(errors='replace') for v in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\\0') if v]
 except OSError:a=[]
 assert not ('scripts/star/train_star.py' in a and '--config' in a and a[a.index('--config')+1].startswith({ROOTS[DST]+'/runs/'!r})), 'Unexpected owned cohort on GPU0'
 foreign.append(pid)
assert seen
mem=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
assert mem>=8*1024**3 and len(os.sched_getaffinity(0))>=16 and os.getloadavg()[0]<16
assert shutil.disk_usage({ROOTS[DST]!r}).free>=16*1024**3
print(json.dumps(dict(gpu_uuid=r[1],owned_GPU_OS_pids=servers,foreign_pids=foreign,
 memory_free_MiB=int(r[2]),utilization_percent=int(r[3]),no_foreign_process_actions=True,
 admission_policy='one_existing_stopped_cell_shared_GPU0_memory_CPU_disk_bounded',
 GPU_reserve_mib=16384,client_allowance_mib=2048,load=os.getloadavg(),memory_available_bytes=mem)))
'''))


def finish(h, observer, state):
    current = h.status(DST)
    assert current['boot_id'] == state['target_boot_id']
    live = [w for w in current['workers'] if w['pid'] == state['target_pid']]
    assert len(live) == 1, 'Recorded target missing; inspect rather than relaunch'
    new = [r for r in current['progress'] if Path(r['csv']).parent.name not in state['prior_logger_names']
           and r['steps'] >= state['resume_steps'] + 5000]
    if not new:
        return False
    assert not h.status(SRC)['workers'], 'Unexpected source duplicate; inspect both placements'
    native = observer.probe(DST)
    worker = next(w for w in native['training'] if w['pid'] == state['target_pid'])
    assert worker['cuda_execution_backend'] == 'mps_verified'
    assert worker['mps_gpu_uuid_verified'] == state['target_capacity']['gpu_uuid']
    state.update(phase='completed_first_new_epoch_on_A100_source_cold_archive_preserved',
                 first_target_epoch=new[-1], first_target_worker=worker, target_identity=live[0])
    save(state)
    h.py(DST, f'from pathlib import Path; Path({h.container(DST)+"/handoff_"+SIDE+"_state.json"!r}).write_text({json.dumps(state,indent=2)+chr(10)!r})')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finish', action='store_true')
    args = parser.parse_args()
    helper = BASE / 'rebalance_goal2030_a100_to_5080_20261020_0621.py'
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == '690fb3391274fe30f08aabb1b91f8df246276547bac9b3bd21a3f3a2c4158323'
    h = module(helper, 'cold_point2027_rpc')
    h.SRC, h.DST, h.JOB, h.ROOTS, h.CODES = SRC, DST, JOB, ROOTS, CODES
    observer = module(BASE / 'watch_gpu_capacity.py', 'cold_point2027_observer')
    if args.finish:
        state = json.loads(STATE.read_text())
        assert state['job'] == JOB and state.get('target_pid')
        if state['phase'].startswith('completed_first_new_epoch'):
            print('Already completed; no action', flush=True)
        else:
            print('finished' if finish(h, observer, state) else 'Same target live; waiting', flush=True)
        return
    assert not STATE.exists(), 'Inspect existing transaction; never launch a duplicate'
    before = json.loads((BASE / 'a100_low_load_candidate_20261023.json').read_text())
    now = json.loads((BASE / 'gpu_capacity_state.json').read_text())
    g0 = next(g for g in before['gpus'] if g['index'] == 0)
    g1 = next(g for g in now['hosts'][DST]['gpus'] if g['index'] == 0)
    assert now['observed_at_unix'] - before['observed_at_unix'] >= 1800
    assert 0 <= time.time() - now['observed_at_unix'] < 2100
    assert g0['uuid'] == g1['uuid']
    assert not g1['safeflow_mps_client_pids']
    inventory = json.loads((BASE / INVENTORY).read_text())
    item = next(r for r in inventory['rows'] if r['job'] == JOB)
    original = h.status(SRC)
    assert original['boot_id'] == inventory['boot_id'] and not original['workers']
    assert not h.status(DST)['workers']
    src, dst = h.container(SRC), h.container(DST)
    h.ssh(DST, 'test ! -e ' + shlex.quote(dst))
    for host in [SRC, DST]:
        assert h.ssh(host, 'git -C ' + shlex.quote(CODES[host]) + ' rev-parse HEAD').strip() == COMMIT
        assert not h.ssh(host, 'git -C ' + shlex.quote(CODES[host]) + ' status --porcelain').strip()
    target_capacity = cold_capacity(h)
    # The user requires continued GPU sharing, not exclusive occupancy. Keep
    # the live memory/load admission guard and record other projects, but do
    # not require the same foreign PID set as an earlier half-hour sample.
    assert len(target_capacity['owned_GPU_OS_pids']) == 1, 'Only our idle MPS server should already be on GPU0'
    size = int(h.ssh(SRC, 'du -sb ' + shlex.quote(src)).split()[0])
    #20GiB remains the existing monitor's alert threshold, not a scientific
    #stop rule. This single1.26GB capsule plus rolling checkpoints fits with
    #a16GiB operational reserve; no archives or other projects are removed.
    h.py(DST, f'import shutil; assert shutil.disk_usage({ROOTS[DST]!r}).free>={size}+16*1024**3+512*1024**2')
    versions = {host: h.py(host, "import importlib.metadata; print(importlib.metadata.version('torch'))", True).strip()
                for host in [SRC, DST]}
    state = dict(job=JOB, source_host=SRC, target_host=DST, source_commit=COMMIT,
        total_budget_steps=10000000, new_experiment_cells=0, scientific_parameters_changed=False,
        additional_evaluation_rollouts=0, other_projects_paused=False, source_signals_sent=0,
        training_launched_on_5080=False, source_was_already_stopped=True,
        source_archive_preserved=False, target_launch_requested=False,
        inventory_file=INVENTORY, resume_steps=item['counters']['env_steps'],
        counters=item['counters'], checkpoint_sha256=item['checkpoint_sha256'],
        last_logged_source_steps=item['last_logged_steps'],
        last_logged_source_cost=item['last_logged_cumulative_cost'],
        resume_cumulative_training_cost=item['counters']['cumulative_training_cost'],
        known_repeated_steps_lower_bound=item['known_repeated_steps_if_resumed'],
        known_extra_training_cost_lower_bound=item['known_additional_training_cost_if_resumed'],
        unlogged_source_tail_unknown_not_zero=True, bitwise_equivalence_claimed=False,
        source_torch_version=versions[SRC], target_torch_version=versions[DST],
        phase='copying_stopped_full_container_without_using_source_GPU',
        minimum_post_copy_free_bytes=16*1024**3,unchanged_storage_alert_threshold_gib=20,
        target_capacity=target_capacity, source_container_bytes=size,
        headroom_before=before, headroom_qualification=g1,
        headroom_qualification_time=now['observed_at_unix'],
        prior_logger_names=[Path(r['csv']).parent.name for r in original['progress']],
        source_archive=ROOTS[SRC]+'/recovery_backups/retired_gpu_workers/'+JOB+'_'+SIDE,
        selection_reason='Shortest remaining stopped Point budget; no5080 GPU restart and no reward/cost-based selection',
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(state)
    try:
        send = 'nice -n 15 tar -cf - -C ' + shlex.quote(str(Path(src).parent)) + ' -- ' + shlex.quote(JOB)
        receive = ('set -e; test ! -e ' + shlex.quote(dst) + '; stage_dir=$(mktemp -d '
                   + shlex.quote(ROOTS[DST]+'/runs/.cold-point2027.XXXXXX')
                   + '); tar -xf - --no-same-owner -C "$stage_dir"; test ! -e '+shlex.quote(dst)
                   + '; mv "$stage_dir/'+JOB+'" '+shlex.quote(dst)+'; rmdir "$stage_dir"')
        with subprocess.Popen([*h.OPTIONS, SRC, send], stdout=subprocess.PIPE) as producer:
            try:
                copied = subprocess.run([*h.OPTIONS, DST, receive], stdin=producer.stdout, timeout=900)
            finally:
                producer.stdout.close()
                producer.wait(timeout=30)
            assert producer.returncode == copied.returncode == 0
        relative = Path(item['checkpoint']).relative_to(Path(src))
        checkpoint = str(Path(dst) / relative)
        state['target_checkpoint'] = checkpoint
        relocation = dict(format='shapo-artifact-relocation-v1',job=JOB,original_host=SRC,
            target_host=DST,original_container=src,scientific_protocol_changed=False,original_evidence_modified=False)
        state['target_boot_id'] = h.py(DST, "from pathlib import Path; print(Path('/proc/sys/kernel/random/boot_id').read_text().strip())").strip()
        h.py(DST, f'''
import hashlib
from pathlib import Path
d=Path({dst!r});p=Path({checkpoint!r})
assert hashlib.sha256(p.read_bytes()).hexdigest()=={item['checkpoint_sha256']!r}
for name in ['artifact_relocation.json','receipt.json','exit_code.txt']:
 p=d/name
 if p.exists():p.rename(d/(name+'.before_{SIDE}'))
(d/'artifact_relocation.json').write_text({json.dumps(relocation,indent=2)+chr(10)!r})
''')
        archive = '/home/carla/4T/SAFEFLOW2027_4080/runtime_resume_inputs_20261023/cold_point2027/latest_full.pt'
        h.py('4080', f'from pathlib import Path; p=Path({archive!r}); p.parent.mkdir(parents=True,exist_ok=True); assert not p.exists() and not Path(str(p)+".partial").exists(); assert p.parent.stat().st_dev!=Path({ROOTS["4080"]!r}).stat().st_dev')
        subprocess.run(['scp','-3','-o','BatchMode=yes','-o','ConnectTimeout=12',DST+':'+checkpoint,'4080:'+archive+'.partial'],check=True,timeout=600)
        h.py('4080', f'import hashlib; from pathlib import Path; p=Path({archive!r}); q=Path(str(p)+".partial"); assert hashlib.sha256(q.read_bytes()).hexdigest()=={item["checkpoint_sha256"]!r}; q.rename(p)')
        state.update(phase='copied_input_verified_and_pinned_on_4T',archive_resume_input=archive)
        save(state)
        capacity = cold_capacity(h)
        assert len(capacity['owned_GPU_OS_pids']) == 1, 'Our target cohort changed; inspect before admission'
        state['target_capacity_at_launch'] = capacity
        h.py(DST, f'import shutil; assert shutil.disk_usage({ROOTS[DST]!r}).free>=16*1024**3')
        assert not h.status(SRC)['workers'] and not h.status(DST)['workers']
        h.py(SRC, f'''
from pathlib import Path
d=Path({src!r});a=Path({state['source_archive']!r})
assert d.is_dir() and not d.is_symlink() and not a.exists()
a.parent.mkdir(parents=True,exist_ok=True)
assert d.stat().st_dev==a.parent.stat().st_dev
d.rename(a)
''')
        state.update(source_archive_preserved=True,target_launch_requested=True,
                     phase='source_cold_archive_preserved_target_launch_requested')
        save(state)
        state['target_pid'] = int(h.py(DST, f'''
import os,subprocess
from pathlib import Path
root=Path({ROOTS[DST]!r});d=Path({dst!r})
env=dict(os.environ,CUDA_VISIBLE_DEVICES={g1['uuid']!r},CUDA_MPS_PIPE_DIRECTORY={PIPE!r},
 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
 PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH={CODES[DST]!r})
a=['nice','-n','15',str(root/'env/bin/python'),str(root/'operations/mps_training_bootstrap_20261020.py'),
 'scripts/star/train_star.py','--config',str(d/'resolved_formal_config.yaml'),'--seed','2027',
 '--steps','10000000','--device','cuda:0','--vector-envs','1','--batch-size','256','--replay-size','400000',
 '--steps-per-epoch','5000','--log-dir',str(d/'training'),'--result',str(d/'receipt.json'),
 '--no-tensorboard','--resume',{checkpoint!r}]
with (d/'train_{SIDE}.log').open('xb') as log:
 p=subprocess.Popen(a,cwd={CODES[DST]!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
print(p.pid)
''').strip())
        state.update(phase='same_A100_target_running_no_5080_GPU_launch');save(state)
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            if finish(h, observer, state):
                return
            time.sleep(45)
        state.update(phase='waiting_for_same_live_target_epoch_not_failure');save(state)
    except Exception as error:
        state.update(phase='needs_inspection_no_automatic_relaunch',error=str(error))
        save(state)
        raise


if __name__ == '__main__':
    main()
