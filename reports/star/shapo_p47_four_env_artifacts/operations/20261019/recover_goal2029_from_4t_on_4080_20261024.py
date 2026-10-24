#!/usr/bin/env python3
"""Restore one stopped fixed-budget cell from raw4T evidence, sharing4080.

No source GPU restart or signal. Observe a new epoch using the existing
half-hour monitor before confirming this continuation; never auto-relaunch.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

BASE=Path(__file__).resolve().parent
JOB='point-goal-2029'
SIDE='goal2029_from4t_to4080_20261024'
STATE=BASE/(SIDE+'_state.json')
SRC,DST='5080','4080'
COMMIT='62ca7399357c10645a955dcbead7ed2153a82c69'
ROOTS={SRC:'/home/autodrive/LZQW/SAFEFLOW2027',DST:'/home/carla/SAFEFLOW2027_4080'}
CODES={h:r+'/code' for h,r in ROOTS.items()}
PIPE=ROOTS[DST]+'/operations/mps_formal_4080_20261021'


def save(s):
    s['updated_at_unix']=time.time();tmp=STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(STATE)
    print(s['phase'],flush=True)


def module(path):
    spec=importlib.util.spec_from_file_location('archive_recovery_rpc',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def capacity(h,expected):
    return json.loads(h.py(DST,f'''
import csv,json,os,shutil,subprocess
from pathlib import Path
expected={expected!r}
def query(flag):return list(csv.reader(subprocess.check_output(['nvidia-smi',flag,'--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()))
g=next([v.strip() for v in r] for r in query('--query-gpu=index,uuid,memory.free,utilization.gpu,compute_mode') if int(r[0])==0)
assert int(g[2])>=10240 and int(g[3])<=85 and g[4]=='Default','Shared GPU has insufficient current headroom'
env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY={PIPE!r})
def mps(c):return subprocess.run(['nvidia-cuda-mps-control'],input=c+'\\n',env=env,text=True,capture_output=True,check=True,timeout=10).stdout
servers=[int(v) for v in mps('get_server_list').split() if v.isdigit()];assert len(servers)==1
clients=[int(v) for v in mps('get_client_list '+str(servers[0])).split() if v.isdigit()]
assert set(clients)==set(expected) and len(clients)<5,'Owned pool changed or has no replacement slot'
for pid,row in expected.items():
 p=Path('/proc',str(pid));a=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\\0') if v]
 assert 'scripts/star/train_star.py' in a
 assert a[a.index('--config')+1]=={ROOTS[DST]+'/runs/shapo_p47_four_env/runs/'!r}+row['job']+'/resolved_formal_config.yaml'
 assert int((p/'stat').read_text().rsplit(') ',1)[1].split()[19])==row['start_ticks']
mem=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
assert mem>=8*1024**3 and len(os.sched_getaffinity(0))>=16 and os.getloadavg()[0]<16
assert shutil.disk_usage({ROOTS[DST]!r}).free>=16*1024**3
print(json.dumps(dict(gpu_uuid=g[1],free_gpu_MiB=int(g[2]),total_gpu_utilization_percent=int(g[3]),
 owned_clients=clients,MPS_server_pid=servers[0],GPU_reserve_MiB=8192,client_allowance_MiB=2048,
 load=os.getloadavg(),memory_available_bytes=mem,foreign_process_actions=0)))
'''))


def finish(h,s):
    frame=json.loads((BASE/'gpu_capacity_state.json').read_text())
    if frame['observed_at_unix']<=s['launched_at_unix']:return False
    dst=frame['hosts'].get(DST,{})
    if dst.get('status')!='observed':return False
    rows=[w for w in dst.get('training',[]) if w['pid']==s['target_pid'] and w['job']==JOB]
    if not rows or (rows[0].get('steps') or 0)<s['resume_steps']+5000:return False
    w=rows[0]
    assert w['cuda_execution_backend']=='mps_verified' and w['mps_gpu_uuid_verified']==s['target_capacity']['gpu_uuid']
    current=h.status(DST);assert current['boot_id']==s['target_boot_id']
    live=next(r for r in current['workers'] if r['pid']==s['target_pid'])
    assert live['start_ticks']==s['target_start_ticks']
    epochs=[r for r in current['progress'] if Path(r['csv']).parent.name not in s['prior_logger_names'] and r['steps']>=s['resume_steps']+5000]
    assert epochs and not h.status(SRC)['workers'],'Source duplicate or missing target epoch: inspect, never restart'
    s.update(phase='completed_new_target_epoch_from_raw_archive_source_preserved',
        first_target_epoch=epochs[-1],first_target_worker=w,target_identity=live,
        confirming_scheduled_frame_unix=frame['observed_at_unix'])
    save(s)
    h.py(DST,f'from pathlib import Path; Path({h.container(DST)+"/handoff_"+SIDE+"_state.json"!r}).write_text({json.dumps(s,indent=2)+chr(10)!r})')
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--finish',action='store_true');args=parser.parse_args()
    helper=BASE/'rebalance_goal2030_a100_to_5080_20261020_0621.py'
    assert hashlib.sha256(helper.read_bytes()).hexdigest()=='690fb3391274fe30f08aabb1b91f8df246276547bac9b3bd21a3f3a2c4158323'
    h=module(helper);h.SRC,h.DST,h.JOB,h.ROOTS,h.CODES=SRC,DST,JOB,ROOTS,CODES
    if args.finish:
        s=json.loads(STATE.read_text());assert s.get('target_pid')
        print('Already confirmed' if s['phase'].startswith('completed_') else ('Confirmed' if finish(h,s) else 'Same target; await scheduled progress'),flush=True);return
    assert not STATE.exists(),'Inspect existing transaction instead of launching a duplicate'
    frame=json.loads((BASE/'gpu_capacity_state.json').read_text());assert 0<=time.time()-frame['observed_at_unix']<2100
    assert all(w['job']!=JOB for host in frame['hosts'].values() for w in host.get('training',[]))
    expected={w['pid']:dict(job=w['job'],start_ticks=w['start_ticks']) for w in frame['hosts'][DST]['training']}
    assert len(expected)<=4,'At most one cell replaces one of the former five owned slots'
    pin=json.loads((BASE/'remaining_goal_archive_inputs_20261023.json').read_text())
    item=next(r for r in pin['rows'] if r['job']==JOB)
    inventory=json.loads((BASE/'incident_5080_2334_20261022_checkpoint_inventory.json').read_text())
    original=next(r for r in inventory['rows'] if r['job']==JOB)
    source=h.status(SRC);assert source['boot_id']==inventory['boot_id'] and not source['workers']
    last=max(source['progress'],key=lambda r:r['mtime_ns'])
    assert last['steps']==original['last_logged_steps'] and last['cost']==original['last_logged_cumulative_cost']
    assert not h.status(DST)['workers']
    for host in (SRC,DST):
        assert h.ssh(host,'git -C '+CODES[host]+' rev-parse HEAD').strip()==COMMIT
        assert not h.ssh(host,'git -C '+CODES[host]+' status --porcelain').strip()
    h.py(SRC,f'import hashlib; from pathlib import Path; assert hashlib.sha256(Path({original["checkpoint"]!r}).read_bytes()).hexdigest()=={item["checkpoint_sha256"]!r}')
    cap=capacity(h,expected)
    versions={host:h.py(host,"import importlib.metadata; print(importlib.metadata.version('torch'))",True).strip() for host in (SRC,DST)}
    s=dict(phase='restoring_recorded_scientific_capsule',job=JOB,source_host=SRC,target_host=DST,
        source_commit=COMMIT,total_budget_steps=10000000,resume_steps=item['resume_steps'],
        counters=item['counters_from_existing_incident_inventory'],checkpoint_sha256=item['checkpoint_sha256'],
        last_logged_source_steps=original['last_logged_steps'],last_logged_source_cost=original['last_logged_cumulative_cost'],
        resume_cumulative_training_cost=original['counters']['cumulative_training_cost'],
        known_repeated_steps_lower_bound=original['known_repeated_steps_if_resumed'],
        known_extra_training_cost_lower_bound=original['known_additional_training_cost_if_resumed'],
        inventory_file='incident_5080_2334_20261022_checkpoint_inventory.json',
        unlogged_source_tail_unknown_not_zero=True,source_original_preserved_in_place=True,
        source_signals_sent=0,training_launched_on_5080=False,new_experiment_cells=0,
        scientific_parameters_changed=False,additional_evaluation_rollouts=0,other_projects_paused=False,
        extra_verification_model_loads=0,bitwise_equivalence_claimed=False,
        source_torch_version=versions[SRC],target_torch_version=versions[DST],
        expected_existing_clients=expected,target_capacity=cap,source_raw_manifest=item['manifest'],
        source_raw_manifest_sha256=item['manifest_sha256'],
        prior_logger_names=sorted({Path(r['csv']).parent.name for r in source['progress']}),
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(s)
    try:
        restored=json.loads(h.py(DST,f'''
import hashlib,json,os,shutil,stat
from pathlib import Path
d=Path({h.container(DST)!r});manifest=Path({item['manifest']!r});raw=manifest.read_bytes();m=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()=={item['manifest_sha256']!r}
assert m['origin']=='5080' and m['job']=={JOB!r} and m['source_commit']=={COMMIT!r}
assert d.is_dir() and not d.is_symlink()
blobs=Path('/home/carla/4T/SAFEFLOW2027_4080/scientific_evidence_archive/blobs')
backup=Path('/home/carla/4T/SAFEFLOW2027_4080/recovery_backups/{SIDE}_target_before_restore')
assert os.path.ismount('/home/carla/4T') and not backup.exists()
assert shutil.disk_usage(d).free>={item['archived_logical_bytes']}+16*1024**3+512*1024**2
known={{r['sha256'] for r in m['files']}}
cfg=next(r for r in m['files'] if r['path']=='resolved_formal_config.yaml')
assert hashlib.sha256((d/'resolved_formal_config.yaml').read_bytes()).hexdigest()==cfg['sha256']
assert (d/'source_commit.txt').read_text().strip()=={COMMIT!r}
old=[]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
for p in sorted(d.rglob('*')):
 st=p.lstat();assert not p.is_symlink()
 if p.is_dir():continue
 assert stat.S_ISREG(st.st_mode);sha=digest(p)
 if p.suffix in ('.pt','.csv'):assert sha in known,'Do not replace an independent old branch'
 old.append(dict(path=str(p.relative_to(d)),sha256=sha,bytes=st.st_size,mtime_ns=st.st_mtime_ns,mode=stat.S_IMODE(st.st_mode)))
assert shutil.disk_usage(blobs).free>={item['archived_logical_bytes']}+sum(r['bytes'] for r in old)+100*1024**3
backup.mkdir(mode=0o700,parents=True)
for r in old:
 p=d/r['path'];q=backup/r['path'];q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 assert digest(q)==r['sha256'] and digest(p)==r['sha256']
(backup/'OLD_TARGET_FILE_MANIFEST.json').write_text(json.dumps(old,indent=2)+'\\n')
redirect={{'artifact_relocation.json','receipt.json','exit_code.txt'}}
for name in redirect:
 p=d/name
 if p.exists():p.rename(d/(name+'.old_target_before_{SIDE}'))
copied=[];reused=[]
for r in m['files']:
 rel=Path(r['path']);assert not rel.is_absolute() and '..' not in rel.parts
 blob=blobs/r['sha256'];assert blob.is_file() and not blob.is_symlink() and blob.stat().st_size==r['bytes']
 p=d/(r['path']+'.source_before_{SIDE}' if r['path'] in redirect else r['path'])
 if p.exists() and digest(p)==r['sha256']:reused.append(r['path']);continue
 if p.exists() and rel.parts[0]=='training':raise AssertionError('Conflicting scientific history; preserve old backup and inspect')
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.incoming_{SIDE}')
 assert not tmp.exists();shutil.copyfile(blob,tmp);assert digest(tmp)==r['sha256']
 os.chmod(tmp,r['mode']);os.utime(tmp,ns=(r['mtime_ns'],r['mtime_ns']));os.replace(tmp,p);copied.append(r['path'])
checkpoint=d/{item['checkpoint_relative_path']!r};assert digest(checkpoint)=={item['checkpoint_sha256']!r}
relocation=dict(format='shapo-artifact-relocation-v1',job={JOB!r},original_host='5080',target_host='4080',
 original_container={item['source_container']!r},scientific_protocol_changed=False,original_evidence_modified=False)
(d/'artifact_relocation.json').write_text(json.dumps(relocation,indent=2)+'\\n')
print(json.dumps(dict(restored_scientific_files=len(m['files']),copied_files=copied,reused_identical_files=reused,
 old_target_backup=str(backup),old_target_manifest=old,raw_scope='All recorded scientific suffixes, not a complete filesystem clone',
 checkpoint=str(checkpoint),archive_resume_input=str(blobs/{item['checkpoint_sha256']!r}),
 primary_free_after_bytes=shutil.disk_usage(d).free)))
'''))
        s.update(phase='scientific_capsule_restored_no_training_yet',restoration=restored,
            target_checkpoint=restored['checkpoint'],archive_resume_input=restored['archive_resume_input']);save(s)
        cap=capacity(h,expected);assert not h.status(SRC)['workers'] and not h.status(DST)['workers']
        s['target_capacity_at_launch']=cap
        s['target_boot_id']=h.py(DST,"from pathlib import Path;print(Path('/proc/sys/kernel/random/boot_id').read_text().strip())").strip()
        launched=json.loads(h.py(DST,f'''
import json,os,subprocess
from pathlib import Path
root=Path({ROOTS[DST]!r});d=Path({h.container(DST)!r})
env=dict(os.environ,CUDA_VISIBLE_DEVICES={cap['gpu_uuid']!r},CUDA_MPS_PIPE_DIRECTORY={PIPE!r},
 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1',
 PYTHONDONTWRITEBYTECODE='1',PYTHONPATH={CODES[DST]!r})
a=['nice','-n','15',str(root/'env/bin/python'),str(root/'operations/mps_training_bootstrap_20261020.py'),
 'scripts/star/train_star.py','--config',str(d/'resolved_formal_config.yaml'),'--seed','2029','--steps','10000000',
 '--device','cuda:0','--vector-envs','1','--batch-size','256','--replay-size','400000','--steps-per-epoch','5000',
 '--log-dir',str(d/'training'),'--result',str(d/'receipt.json'),'--no-tensorboard','--resume',{s['target_checkpoint']!r}]
with (d/'train_{SIDE}.log').open('xb') as log:
 p=subprocess.Popen(a,cwd={CODES[DST]!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
ticks=int(Path('/proc',str(p.pid),'stat').read_text().rsplit(') ',1)[1].split()[19])
print(json.dumps(dict(pid=p.pid,start_ticks=ticks)))
'''))
        s.update(phase='same_target_running_await_scheduled_new_epoch',target_pid=launched['pid'],
            target_start_ticks=launched['start_ticks'],launched_at_unix=time.time());save(s)
        deadline=time.monotonic()+5400
        while time.monotonic()<deadline:
            if finish(h,s):return
            time.sleep(5)  # Local scheduled JSON only; no recurring SSH probes.
        s.update(phase='waiting_for_same_target_scheduled_epoch_not_failure');save(s)
    except Exception as error:
        s.update(phase='needs_inspection_no_automatic_relaunch',error=str(error));save(s);raise


if __name__=='__main__':main()
