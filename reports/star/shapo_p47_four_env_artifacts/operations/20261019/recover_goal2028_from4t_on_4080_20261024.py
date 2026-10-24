#!/usr/bin/env python3
"""Continue the last stopped existing cell in the free shared4080 slot.

Restore the pinned scientific capsule from4T to a new primary target.
Preserve the5080 original and the4T capsule. Use a12GiB post-copy primary
reserve for this one replacement slot; the previous16GiB admission rule is
not a scientific protocol or validator gate. Expected input is1.932GiB;
the retained buffer exceeds the whole input plus simultaneous checkpoint
temporary files for four owned Point clients. Shared-filesystem growth is
still uncertain and remains subject to the existing half-hour monitor.
No tensor is loaded outside the ordinary training resume.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

BASE=Path(__file__).resolve().parent
JOB='point-goal-2028'
SIDE='goal2028_from4t_to4080_20261024'
STATE=BASE/(SIDE+'_state.json')
SRC,DST='5080','4080'
ROOTS={SRC:'/home/autodrive/LZQW/SAFEFLOW2027',DST:'/home/carla/SAFEFLOW2027_4080'}
CODES={SRC:ROOTS[SRC]+'/code',DST:ROOTS[DST]+'/code'}
PIPE=ROOTS[DST]+'/operations/mps_formal_4080_20261021'
COMMIT='62ca7399357c10645a955dcbead7ed2153a82c69'
OPTIONS=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def save(state):
    state['updated_at_unix']=time.time()
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
    print(state['phase'],flush=True)


def capacity(h,expected,incoming_bytes=0):
    return json.loads(h.py(DST,f'''
import csv,json,os,shutil,subprocess
from pathlib import Path
expected={expected!r}
rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu,compute_mode','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines())
g=next([v.strip() for v in row] for row in rows if int(row[0])==0)
assert g[1]=='GPU-66ed7e12-a0c3-9273-463d-56a5e5f0a4f7'
assert int(g[2])>=10240 and int(g[3])<=85 and g[4]=='Default','Insufficient shared GPU headroom'
env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY={PIPE!r})
def control(c):return subprocess.run(['nvidia-cuda-mps-control'],input=c+'\\n',env=env,text=True,capture_output=True,check=True,timeout=10).stdout
servers=[int(v) for v in control('get_server_list').split() if v.isdigit()];assert len(servers)==1
clients=[int(v) for v in control('get_client_list '+str(servers[0])).split() if v.isdigit()]
assert set(clients)==set(expected) and len(clients)<=3,'Expected owned cohort changed; no launch'
for pid,row in expected.items():
 p=Path('/proc',str(pid));a=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\\0') if v]
 assert 'scripts/star/train_star.py' in a
 assert a[a.index('--config')+1]=={ROOTS[DST]+'/runs/shapo_p47_four_env/runs/'!r}+row['job']+'/resolved_formal_config.yaml'
 assert int((p/'stat').read_text().rsplit(') ',1)[1].split()[19])==row['start_ticks']
memory=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
assert memory>=8*1024**3 and len(os.sched_getaffinity(0))>=16 and os.getloadavg()[0]<16
free=shutil.disk_usage({ROOTS[DST]!r}).free
assert free>={incoming_bytes}+12*1024**3,'Preserve12GiB after the recorded capsule copy'
print(json.dumps(dict(gpu_uuid=g[1],physical_gpu_index=0,free_gpu_MiB=int(g[2]),total_gpu_utilization_percent=int(g[3]),
 owned_clients=clients,MPS_server_pid=servers[0],GPU_reserve_MiB=8192,client_allowance_MiB=2048,
 primary_free_bytes=free,minimum_post_copy_free_bytes=12*1024**3,incoming_logical_bytes={incoming_bytes},
 load=os.getloadavg(),memory_available_bytes=memory,foreign_process_actions=0)))
'''))


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--finish',action='store_true');args=parser.parse_args()
    helper=BASE/'rebalance_goal2030_a100_to_5080_20261020_0621.py'
    assert hashlib.sha256(helper.read_bytes()).hexdigest()=='690fb3391274fe30f08aabb1b91f8df246276547bac9b3bd21a3f3a2c4158323'
    h=module(helper,'recovery_rpc')
    h.SRC,h.DST,h.JOB,h.ROOTS,h.CODES=SRC,DST,JOB,ROOTS,CODES
    observer=module(BASE/'recover_goal2029_from_4t_on_4080_20261024.py','scheduled_epoch_observer')
    observer.SRC,observer.DST,observer.JOB,observer.STATE,observer.SIDE=SRC,DST,JOB,STATE,SIDE
    if args.finish:
        s=json.loads(STATE.read_text());assert s.get('target_pid')
        print('Already confirmed' if s['phase'].startswith('completed_') else ('Confirmed' if observer.finish(h,s) else 'Same target; await scheduled progress'),flush=True)
        return
    assert not STATE.exists(),'Inspect the same transaction; never auto-relaunch'
    frame=json.loads((BASE/'gpu_capacity_state.json').read_text())
    assert 0<=time.time()-frame['observed_at_unix']<2100
    assert all(w['job']!=JOB for host in frame['hosts'].values() for w in host.get('training',[]))
    gpu=next(g for g in frame['hosts'][DST]['gpus'] if g['index']==0)
    ids=set(gpu['safeflow_mps_client_pids'])
    expected={w['pid']:dict(job=w['job'],start_ticks=w['start_ticks']) for w in frame['hosts'][DST]['training'] if w['pid'] in ids}
    assert len(expected)==len(ids)<=3
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
    h.py(SRC,f'import hashlib;from pathlib import Path;assert hashlib.sha256(Path({original["checkpoint"]!r}).read_bytes()).hexdigest()=={item["checkpoint_sha256"]!r}')
    cap=capacity(h,expected,item['archived_logical_bytes'])
    versions={host:h.py(host,"import importlib.metadata;print(importlib.metadata.version('torch'))",True).strip() for host in (SRC,DST)}
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
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        placement_reason='Replace completed Button2030 in the fast4080 pool without using reserved5080; last stopped fixed-budget cell, not reward/cost selection',
        minimum_post_copy_free_bytes=12*1024**3,previous_admission_reserve_bytes=16*1024**3,
        engineering_admission_reserve_changed=True,scientific_gate_changed=False,
        reserve_rationale='1.932GiB recorded input;12GiB post-copy buffer exceeds that input plus four simultaneous243MB checkpoint staging files; immutable source/4T inputs retained and half-hour disk monitoring unchanged',
        extra_half_GiB_copy_slack_required=False)
    save(s)
    d=h.container(DST);stage=d+'.incoming_'+SIDE
    try:
        # There is no previous target branch to overwrite. Fail before copying
        # if any current canonical container or one-shot staging path exists.
        h.py(DST,f'from pathlib import Path;d=Path({d!r});p=Path({stage!r});assert not d.exists() and not d.is_symlink() and not p.exists();p.mkdir(mode=0o700)')
        producer=f'''
import hashlib,json,sys,tarfile
from pathlib import Path
p=Path({item['manifest']!r});raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=={item['manifest_sha256']!r}
m=json.loads(raw);assert m['origin']=='5080' and m['job']=={JOB!r} and m['source_commit']=={COMMIT!r}
blobs=Path('/home/carla/4T/SAFEFLOW2027_4080/scientific_evidence_archive/blobs')
assert len(m['files'])=={item['archived_raw_files']} and sum(r['bytes'] for r in m['files'])=={item['archived_logical_bytes']}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as out:
 for r in m['files']:
  rel=Path(r['path']);assert not rel.is_absolute() and '..' not in rel.parts
  blob=blobs/r['sha256'];assert blob.is_file() and not blob.is_symlink() and blob.stat().st_size==r['bytes']
  info=tarfile.TarInfo(r['path']);info.size=r['bytes'];info.mode=r['mode'];info.mtime=r['mtime_ns']//1000000000
  with blob.open('rb') as stream:out.addfile(info,stream)
'''
        raw=h.ssh('4080','python3 -c '+shlex.quote(f'import hashlib,json;from pathlib import Path;r=Path({item["manifest"]!r}).read_bytes();print(json.dumps(dict(sha256=hashlib.sha256(r).hexdigest(),manifest=json.loads(r))))'))
        envelope=json.loads(raw);assert envelope['sha256']==item['manifest_sha256']
        manifest=envelope['manifest']
        # Both endpoints are on4080. Keep the transfer on that host and make
        # pipeline failures visible; do not round-trip two gigabytes locally.
        pipeline=('set -o pipefail; nice -n 19 ionice -c 3 python3 -c '+shlex.quote(producer)
                  +' | nice -n 19 ionice -c 3 tar -xpf - --no-same-owner -C '+shlex.quote(stage))
        h.ssh(DST,'bash -c '+shlex.quote(pipeline),timeout=1200)
        restored=json.loads(h.py(DST,f'''
import hashlib,json,os,shutil,stat
from pathlib import Path
p=Path({stage!r});d=Path({d!r});rows={manifest['files']!r}
assert p.is_dir() and not p.is_symlink() and not d.exists()
assert {{str(v.relative_to(p)) for v in p.rglob('*') if v.is_file()}}=={{r['path'] for r in rows}}
for r in rows:
 f=p/r['path'];st=f.lstat();assert stat.S_ISREG(st.st_mode) and not f.is_symlink() and st.st_size==r['bytes']
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
 assert h.hexdigest()==r['sha256']
 os.chmod(f,r['mode']);os.utime(f,ns=(r['mtime_ns'],r['mtime_ns']))
assert (p/'source_commit.txt').read_text().strip()=={COMMIT!r}
for name in ('artifact_relocation.json','receipt.json','exit_code.txt'):
 f=p/name
 if f.exists():f.rename(p/(name+'.source_before_{SIDE}'))
relocation=dict(format='shapo-artifact-relocation-v1',job={JOB!r},original_host='5080',target_host={DST!r},
 original_container={item['source_container']!r},scientific_protocol_changed=False,original_evidence_modified=False)
(p/'artifact_relocation.json').write_text(json.dumps(relocation,indent=2)+'\\n')
assert shutil.disk_usage(p).free>=12*1024**3
assert p.parent==d.parent and p.stat().st_dev==p.parent.stat().st_dev
os.rename(p,d)
print(json.dumps(dict(restored_scientific_files=len(rows),copied_files=[r['path'] for r in rows],reused_identical_files=[],
 previous_target_absent=True,raw_scope='All recorded scientific suffixes, not a complete filesystem clone',
 checkpoint=str(d/{item['checkpoint_relative_path']!r}),archive_resume_input={item['checkpoint_blob']!r},
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
root=Path({ROOTS[DST]!r});d=Path({d!r})
env=dict(os.environ,CUDA_VISIBLE_DEVICES={cap['gpu_uuid']!r},CUDA_MPS_PIPE_DIRECTORY={PIPE!r},
 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1',
 PYTHONDONTWRITEBYTECODE='1',PYTHONPATH={CODES[DST]!r})
a=['nice','-n','15',str(root/'env/bin/python'),str(root/'operations/mps_training_bootstrap_20261020.py'),
 'scripts/star/train_star.py','--config',str(d/'resolved_formal_config.yaml'),'--seed','2028','--steps','10000000',
 '--device','cuda:0','--vector-envs','1','--batch-size','256','--replay-size','400000','--steps-per-epoch','5000',
 '--log-dir',str(d/'training'),'--result',str(d/'receipt.json'),'--no-tensorboard','--resume',{restored['checkpoint']!r}]
with (d/'train_{SIDE}.log').open('xb') as log:
 proc=subprocess.Popen(a,cwd={CODES[DST]!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
ticks=int(Path('/proc',str(proc.pid),'stat').read_text().rsplit(') ',1)[1].split()[19])
print(json.dumps(dict(pid=proc.pid,start_ticks=ticks)))
'''))
        s.update(phase='same_target_running_await_scheduled_new_epoch',target_pid=launched['pid'],
                 target_start_ticks=launched['start_ticks'],launched_at_unix=time.time());save(s)
        deadline=time.monotonic()+5400
        while time.monotonic()<deadline:
            if observer.finish(h,s):return
            time.sleep(5)  # Local scheduled JSON only; no repeated SSH sampling.
        s['phase']='waiting_for_same_target_scheduled_epoch_not_failure';save(s)
    except BaseException as error:
        s.update(phase='needs_inspection_no_automatic_relaunch',error=str(error));save(s)
        raise


if __name__=='__main__':main()
