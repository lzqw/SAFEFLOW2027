#!/usr/bin/env python3
"""Put terminal assembly copies on A100 data disk; native sources stay put."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

import collect_results_data_cache_20261024 as collector

BASE=Path(__file__).resolve().parent
STATE=BASE/'terminal_result_cache_migration_20261024_state.json'
RECEIPT=BASE/'terminal_result_cache_migration_20261024_receipt.json'
UNIT='safeflow-recovered-result-collector.service'
FOREIGN={'ant-2027','walker2d-2027','ant-2028','walker2d-2029'}


def identity(pid):
    p=Path('/proc')/str(pid)
    a=p.joinpath('cmdline').read_bytes().split(b'\0')
    assert str(BASE/'collect_results.py').encode() in a
    f=p.joinpath('stat').read_text().rsplit(') ',1)[1].split()
    return int(f[19]),f[0]


def main():
    assert not STATE.exists() and not RECEIPT.exists(), 'Inspect one-shot migration instead of replaying'
    pid=int(subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','--value'],text=True))
    ticks,status=identity(pid);assert status not in ('T','t')
    before=json.loads(collector.STATE.read_text());assert set(before['imported'])==FOREIGN
    done=collector.completed('tjucta-a100',collector.A100_PRIMARY_RUNS,[2027,2028,2029,2030,2031])
    assert len(done)==8 and FOREIGN<=set(done), 'Reinspect changing terminal set before migration'
    record=dict(phase='waiting_for_own_collector_idle',source_run_root=collector.A100_PRIMARY_RUNS,
        target_run_root=collector.A100_RUNS,terminal_jobs=done,foreign_cache_jobs=sorted(FOREIGN),
        old_collector_pid=pid,old_collector_start_ticks=ticks,training_processes_signalled=0,
        other_projects_modified=False,scientific_parameters_changed=False,additional_evaluation_rollouts=0,
        model_or_tensor_loads=0,frozen_validator_modified=False,
        new_collector_sha256=hashlib.sha256((BASE/'collect_results_data_cache_20261024.py').read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():
        record['updated_at_unix']=time.time();STATE.write_text(json.dumps(record,indent=2)+'\n')
    save();held=False;stopped=False;activated=False
    try:
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            assert identity(pid)[0]==ticks
            children=Path(f'/proc/{pid}/task/{pid}/children')
            if not children.read_text().strip():
                os.kill(pid,signal.SIGSTOP);held=True
                for _ in range(40):
                    if identity(pid)==(ticks,'T'):break
                    time.sleep(.025)
                assert identity(pid)==(ticks,'T')
                if not children.read_text().strip():break
                os.kill(pid,signal.SIGCONT);held=False
            time.sleep(1)
        else:raise RuntimeError('Collector remains busy; no source cache released')
        subprocess.run(['systemctl','--user','stop',UNIT],check=True,timeout=30);stopped=True;held=False
        record['phase']='copying_terminal_caches_without_touching_native_sources';save()
        remote=f'''
import hashlib,json,os,shutil,stat,subprocess,time
from pathlib import Path
source=Path({collector.A100_PRIMARY_RUNS!r});target=Path({collector.A100_RUNS!r})
namespace=Path('/storage_0/SAFEFLOW2027_lzqw_uid1007')
jobs={done!r};foreign=set({sorted(FOREIGN)!r})
assert os.getuid()==1007 and os.path.ismount('/storage_0')
assert namespace.is_dir() and not namespace.is_symlink() and namespace.stat().st_uid==1007 and namespace.stat().st_mode&0o077==0
assert not target.exists() and not target.is_symlink()
def rows(root):
 out=[]
 for p in sorted(root.rglob('*')):
  st=p.lstat();rel=str(p.relative_to(root));assert not p.is_symlink()
  if p.is_dir():out.append(dict(path=rel,kind='directory',mode=stat.S_IMODE(st.st_mode)));continue
  assert stat.S_ISREG(st.st_mode) and st.st_uid==1007
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
  now=p.stat();assert (st.st_ino,st.st_size,st.st_mtime_ns)==(now.st_ino,now.st_size,now.st_mtime_ns)
  out.append(dict(path=rel,kind='file',bytes=st.st_size,mode=stat.S_IMODE(st.st_mode),mtime_ns=st.st_mtime_ns,sha256=h.hexdigest()))
 return out
def inactive(name):
 for proc in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=[v.decode(errors='replace') for v in proc.read_bytes().split(b'\\0') if v]
  except OSError:continue
  d=str(source/'runs'/name)
  if any(v.endswith('scripts/star/train_star.py') for v in a) and '--config' in a:assert str(Path(a[a.index('--config')+1]).parent)!=d
  if any(v.endswith('scripts/star/validate_shapo_p47_run.py') for v in a) and '--container' in a:assert a[a.index('--container')+1]!=d
expected={{}}
for name in jobs:
 inactive(name);d=source/'runs'/name;assert d.is_dir() and not d.is_symlink()
 expected[name]=rows(d)
total=sum(r.get('bytes',0) for records in expected.values() for r in records)
assert shutil.disk_usage(namespace).free>=total+100*1024**3
(target/'runs').mkdir(mode=0o700,parents=True)
out=[]
for name in jobs:
 inactive(name);src=source/'runs'/name;dst=target/'runs'/name
 subprocess.run(['cp','-a','--reflink=auto','--',str(src),str(dst)],check=True)
 assert rows(dst)==expected[name] and rows(src)==expected[name]
 annotation='preserved_existing_remote_collection_edge'
 if name not in foreign:
  p=dst/'artifact_relocation.json'
  if p.exists():p.rename(dst/'artifact_relocation.before_terminal_cache_20261024.json')
  relocation=dict(format='shapo-artifact-relocation-v1',job=name,original_container=str(src),original_host='tjucta-a100',target_host='tjucta-a100',scientific_protocol_changed=False,original_evidence_modified=False)
  p.write_text(json.dumps(relocation,indent=2)+'\\n');annotation='added_storage_copy_edge_primary_source_preserved'
 else:
  relocation=json.loads((dst/'artifact_relocation.json').read_text())
  assert relocation['original_host']=='4080' and relocation['target_host']=='tjucta-a100'
  assert relocation['original_container']=='/home/carla/SAFEFLOW2027_4080/runs/shapo_p47_four_env/runs/'+name
 out.append(dict(job=name,source=str(src),target=str(dst),original_files=expected[name],
  original_manifest_sha256=hashlib.sha256(json.dumps(expected[name],sort_keys=True).encode()).hexdigest(),
  annotation=annotation,primary_replica_released=False))
for name in ('launch_manifest.txt','source_commit.txt','formal_matrix_manifest.json','frozen_protocol.json'):
 p=source/name
 if p.is_file() and not p.is_symlink():shutil.copy2(p,target/name)
proof=dict(phase='terminal_cache_copies_verified_sources_retained',entries=out,
 source_run_root=str(source),target_run_root=str(target),copied_terminal_jobs=len(jobs),
 training_processes_signalled=0,additional_evaluation_rollouts=0,model_or_tensor_loads=0,
 frozen_validator_modified=False,scientific_parameters_changed=False,original_native_sources_retained=True,
 primary_filesystem_free_before_release_bytes=shutil.disk_usage(source).free,
 cache_filesystem_free_after_copy_bytes=shutil.disk_usage(target).free,updated_at_unix=time.time())
manifest=target/'terminal_cache_storage_migration_20261024.json'
manifest.write_text(json.dumps(proof,indent=2)+'\\n')
print(json.dumps(dict(remote_receipt=str(manifest),copied_terminal_jobs=len(jobs))))
'''
        result=json.loads(collector.ssh('tjucta-a100','taskset -c 16-27 nice -n 19 ionice -c 3 python3 -c '+shlex.quote(remote),timeout=1200).stdout)
        record['remote_receipt']=result['remote_receipt'];save()
        cache_done=collector.completed('tjucta-a100',collector.A100_RUNS,[2027,2028,2029,2030,2031])
        assert set(cache_done)==set(done),'Preserve primary replicas if new cache is not readable'
        originals=collector.completed('4080',collector.REMOTE_RUNS['4080'],[2027,2028,2029,2030,2031])
        assert FOREIGN<=set(originals),'Original foreign terminal sources must still exist'
        release=f'''
import hashlib,json,os,shutil,time
from pathlib import Path
source=Path({collector.A100_PRIMARY_RUNS!r});target=Path({collector.A100_RUNS!r})
p=Path({result['remote_receipt']!r});proof=json.loads(p.read_text())
assert proof['phase']=='terminal_cache_copies_verified_sources_retained'
foreign=set({sorted(FOREIGN)!r})
for row in proof['entries']:
 if row['job'] not in foreign:continue
 src=Path(row['source']);dst=Path(row['target'])
 assert src==source/'runs'/row['job'] and dst==target/'runs'/row['job']
 assert src.is_dir() and not src.is_symlink() and dst.is_dir() and not dst.is_symlink()
 for r in row['original_files']:
  if r['kind']=='directory':continue
  a=src/r['path'];b=dst/r['path']
  assert a.is_file() and not a.is_symlink() and b.is_file() and not b.is_symlink()
  assert hashlib.sha256(a.read_bytes()).hexdigest()==hashlib.sha256(b.read_bytes()).hexdigest()==r['sha256']
 for proc in Path('/proc').glob('[0-9]*/cmdline'):
  try:args=[v.decode(errors='replace') for v in proc.read_bytes().split(b'\\0') if v]
  except OSError:continue
  if any(v.endswith('scripts/star/train_star.py') for v in args) and '--config' in args:assert str(Path(args[args.index('--config')+1]).parent)!=str(src)
  if any(v.endswith('scripts/star/validate_shapo_p47_run.py') for v in args) and '--container' in args:assert args[args.index('--container')+1]!=str(src)
 assert src.parent==source/'runs' and src.name in foreign and src.stat().st_uid==1007
 shutil.rmtree(src);row['primary_replica_released']=True
proof.update(phase='terminal_cache_on_data_four_foreign_primary_replicas_released',
 primary_filesystem_free_after_release_bytes=shutil.disk_usage(source).free,
 native_primary_sources_moved=0,foreign_original_sources_moved=0,
 source_caches_released=4,updated_at_unix=time.time())
p.write_text(json.dumps(proof,indent=2)+'\\n');print(json.dumps(dict(phase=proof['phase'],primary_free_bytes=proof['primary_filesystem_free_after_release_bytes'])))
'''
        released=json.loads(collector.ssh('tjucta-a100','nice -n 19 ionice -c 3 python3 -c '+shlex.quote(release),timeout=600).stdout)
        record.update(phase='data_cache_verified_primary_replica_release_complete',release=released);save()
        subprocess.run(['scp','-q','-o','BatchMode=yes','tjucta-a100:'+result['remote_receipt'],str(RECEIPT)],check=True,timeout=60)
        subprocess.run(['systemctl','--user','daemon-reload'],check=True,timeout=30)
        subprocess.run(['systemctl','--user','start',UNIT],check=True,timeout=30);activated=True
        record.update(phase='completed_new_collector_started',new_collector_pid=int(subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','--value'],text=True)));save()
        print(json.dumps(dict(phase=record['phase'],cached_jobs=8,new_collector_pid=record['new_collector_pid'])),flush=True)
    except BaseException as error:
        record.update(phase='needs_inspection_no_automatic_retry',error=str(error));save();raise
    finally:
        if held:
            assert identity(pid)[0]==ticks;os.kill(pid,signal.SIGCONT)
        if stopped and not activated:
            # Loaded unit still has the old command unless daemon-reload ran.
            # Starting it preserves collection availability; neither version
            # signals training or silently overwrites existing cache targets.
            subprocess.run(['systemctl','--user','start',UNIT],check=True,timeout=30)


if __name__=='__main__':main()
