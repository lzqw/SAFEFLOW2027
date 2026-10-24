#!/usr/bin/env python3
"""Archive three obsolete 4080 export copies, not active/canonical checkpoints."""
import json
from pathlib import Path
import shlex
import subprocess
import time
import hashlib

BASE=Path(__file__).resolve().parent
STATE=BASE/'inactive_4080_export_offload_20261024_state.json'
RECEIPT=BASE/'inactive_4080_export_offload_20261024_receipt.json'
REMOTE_RECEIPT='/home/carla/4T/SAFEFLOW2027_4080/recovery_backups/inactive_exports_20261024/relocation_receipt.json'
REMOTE=r'''
import hashlib,json,os,shutil,stat,subprocess,time
from pathlib import Path
project=Path('/home/carla/SAFEFLOW2027_4080');source=project/'recovery_backups/export'
archive=Path('/home/carla/4T/SAFEFLOW2027_4080/recovery_backups/inactive_exports_20261024')
jobs=['point-button-2031','point-goal-2028','walker2d-2028']
assert os.getuid()==1000 and os.path.ismount('/home/carla/4T')
assert not archive.exists() and not archive.is_symlink()
assert source.resolve()==source and source.stat().st_dev!=Path('/home/carla/4T').stat().st_dev
def vacant(job):
 p=project/'runs/shapo_p47_four_env/runs'/job
 assert not p.exists() and not p.is_symlink(),'Canonical job returned; do not touch its exports'
 for proc in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=[v.decode(errors='replace') for v in proc.read_bytes().split(b'\0') if v]
  except (OSError,ProcessLookupError):continue
  if 'scripts/star/train_star.py' in a and '--config' in a:
   assert a[a.index('--config')+1]!=str(p/'resolved_formal_config.yaml')
def rows(root):
 out=[]
 for p in sorted(root.iterdir()):
  st=p.lstat();assert stat.S_ISREG(st.st_mode) and not p.is_symlink() and st.st_uid==1000
  h=hashlib.sha256()
  with p.open('rb') as stream:
   for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
  now=p.stat();assert (st.st_ino,st.st_size,st.st_mtime_ns)==(now.st_ino,now.st_size,now.st_mtime_ns)
  out.append(dict(name=p.name,bytes=st.st_size,mtime_ns=st.st_mtime_ns,mode=stat.S_IMODE(st.st_mode),sha256=h.hexdigest()))
 return out
for job in jobs:
 vacant(job);p=source/job;assert p.is_dir() and not p.is_symlink()
assert shutil.disk_usage('/home/carla/4T').free>102*1024**3
archive.mkdir(mode=0o700,parents=True)
receipt=archive/'relocation_receipt.json'
record=dict(phase='archiving_three_inactive_export_copies',jobs=jobs,entries=[],
 primary_free_before_bytes=shutil.disk_usage(project).free,
 training_processes_signalled=0,canonical_training_files_modified=0,active_export_files_modified=0,
 additional_evaluation_rollouts=0,model_or_tensor_loads=0,other_projects_modified=False,
 scientific_parameters_changed=False,original_bytes_retained_on_separate_4T=True)
def save():
 record['updated_at_unix']=time.time();tmp=receipt.with_suffix('.tmp')
 tmp.write_text(json.dumps(record,indent=2)+'\n');os.replace(tmp,receipt)
save()
for job in jobs:
 vacant(job);src=source/job;dst=archive/job;before=rows(src)
 subprocess.run(['cp','-a','--reflink=auto','--',str(src),str(dst)],check=True)
 assert rows(dst)==before and rows(src)==before
 entry=dict(job=job,original_export_directory=str(src),archive_directory=str(dst),files=before,
  manifest_sha256=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
  logical_bytes=sum(r['bytes'] for r in before),released_files=0)
 record['entries'].append(entry);save()
 for row in before:
  vacant(job);p=src/row['name'];st=p.lstat()
  assert stat.S_ISREG(st.st_mode) and st.st_size==row['bytes'] and st.st_mtime_ns==row['mtime_ns']
  assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
  p.unlink();entry['released_files']+=1
 assert not list(src.iterdir())
 pointer=dict(kind='historical_inactive_4080_export_copy_not_current_backup',archive_directory=str(dst),
  original_export_directory=str(src),manifest_sha256=entry['manifest_sha256'],
  note='Exact old bytes remain on4T. This directory stays on the primary filesystem for later fresh exports. Do not substitute these historical files for a newer active input or a missing original parent.')
 (src/'cold_export_before_final_import_20261024.json').write_text(json.dumps(pointer,indent=2)+'\n')
 save()
record.update(phase='three_inactive_export_copies_archived_and_primary_replicas_released',
 primary_free_after_bytes=shutil.disk_usage(project).free,
 original_export_parent_directories_preserved=True,distinct_original_bytes_removed_from_all_locations=0)
save();print(json.dumps(dict(phase=record['phase'],receipt=str(receipt),primary_free_after_bytes=record['primary_free_after_bytes'])),flush=True)
'''


def main():
    assert not STATE.exists() and not RECEIPT.exists(), 'Inspect existing one-shot operation; no blind replay'
    s=dict(phase='running',helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),started_at_unix=time.time())
    STATE.write_text(json.dumps(s,indent=2)+'\n')
    try:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','4080',
            'taskset -c 16-27 nice -n 19 ionice -c 3 python3 -c '+shlex.quote(REMOTE)],check=True,timeout=300)
        subprocess.run(['scp','-q','-o','BatchMode=yes','4080:'+REMOTE_RECEIPT,str(RECEIPT)],check=True,timeout=60)
        r=json.loads(RECEIPT.read_text());assert r['phase']=='three_inactive_export_copies_archived_and_primary_replicas_released'
        s.update(phase='completed',finished_at_unix=time.time())
    except BaseException as error:s.update(phase='needs_inspection_no_automatic_retry',error=str(error));raise
    finally:STATE.write_text(json.dumps(s,indent=2)+'\n')


if __name__=='__main__':main()
