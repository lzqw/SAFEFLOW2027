#!/usr/bin/env python3
"""Move only eight named cold archives to4T, preserving their original paths.

This includes one72-hour-cold legacy experiment directory, not a formal matrix source.
Parent directories stay on the project filesystem, so future same-FS
retirement renames continue working. No canonical training container is moved.
"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

BASE=Path(__file__).resolve().parent
STATE=BASE/'cold_a100_directory_offload_20261024_state.json'
RECEIPT=BASE/'cold_a100_directory_offload_20261024_receipt.json'
REMOTE_RECEIPT='/storage_0/SAFEFLOW2027_lzqw_uid1007/recovery_backups/cold_directories_20261024/relocation_receipt.json'
REMOTE=r'''
import hashlib,json,os,shutil,stat,subprocess,time
from pathlib import Path
root=Path('/home/lzqw/SAFEFLOW2027_A100')
archive=Path('/storage_0/SAFEFLOW2027_lzqw_uid1007/recovery_backups/cold_directories_20261024')
receipt=archive/'relocation_receipt.json'
names=[
 "recovery_backups/retired_cpu_workers/ant-2031_cpu_handoff_4080_20261020_0851_1792458448988080133",
 "recovery_backups/retired_gpu_workers/point-button-2028_a100_to_4080_point2028_20261022",
 "recovery_backups/retired_gpu_workers/point-button-2030_a100_to_4080_point2030_20261022",
 "recovery_backups/retired_duplicate_branches_20261022/ant-2028",
 "recovery_backups/retired_duplicate_branches_20261022/walker2d-2028",
 "recovery_backups/retired_duplicate_branches_20261022/walker2d-2029",
 "recovery_backups/historical_ant2029_4080_input_20261022",
 "runs/ant"
]
assert os.getuid()==root.stat().st_uid==1007 and os.path.ismount('/storage_0')
assert not archive.exists() and not archive.is_symlink(), 'Inspect existing one-shot archive instead of replaying'
assert root.stat().st_dev!=Path('/storage_0').stat().st_dev
sources=[root/name for name in names]
for src in sources:
 assert src.is_dir() and not src.is_symlink() and not src.parent.is_symlink()
 assert src.stat().st_uid==1007 and src.parent.stat().st_dev==root.stat().st_dev
 assert not src.with_name(src.name+'.relocating_20261024').exists()

def scan(root,with_hash=True):
 rows=[];inodes=set();cache={}
 for p in sorted(root.rglob('*')):
  st=p.lstat();rel=str(p.relative_to(root));assert st.st_uid==1007 and not stat.S_ISLNK(st.st_mode),str(p)
  if stat.S_ISDIR(st.st_mode):
   rows.append(dict(path=rel,kind='directory',mode=stat.S_IMODE(st.st_mode)));continue
  assert stat.S_ISREG(st.st_mode),str(p)
  key=(st.st_dev,st.st_ino);inodes.add(key)
  row=dict(path=rel,kind='file',bytes=st.st_size,mode=stat.S_IMODE(st.st_mode),mtime_ns=st.st_mtime_ns)
  if with_hash:
   if key not in cache:
    h=hashlib.sha256()
    with p.open('rb') as f:
     for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    cache[key]=h.hexdigest()
   row['sha256']=cache[key]
   now=p.stat();assert (now.st_ino,now.st_size,now.st_mtime_ns)==(st.st_ino,st.st_size,st.st_mtime_ns),'Source changed'
  rows.append(row)
 return rows,inodes

namespace=Path('/storage_0/SAFEFLOW2027_lzqw_uid1007')
assert namespace.is_dir() and not namespace.is_symlink() and namespace.stat().st_uid==1007
assert namespace.stat().st_mode & 0o077 == 0
legacy=root/'runs/ant'
assert max(p.stat().st_mtime for p in legacy.rglob('*') if p.is_file())<time.time()-72*3600
all_inodes=set()
for src in sources:all_inodes.update(scan(src,False)[1])

def no_active_writers():
 for proc in Path('/proc').glob('[0-9]*'):
  try:
   if proc.stat().st_uid!=1007:continue
   cwd=(proc/'cwd').resolve(strict=True)
   assert not any(cwd==s or s in cwd.parents for s in sources),'Process cwd uses a cold archive'
   a=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
   for arg in a:
    if arg.startswith('/'):
     p=Path(arg);assert not any(p==src or src in p.parents for src in sources),'A process still references this cold directory'
   if 'scripts/star/train_star.py' in a and '--config' in a and a[a.index('--config')+1].startswith(str(root)+'/runs/'):
    for flag in ('--resume','--log-dir','--config'):
     if flag in a:
      p=Path(a[a.index(flag)+1]);assert not any(p==s or s in p.parents for s in sources),'Live training uses a cold archive'
   for fd in (proc/'fd').iterdir():
    try:
     st=fd.stat()
     if (st.st_dev,st.st_ino) not in all_inodes:continue
     info=(proc/'fdinfo'/fd.name).read_text()
     flags=int(next(v.split()[1] for v in info.splitlines() if v.startswith('flags:')),8)
     assert flags&os.O_ACCMODE==os.O_RDONLY,'Open writer uses cold archive inode'
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue

no_active_writers()
size=sum(sum(r.get('bytes',0) for r in scan(src,False)[0]) for src in sources)
assert shutil.disk_usage('/storage_0').free>=size+100*1024**3
archive.mkdir(mode=0o700,parents=True)
record=dict(phase='copying_named_cold_directories',source_project=str(root),archive_root=str(archive),
 entries=[],primary_free_before_bytes=shutil.disk_usage(root).free,
 canonical_training_containers_moved=0,training_processes_signalled=0,foreign_projects_modified=False,
 model_or_tensor_loads=0,additional_evaluation_rollouts=0,scientific_parameters_changed=False,
 future_retirement_parent_directories_preserved=True,original_paths_preserved_with_same_content_links=True,
 historical_parent_gaps_repaired=False)
def save():
 record['updated_at_unix']=time.time();temp=receipt.with_suffix('.tmp')
 temp.write_text(json.dumps(record,indent=2)+'\n');os.replace(temp,receipt)
save()
for src,name in zip(sources,names):
 dst=archive/name;stage=dst.with_name(dst.name+'.incoming');hold=src.with_name(src.name+'.relocating_20261024')
 dst.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 assert not dst.exists() and not stage.exists() and not hold.exists()
 rows,_=scan(src);no_active_writers()
 subprocess.run(['cp','-a','--reflink=auto','--',str(src),str(stage)],check=True)
 copied,_=scan(stage);assert copied==rows,'Verified original remains on copy mismatch'
 assert scan(src)[0]==rows,'Source changed; do not switch paths'
 no_active_writers();os.rename(stage,dst);os.rename(src,hold)
 try:src.symlink_to(dst,target_is_directory=True)
 except BaseException:os.rename(hold,src);raise
 assert src.resolve()==dst and src.parent.stat().st_dev==root.stat().st_dev
 entry=dict(original_path=str(src),archive_path=str(dst),files=rows,
  file_manifest_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
  logical_bytes=sum(r.get('bytes',0) for r in rows),source_redundant_copy_retained_at=str(hold),
  phase='verified_archive_original_path_preserved')
 record['entries'].append(entry);save()
 assert hold.parent==root/Path(name).parent
 assert hold.is_dir() and not hold.is_symlink() and hold.stat().st_uid==1007
 assert hold.name==Path(name).name+'.relocating_20261024'
 assert scan(hold)[0]==rows
 shutil.rmtree(hold)
 entry.update(phase='relocated_primary_redundant_copy_released',source_redundant_copy_retained_at=None)
 record['primary_free_current_bytes']=shutil.disk_usage(root).free;save()
 print(json.dumps(dict(completed=str(src),primary_free_bytes=record['primary_free_current_bytes'])),flush=True)
record.update(phase='eight_cold_directories_relocated_all_original_paths_preserved',
 primary_free_after_bytes=shutil.disk_usage(root).free,archive_free_after_bytes=shutil.disk_usage(archive).free,
 free_counter_caveat='Filesystem free counters include concurrent unrelated filesystem activity; shared hardlinks may retain some original inodes.')
save();print(json.dumps(dict(phase=record['phase'],receipt=str(receipt),entries=len(record['entries']),
 primary_free_after_bytes=record['primary_free_after_bytes'])),flush=True)
'''


def main():
    assert not STATE.exists() and not RECEIPT.exists(), 'Inspect prior maintenance; never replay blindly'
    state=dict(phase='running_named_cold_directory_offload',helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        remote_receipt=REMOTE_RECEIPT,started_at_unix=time.time(),training_processes_signalled=0)
    STATE.write_text(json.dumps(state,indent=2)+'\n')
    try:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=15',
            '-o','ServerAliveCountMax=3','tjucta-a100','taskset -c 16-27 nice -n 19 ionice -c 3 python3 -c '+shlex.quote(REMOTE)],check=True,timeout=1200)
        subprocess.run(['scp','-q','-o','BatchMode=yes','tjucta-a100:'+REMOTE_RECEIPT,str(RECEIPT)],check=True,timeout=60)
        record=json.loads(RECEIPT.read_text());assert record['phase']=='eight_cold_directories_relocated_all_original_paths_preserved'
        state.update(phase='completed',finished_at_unix=time.time(),local_receipt=str(RECEIPT))
    except BaseException as error:
        state.update(phase='needs_inspection_no_automatic_retry',error=str(error));raise
    finally:STATE.write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
