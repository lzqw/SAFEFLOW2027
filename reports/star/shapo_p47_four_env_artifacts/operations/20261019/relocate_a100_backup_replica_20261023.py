#!/usr/bin/env python3
"""Move only our secondary 4080 checkpoint replicas to A100's other disk.

The original logical backup path remains a symlink. No training container,
scientific validator, trainer process, or foreign project is changed.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

BASE = Path(__file__).resolve().parent
RECEIPT = BASE / 'a100_backup_replica_storage_relocation_20261023.json'
UNIT = 'safeflow-recovered-checkpoint-backup.service'
LOGICAL = '/home/lzqw/SAFEFLOW2027_A100/recovery_backups/from_4080'
TARGET = '/storage_0/SAFEFLOW2027_lzqw_uid1007/recovery_backups/from_4080'

# Executed over SSH as the same unprivileged project owner. The only removal
# target is the explicitly named old replica directory, after full byte proof.
REMOTE = r'''
import hashlib,json,os,shutil,stat,time
from pathlib import Path
logical=Path('/home/lzqw/SAFEFLOW2027_A100/recovery_backups/from_4080')
namespace=Path('/storage_0/SAFEFLOW2027_lzqw_uid1007')
target=namespace/'recovery_backups/from_4080'
stage=target.with_name('from_4080.incoming_20261023')
old=logical.with_name('from_4080.relocated_verified_20261023')
receipt=Path('/home/lzqw/SAFEFLOW2027_A100/operations/a100_backup_replica_storage_relocation_20261023.json')
assert os.getuid()==1007
assert logical.is_dir() and not logical.is_symlink()
assert logical.stat().st_uid==1007
assert not target.exists() and not target.is_symlink()
assert not stage.exists() and not stage.is_symlink()
assert not old.exists() and not old.is_symlink()
assert not receipt.exists(), 'Do not replay a completed operation'
assert Path('/storage_0').is_mount()
assert os.stat('/storage_0').st_dev!=logical.stat().st_dev
if not namespace.exists():
    namespace.mkdir(mode=0o700)
assert not namespace.is_symlink() and namespace.stat().st_uid==1007
assert stat.S_IMODE(namespace.stat().st_mode)==0o700
target.parent.mkdir(mode=0o700,exist_ok=True)
assert not target.parent.is_symlink() and target.parent.stat().st_uid==1007

def inventory(root):
    rows=[];groups={};allowed={f'{slug}-{seed}' for slug in ('point-goal','point-button','ant','walker2d') for seed in range(2027,2032)}
    for entry in root.iterdir():
        assert entry.name in allowed and entry.is_dir() and not entry.is_symlink(), str(entry)
    for path in sorted(root.rglob('*')):
        st=path.lstat();rel=str(path.relative_to(root))
        assert st.st_uid==1007 and not stat.S_ISLNK(st.st_mode), rel
        if stat.S_ISDIR(st.st_mode):
            rows.append(dict(path=rel,kind='directory',mode=stat.S_IMODE(st.st_mode)))
            continue
        assert stat.S_ISREG(st.st_mode), rel
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024**2),b''):
                digest.update(block)
        now=path.stat()
        assert (st.st_ino,st.st_size,st.st_mtime_ns)==(now.st_ino,now.st_size,now.st_mtime_ns), 'Concurrent modification'
        rows.append(dict(path=rel,kind='file',bytes=st.st_size,mode=stat.S_IMODE(st.st_mode),mtime_ns=st.st_mtime_ns,sha256=digest.hexdigest()))
        groups.setdefault((st.st_dev,st.st_ino),[]).append(rel)
    hardlinks=sorted(sorted(group) for group in groups.values() if len(group)>1)
    return rows,hardlinks

before_home=shutil.disk_usage(logical).free
before_storage=shutil.disk_usage(target.parent).free
rows,links=inventory(logical)
size=sum(row.get('bytes',0) for row in rows)
assert before_storage>=size+100*1024**3, 'Preserve 100GiB for other data-disk users'
print(json.dumps(dict(phase='copying_owned_secondary_replicas',file_count=sum(r['kind']=='file' for r in rows),logical_bytes=size)),flush=True)
# cp -a preserves hard-link relationships as well as contents and timestamps.
import subprocess
subprocess.run(['cp','-a','--reflink=auto','--',str(logical),str(stage)],check=True)
copied,copied_links=inventory(stage)
assert copied==rows and copied_links==links, 'Keep original on verification failure'
assert inventory(logical)==(rows,links), 'Source changed; do not switch paths'
os.rename(stage,target)
os.rename(logical,old)
try:
    logical.symlink_to(target,target_is_directory=True)
except BaseException:
    os.rename(old,logical)
    raise
assert logical.resolve()==target and target.stat().st_dev!=old.stat().st_dev
record=dict(phase='copy_verified_logical_path_redirected_original_retained',
    logical_path=str(logical),physical_destination=str(target),original_retained_at=str(old),
    source_manifest=rows,hardlink_groups=links,logical_bytes=size,
    manifest_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
    project_free_before_bytes=before_home,data_disk_free_before_bytes=before_storage,
    trainers_signalled=0,foreign_projects_modified=False,scientific_parameters_changed=False,
    additional_scientific_rollouts=0,backup_payloads_lost=0,updated_at_unix=time.time())
receipt.write_text(json.dumps(record,indent=2)+'\n')
# This exact directory contains only verified replicas, not training containers.
# The complete copy remains at TARGET and through the original logical path.
assert old==Path('/home/lzqw/SAFEFLOW2027_A100/recovery_backups/from_4080.relocated_verified_20261023')
assert old.is_dir() and not old.is_symlink() and old.stat().st_uid==1007
assert inventory(old)==(rows,links)
shutil.rmtree(old)
record.update(phase='completed_owned_backup_replica_relocated_original_path_preserved',
    redundant_home_copy_removed=True,original_retained_at=None,
    project_free_after_bytes=shutil.disk_usage(logical.parent).free,
    data_disk_free_after_bytes=shutil.disk_usage(target).free,updated_at_unix=time.time())
receipt.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
'''


def identity(pid):
    proc=Path('/proc')/str(pid)
    args=proc.joinpath('cmdline').read_bytes().split(b'\0')
    assert str(BASE/'backup_two_hosts.py').encode() in args
    fields=proc.joinpath('stat').read_text().rsplit(') ',1)[1].split()
    return int(fields[19]),fields[0]


def main():
    assert not RECEIPT.exists(), 'One-shot operation; inspect instead of replaying'
    status=subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','--value'],text=True)
    pid=int(status.strip());ticks,state=identity(pid)
    assert state not in ('T','t')
    record=dict(phase='preparing',local_backup_pid=pid,local_backup_start_ticks=ticks,
        exact_source=LOGICAL,exact_destination=TARGET,trainers_signalled=0,
        other_projects_paused=False,additional_scientific_rollouts=0,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    held=False;stopped=False
    try:
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            assert identity(pid)[0]==ticks
            children=Path(f'/proc/{pid}/task/{pid}/children')
            if not children.read_text().strip():
                os.kill(pid,signal.SIGSTOP);held=True
                for _ in range(40):
                    if identity(pid)==(ticks,'T'):
                        break
                    time.sleep(.025)
                assert identity(pid)==(ticks,'T')
                if not children.read_text().strip():
                    break
                os.kill(pid,signal.SIGCONT);held=False
            time.sleep(1)
        else:
            raise RuntimeError('Backup writer busy; no data move attempted')
        # Stop only our idle backup service, so its restart loads the new exact
        # canonical-destination allowlist. No training/MPS unit is involved.
        subprocess.run(['systemctl','--user','stop',UNIT],check=True,timeout=30)
        stopped=True;held=False
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',
            '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tjucta-a100',
            'python3 -c '+shlex.quote(REMOTE)],text=True,capture_output=True,timeout=600)
        record.update(remote_returncode=result.returncode,remote_stderr=result.stderr)
        result.check_returncode()
        record['remote']=json.loads(result.stdout.splitlines()[-1])
        assert record['remote']['phase']=='completed_owned_backup_replica_relocated_original_path_preserved'
        record['phase']='completed'
    except BaseException as error:
        record.update(phase='needs_inspection_no_automatic_retry',error=str(error))
        raise
    finally:
        if stopped:
            subprocess.run(['systemctl','--user','start',UNIT],check=True,timeout=30)
            record['backup_service_restarted']=True
            record['new_backup_pid']=int(subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','--value'],text=True))
        elif held:
            assert identity(pid)[0]==ticks
            os.kill(pid,signal.SIGCONT)
        record['updated_at_unix']=time.time()
        RECEIPT.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k!='remote'}),flush=True)


if __name__=='__main__':
    main()
