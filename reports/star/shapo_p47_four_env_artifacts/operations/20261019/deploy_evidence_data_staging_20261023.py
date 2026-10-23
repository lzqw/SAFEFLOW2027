#!/usr/bin/env python3
"""Atomically change only remote raw-archive temporary snapshot storage."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

BASE=Path(__file__).resolve().parent
SOURCE=BASE/'archive_scientific_evidence_data_staging_20261023.py'
RECEIPT=BASE/'evidence_data_staging_deployment_20261023.json'
OLD='1a7836ea0f36642a0c561c2ebd23f4c41970ba769a09690c13032db2b069fa11'
TARGETS={
    '4080':('/home/carla/SAFEFLOW2027_4080','/home/carla/4T',
            '/home/carla/4T/SAFEFLOW2027_4080/scientific_evidence_source_staging'),
    'tjucta-a100':('/home/lzqw/SAFEFLOW2027_A100','/storage_0',
            '/storage_0/SAFEFLOW2027_lzqw_uid1007/evidence_staging'),
}


def ssh(host,code):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',host,
        'python3 -c '+shlex.quote(code)],text=True,timeout=60)


def save(record):
    record['updated_at_unix']=time.time()
    RECEIPT.write_text(json.dumps(record,indent=2)+'\n')


def main():
    assert not RECEIPT.exists(), 'One-shot deployment; inspect any existing receipt'
    new=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert new=='9993864b13b60ed4171bd3e6f9e7ac78b11ca05e0d9a60496b0da9020a4f184d'
    proc=Path('/proc/67233')
    assert str(BASE/'archive_scientific_evidence.py').encode() in proc.joinpath('cmdline').read_bytes().split(b'\0')
    ticks=proc.joinpath('stat').read_text().rsplit(') ',1)[1].split()[19]
    record=dict(phase='deploying',new_helper_sha256=new,expected_old_remote_sha256=OLD,
        coordinator_pid=67233,coordinator_start_ticks=int(ticks),coordinator_restarted=False,
        training_processes_signalled=0,other_projects_paused=False,model_or_tensor_loads=0,
        additional_scientific_rollouts=0,scientific_parameters_changed=False,
        canonical_training_paths_changed=False,existing_archives_moved_or_deleted=False,
        scientific_source_commit='62ca7399357c10645a955dcbead7ed2153a82c69',hosts={},
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(record)
    try:
        for host,(root,mount,parent) in TARGETS.items():
            target=root+'/deploy/archive_scientific_evidence.py'
            staged=root+'/deploy/.archive_data_staging_20261023.incoming.py'
            backup=target+'.before_data_staging_20261023'
            pre=f'''
import hashlib,os,shutil
from pathlib import Path
p=Path({target!r});q=Path({staged!r});b=Path({backup!r})
assert p.is_file() and not p.is_symlink() and p.stat().st_uid==os.getuid()
assert hashlib.sha256(p.read_bytes()).hexdigest()=={OLD!r}
assert not q.exists() and not q.is_symlink() and not b.exists() and not b.is_symlink()
assert os.path.ismount({mount!r}) and shutil.disk_usage({mount!r}).free>=110*1024**3
'''
            ssh(host,pre)
            subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=12',
                str(SOURCE),host+':'+staged],check=True,timeout=60)
            result=ssh(host,f'''
import hashlib,importlib.util,json,os,shutil,stat,time
from pathlib import Path
p=Path({target!r});q=Path({staged!r});b=Path({backup!r})
assert not p.is_symlink() and not q.is_symlink() and p.stat().st_uid==q.stat().st_uid==os.getuid()
assert hashlib.sha256(p.read_bytes()).hexdigest()=={OLD!r}
raw=q.read_bytes();assert hashlib.sha256(raw).hexdigest()=={new!r};compile(raw,str(q),'exec')
assert not b.exists();shutil.copy2(p,b)
assert hashlib.sha256(b.read_bytes()).hexdigest()=={OLD!r}
os.chmod(q,stat.S_IMODE(p.stat().st_mode));os.replace(q,p)
spec=importlib.util.spec_from_file_location('archive_storage_only',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert str(m.snapshot_parent({host!r}))=={parent!r}
assert str(m.snapshot_parent('5080'))=='/home/autodrive/LZQW/SAFEFLOW2027/recovery_backups'
assert m.SOURCE=='62ca7399357c10645a955dcbead7ed2153a82c69'
print(json.dumps(dict(deployed_helper=str(p),old_helper_backup=str(b),
 old_sha256={OLD!r},new_sha256={new!r},temporary_snapshot_parent={parent!r},
 mounted_filesystem={mount!r},free_bytes_at_deployment=shutil.disk_usage({mount!r}).free,
 imported_and_parent_selection_checked_without_snapshot=True,
 extra_snapshot_invocations=0,deployed_at_unix=time.time())))
''')
            record['hosts'][host]=json.loads(result);save(record)
        assert proc.joinpath('stat').read_text().rsplit(') ',1)[1].split()[19]==ticks
        record.update(phase='deployed_pending_natural_archive_cycle',coordinator_identity_unchanged=True,
            cadence_seconds=1800,archive_scope_changed=False,
            compatibility='Old home and new data-disk temporary stages both remain valid cleanup inputs; canonical training and evidence paths are unchanged.')
        save(record)
        print(json.dumps(dict(phase=record['phase'],hosts=list(record['hosts']),
            coordinator_pid=67233,coordinator_restarted=False,extra_snapshots=0)),flush=True)
    except BaseException as error:
        record.update(phase='needs_inspection_no_automatic_redeploy',error=str(error));save(record)
        raise


if __name__=='__main__':
    main()
