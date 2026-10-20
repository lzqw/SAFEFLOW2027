#!/usr/bin/env python3
"""Copy15 closed export-snapshot sets to4T; this program NEVER deletes sources."""
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from archive_retired_GPU_A100_to_4T_20261021_v2 import rpc, SSH

BASE = Path(__file__).resolve().parent
STATE = BASE / 'A100_inactive_export_archive_20261021_copy_state.json'
MANIFEST = BASE / 'A100_inactive_export_archive_20261021_source_manifest.json'
SOURCE_ROOT = '/home/lzqw/SAFEFLOW2027_A100/recovery_backups/export'
TARGET_PARENT = '/home/carla/4T/SAFEFLOW2027_4080/retired_export_snapshots_from_a100/20261021'
TARGET_ROOT = TARGET_PARENT + '/inactive_export_snapshots'
JOBS = sorted(['ant-2028', 'ant-2029', 'ant-2031', 'walker2d-2028', 'walker2d-2029']
              + [slug + '-' + str(seed) for slug in ('point-goal', 'point-button') for seed in range(2027, 2032)])

SOURCE_CODE = r'''
import collections,hashlib,json,os,shutil,stat,sys,time
from pathlib import Path
q=json.load(sys.stdin);root=Path(q['root']);project=Path('/home/lzqw/SAFEFLOW2027_A100')
assert str(root)=='/home/lzqw/SAFEFLOW2027_A100/recovery_backups/export' and root.resolve()==root
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='6928f333-1036-473c-92d5-4c482d5fef59'
assert len(q['jobs'])==len(set(q['jobs']))==15
for job in q['jobs']:
 assert job in {slug+'-'+str(seed) for slug in ('point-goal','point-button','ant','walker2d') for seed in range(2027,2032)}
 d=project/'runs/shapo_p47_four_env/runs'/job
 assert not d.exists() and not d.is_symlink(),'Canonical job returned; preserve export untouched'
 assert (root/job).is_dir() and not (root/job).is_symlink()
active=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=[v.decode(errors='replace') for v in p.read_bytes().split(bytes([0])) if v]
  if not any(v.endswith('scripts/star/train_star.py') or v.endswith('scripts/star/validate_shapo_p47_run.py') for v in a):continue
  if '--config' in a:
   c=Path(a[a.index('--config')+1]);c.relative_to(project/'runs/shapo_p47_four_env/runs')
   assert c.parent.name not in q['jobs'],'Selected export belongs to a live owned job'
   active.append(c.parent.name)
   for fd in (p.parent/'fd').iterdir():
    try:target=os.readlink(fd)
    except OSError:continue
    assert not any(target.startswith(str(root/job)+'/') for job in q['jobs'])
  if '--container' in a:assert Path(a[a.index('--container')+1]).name not in q['jobs']
 except(OSError,ValueError,IndexError):continue
def cookie(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
items=[];dirs=[];counts=collections.Counter();inodes={}
for job in q['jobs']:
 d=root/job;dirs.append(job)
 for p in sorted(d.rglob('*')):
  s=p.lstat();relative=str(p.relative_to(root));assert not p.is_symlink() and root in p.resolve().parents
  if stat.S_ISDIR(s.st_mode):dirs.append(relative);continue
  assert stat.S_ISREG(s.st_mode)
  if q['mode']=='inventory':
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
   assert cookie(p.stat())==cookie(s)
   items.append({'relative_path':relative,'bytes':s.st_size,'sha256':h.hexdigest(),'source_cookie':cookie(s),
                 'source_mode':stat.S_IMODE(s.st_mode),'source_nlink':s.st_nlink})
   key=(s.st_dev,s.st_ino);counts[key]+=1;inodes[key]=s
  else:
   expected=q['files'].get(relative);assert expected is not None and cookie(s)==expected['source_cookie']
   items.append(relative)
if q['mode']=='inventory':
 result={'format':'safeflow-inactive-export-sha256-v1','origin':'tjucta-a100','source_root':str(root),
  'source_boot_id':'6928f333-1036-473c-92d5-4c482d5fef59','created_at_unix':time.time(),
  'scope_jobs':q['jobs'],'directories':dirs,'files':items,'file_count':len(items),
  'total_logical_bytes':sum(r['bytes'] for r in items),
  'physical_bytes_potentially_released':sum(s.st_blocks*512 for key,s in inodes.items() if counts[key]==s.st_nlink),
  'outside_hardlinked_inodes_not_removed':sum(counts[key]<s.st_nlink for key,s in inodes.items()),
  'canonical_containers_absent':True,'live_own_jobs_not_in_scope':sorted(active),
  'not_a_checkpoint_parent_identity_repair':True}
else:
 assert set(items)==set(q['files']) and sorted(dirs)==sorted(q['directories'])
 result={'source_identity_cookies_unchanged':True,'canonical_containers_still_absent':True,
  'file_count':len(items),'source_files_removed':0}
print(json.dumps(result))
'''

TARGET_PREPARE = r'''
import json,shutil,sys,tempfile
from pathlib import Path
q=json.load(sys.stdin);archive=Path('/home/carla/4T/SAFEFLOW2027_4080');project=Path('/home/carla/SAFEFLOW2027_4080')
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='3b362c03-4ed3-4e03-aea4-cf7339356026'
assert archive.resolve()==archive and archive.stat().st_dev!=project.stat().st_dev
parent=Path(q['parent']);assert str(parent)=='/home/carla/4T/SAFEFLOW2027_4080/retired_export_snapshots_from_a100/20261021'
parent.mkdir(parents=True,exist_ok=True);assert parent.resolve()==parent and archive in parent.parents
final=parent/'inactive_export_snapshots';assert not final.exists(),'Never overwrite a prior archive'
assert shutil.disk_usage(parent).free>=q['bytes']+100*1024**3
stage=Path(tempfile.mkdtemp(prefix='.inactive-export-incoming.',dir=parent))
print(json.dumps({'stage':str(stage),'final':str(final),'separate_4T_filesystem_verified':True}))
'''

TARGET_VERIFY = r'''
import hashlib,json,os,stat,sys,time
from pathlib import Path
q=json.load(sys.stdin);m=q['manifest'];stage=Path(q['stage']);final=Path(q['final'])
parent=Path('/home/carla/4T/SAFEFLOW2027_4080/retired_export_snapshots_from_a100/20261021')
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='3b362c03-4ed3-4e03-aea4-cf7339356026'
assert stage.parent==parent and stage.name.startswith('.inactive-export-incoming.') and stage.resolve()==stage
assert final==parent/'inactive_export_snapshots' and not final.exists()
assert stage.stat().st_dev==Path('/home/carla/4T/SAFEFLOW2027_4080').stat().st_dev!=Path('/home/carla/SAFEFLOW2027_4080').stat().st_dev
expected={r['relative_path']:r for r in m['files']};seen=set();dirs=[]
for p in sorted(stage.rglob('*')):
 s=p.lstat();relative=str(p.relative_to(stage));assert not p.is_symlink() and stage in p.resolve().parents
 if stat.S_ISDIR(s.st_mode):dirs.append(relative);continue
 assert stat.S_ISREG(s.st_mode) and relative in expected
 r=expected[relative];assert s.st_size==r['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 assert h.hexdigest()==r['sha256']
 os.chmod(p,r['source_mode']);os.utime(p,ns=(p.stat().st_atime_ns,r['source_cookie'][3]))
 seen.add(relative)
assert seen==set(expected) and sorted(dirs)==sorted(m['directories'])
encoded=json.dumps(m,indent=2,sort_keys=True)+chr(10);assert hashlib.sha256(encoded.encode()).hexdigest()==q['sha256']
(stage/'ARCHIVE_MANIFEST.json').write_text(encoded)
receipt={'phase':'verified_4T_copy_sources_untouched','verified_at_unix':time.time(),'source_root':m['source_root'],
 'target_root':str(final),'manifest_sha256':q['sha256'],'verified_regular_files':len(seen),
 'verified_logical_bytes':m['total_logical_bytes'],'closed_job_count':len(m['scope_jobs']),
 'separate_4T_filesystem_verified':True,'source_files_removed':0,'training_processes_signalled':0}
(stage/'ARCHIVE_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+chr(10))
stage.rename(final);print(json.dumps(receipt))
'''


def save(state, phase, **updates):
    state.update(updates, phase=phase, updated_at_unix=time.time())
    part = STATE.with_suffix('.tmp');part.write_text(json.dumps(state, indent=2) + '\n');part.replace(STATE)
    print(json.dumps({'phase': phase, 'state': STATE.name}), flush=True)


def main():
    assert not STATE.exists() and not MANIFEST.exists(), 'Inspect original copy; never replay'
    state = {'kind': 'copy_only_inactive_A100_exports_to_4T', 'local_pid': os.getpid(),
             'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'source_root': SOURCE_ROOT, 'scope_jobs': JOBS, 'target_root': TARGET_ROOT,
             'source_files_removed': 0, 'training_signals_or_model_runs': 0,
             'source_release_not_authorized_by_this_copy_program': True}
    save(state, 'inventorying_closed_source_exports_source_untouched')
    try:
        m = rpc('tjucta-a100', SOURCE_CODE, {'root': SOURCE_ROOT, 'jobs': JOBS, 'mode': 'inventory'})
        assert len(m['scope_jobs']) == 15 and m['file_count'] == 89
        encoded = json.dumps(m, indent=2, sort_keys=True) + '\n'
        with MANIFEST.open('x') as stream:stream.write(encoded)
        sha = hashlib.sha256(encoded.encode()).hexdigest()
        target = rpc('4080', TARGET_PREPARE, {'parent': TARGET_PARENT, 'bytes': m['total_logical_bytes']})
        save(state, 'copying_closed_exports_source_untouched', manifest_file=MANIFEST.name,
             manifest_sha256=sha, file_count=m['file_count'], logical_bytes=m['total_logical_bytes'], target=target)
        sender = 'nice -n 10 tar --format=posix -czf - -C ' + shlex.quote(SOURCE_ROOT) + ' -- ' + ' '.join(map(shlex.quote, JOBS))
        receiver = 'taskset -c 16-27 nice -n 10 tar -xzf - --no-same-owner -C ' + shlex.quote(target['stage'])
        with subprocess.Popen([*SSH, 'tjucta-a100', sender], stdout=subprocess.PIPE) as producer:
            try:result = subprocess.run([*SSH, '4080', receiver], stdin=producer.stdout, timeout=14400)
            finally:
                producer.stdout.close()
                try:producer.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    producer.terminate()
                    try:producer.wait(timeout=15)
                    except subprocess.TimeoutExpired:producer.kill();producer.wait()
            assert producer.returncode == 0 and result.returncode == 0
        save(state, 'verifying_4T_bytes_source_untouched')
        checked = rpc('4080', TARGET_VERIFY, {'manifest': m, 'sha256': sha, **target})
        unchanged = rpc('tjucta-a100', SOURCE_CODE, {'root': SOURCE_ROOT, 'jobs': JOBS, 'mode': 'verify',
                        'files': {r['relative_path']: r for r in m['files']}, 'directories': m['directories']})
        save(state, 'verified_4T_copy_sources_untouched', target_verification=checked, source_recheck=unchanged)
    except BaseException as error:
        save(state, 'needs_inspection_sources_untouched', error=str(error))
        raise


if __name__ == '__main__':
    main()
