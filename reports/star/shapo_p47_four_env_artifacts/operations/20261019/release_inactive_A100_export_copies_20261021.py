#!/usr/bin/env python3
"""Release only15 archived closed export sets, never canonical runs/live backups."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import archive_inactive_A100_exports_20261021 as copy

BASE = Path(__file__).resolve().parent
STATE = BASE / 'A100_inactive_export_archive_20261021_release_state.json'

TARGET_RECHECK = r'''
import hashlib,json,os,sys,time
from pathlib import Path
q=json.load(sys.stdin);m=q['manifest'];root=Path(q['target'])
assert str(root)=='/home/carla/4T/SAFEFLOW2027_4080/retired_export_snapshots_from_a100/20261021/inactive_export_snapshots'
assert root.resolve()==root and root.stat().st_dev==Path('/home/carla/4T/SAFEFLOW2027_4080').stat().st_dev!=Path('/home/carla/SAFEFLOW2027_4080').stat().st_dev
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='3b362c03-4ed3-4e03-aea4-cf7339356026'
assert hashlib.sha256((root/'ARCHIVE_MANIFEST.json').read_bytes()).hexdigest()==q['sha256']
old=json.loads((root/'ARCHIVE_VERIFICATION.json').read_text())
assert old['manifest_sha256']==q['sha256'] and old['verified_regular_files']==m['file_count']==89
for r in m['files']:
 p=root/r['relative_path'];assert not p.is_symlink() and root in p.resolve().parents
 assert p.is_file() and p.stat().st_size==r['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
  os.fsync(f.fileno())
 assert h.hexdigest()==r['sha256']
for name in ('ARCHIVE_MANIFEST.json','ARCHIVE_VERIFICATION.json'):
 with (root/name).open('rb') as f:os.fsync(f.fileno())
archive=Path('/home/carla/4T/SAFEFLOW2027_4080')
ancestors=[root]+[p for p in root.parents if p==archive or archive in p.parents]
for directory in [root/relative for relative in m['directories']]+ancestors:
 fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
print(json.dumps({'phase':'4T_archive_rechecked_before_source_release','verified_at_unix':time.time(),
 'manifest_sha256':q['sha256'],'target_root':str(root),'verified_regular_files':89,
 'verified_logical_bytes':m['total_logical_bytes'],'closed_jobs':15,'separate_4T_filesystem_verified':True,
 'archive_bytes_and_directory_entries_fsynced':True}))
'''

SOURCE_RELEASE = r'''
import hashlib,json,os,shutil,stat,sys,time
from pathlib import Path
q=json.load(sys.stdin);m=q['manifest'];root=Path(m['source_root']);project=Path('/home/lzqw/SAFEFLOW2027_A100')
assert str(root)=='/home/lzqw/SAFEFLOW2027_A100/recovery_backups/export' and root.resolve()==root
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==m['source_boot_id']=='6928f333-1036-473c-92d5-4c482d5fef59'
assert hashlib.sha256((json.dumps(m,indent=2,sort_keys=True)+chr(10)).encode()).hexdigest()==q['sha256']
checked=q['target_recheck'];assert checked['phase']=='4T_archive_rechecked_before_source_release'
assert checked['manifest_sha256']==q['sha256'] and checked['verified_regular_files']==m['file_count']==89
assert checked['archive_bytes_and_directory_entries_fsynced'] is True
assert checked['closed_jobs']==len(m['scope_jobs'])==15 and 0<=time.time()-checked['verified_at_unix']<=300
target='/home/carla/4T/SAFEFLOW2027_4080/retired_export_snapshots_from_a100/20261021/inactive_export_snapshots'
assert checked['target_root']==target
for job in m['scope_jobs']:
 d=project/'runs/shapo_p47_four_env/runs'/job
 assert not d.exists() and not d.is_symlink(),'Job returned to A100: preserve its exports'
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=[v.decode(errors='replace') for v in p.read_bytes().split(bytes([0])) if v]
  if not any(v.endswith('scripts/star/train_star.py') or v.endswith('scripts/star/validate_shapo_p47_run.py') for v in a):continue
  if '--config' in a:
   c=Path(a[a.index('--config')+1]);c.relative_to(project/'runs/shapo_p47_four_env/runs')
   assert c.parent.name not in m['scope_jobs']
   for fd in (p.parent/'fd').iterdir():
    try:link=os.readlink(fd)
    except OSError:continue
    assert not any(link.startswith(str(root/job)+'/') for job in m['scope_jobs'])
  if '--container' in a:assert Path(a[a.index('--container')+1]).name not in m['scope_jobs']
 except(OSError,ValueError,IndexError):continue
def cookie(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
expected={r['relative_path']:r for r in m['files']};seen=set();dirs=[]
for job in m['scope_jobs']:
 d=root/job;assert d.is_dir() and not d.is_symlink();dirs.append(job)
 for p in sorted(d.rglob('*')):
  s=p.lstat();relative=str(p.relative_to(root));assert not p.is_symlink() and root in p.resolve().parents
  if stat.S_ISDIR(s.st_mode):dirs.append(relative);continue
  assert stat.S_ISREG(s.st_mode) and relative in expected
  r=expected[relative];assert cookie(s)==r['source_cookie'] and stat.S_IMODE(s.st_mode)==r['source_mode']
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
  assert h.hexdigest()==r['sha256'] and cookie(p.stat())==r['source_cookie']
  seen.add(relative)
assert seen==set(expected) and sorted(dirs)==sorted(m['directories'])
ledger=project/'recovery_backups/inactive_export_4T_release_20261021.json'
assert not ledger.exists() and not ledger.with_suffix('.tmp').exists(),'Never repeat a source release'
receipt={'phase':'releasing_only_verified_closed_export_copies','started_at_unix':time.time(),
 'source_root':str(root),'target_host':'4080','target_root':target,'manifest_sha256':q['sha256'],
 'source_files_removed':0,'source_logical_bytes_released':0,'pointer_directories':[],
 'filesystem_free_bytes_before':shutil.disk_usage(root).free,'learner_signals_or_restarts':0,
 'canonical_run_files_modified':0,'active_export_job_files_modified':0,'other_projects_modified':False,
 'original_bytes_remain_on_verified_separate_4T':True,'outside_hardlinks_not_modified':True}
def save():
 part=ledger.with_suffix('.tmp');part.write_text(json.dumps(receipt,indent=2)+chr(10));part.replace(ledger)
save()
try:
 for r in m['files']:
  relative=Path(r['relative_path']);assert not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] in m['scope_jobs']
  canonical=project/'runs/shapo_p47_four_env/runs'/relative.parts[0]
  assert not canonical.exists() and not canonical.is_symlink()
  p=root/relative;assert not p.is_symlink() and root in p.resolve().parents and cookie(p.stat())==r['source_cookie']
  p.unlink();receipt['source_files_removed']+=1;receipt['source_logical_bytes_released']+=r['bytes']
 for relative in sorted(m['directories'],key=lambda x:len(Path(x).parts),reverse=True):
  if relative in m['scope_jobs']:continue
  (root/relative).rmdir()
 for job in m['scope_jobs']:
  d=root/job;assert d.is_dir() and not list(d.iterdir())
  pointer={'kind':'historical_cold_export_snapshot_location_not_current_backup_override',
   'original_export_directory':str(d),'archived_at_host':'4080','archive_directory':target+'/'+job,
   'manifest_at_target':target+'/ARCHIVE_MANIFEST.json','manifest_sha256':q['sha256'],
   'later_terminal_imports_may_repopulate_this_export_directory':True,
   'restore_note':'Copy exact archived files with original mode/mtime and verify the manifest. Restore to a separate directory first. Do not overwrite any later fresh export or running job. This archive is not an alias for a different checkpoint or a repair of missing2027 parents.'}
  (d/'cold_export_before_final_import_20261021.json').write_text(json.dumps(pointer,indent=2)+chr(10))
  receipt['pointer_directories'].append(str(d))
 receipt.update(phase='closed_export_source_copies_released_to_verified_4T',finished_at_unix=time.time(),
  filesystem_free_bytes_after=shutil.disk_usage(root).free,distinct_original_bytes_removed_from_all_locations=0,
  observed_free_delta_includes_shared_filesystem_activity=True)
 save();print(json.dumps(receipt))
except BaseException as error:
 receipt.update(phase='needs_inspection_verified_4T_originals_preserved',error=str(error));save();raise
'''


def save(state, phase, **updates):
    state.update(updates, phase=phase, updated_at_unix=time.time())
    part = STATE.with_suffix('.tmp');part.write_text(json.dumps(state, indent=2) + '\n');part.replace(STATE)
    print(json.dumps({'phase': phase, 'state': STATE.name}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release-verified-inactive-export-copies', action='store_true', required=True)
    parser.parse_args()
    assert not STATE.exists(), 'Inspect original release; never replay'
    old = json.loads(copy.STATE.read_text())
    assert old['phase'] == 'verified_4T_copy_sources_untouched' and old['source_files_removed'] == 0
    assert hashlib.sha256(Path(copy.__file__).read_bytes()).hexdigest() == old['controller_sha256']
    blob = copy.MANIFEST.read_bytes();sha = hashlib.sha256(blob).hexdigest()
    assert sha == old['manifest_sha256']
    m = json.loads(blob)
    assert m['scope_jobs'] == copy.JOBS and m['file_count'] == 89 and m['source_root'] == copy.SOURCE_ROOT
    assert 'verify_inactive_A100_export_relocation(BASE)' in (BASE / 'publish_final.py').read_text()
    state = {'kind': 'verified_inactive_export_source_release', 'local_pid': os.getpid(),
             'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'copy_state_file': copy.STATE.name, 'manifest_file': copy.MANIFEST.name,
             'manifest_sha256': sha, 'source_files_removed': 0, 'new_model_or_evaluation_runs': 0}
    save(state, 'rechecking_4T_archive_before_source_release')
    try:
        checked = copy.rpc('4080', TARGET_RECHECK, {'manifest': m, 'sha256': sha, 'target': copy.TARGET_ROOT})
        save(state, 'releasing_only_manifest_listed_closed_export_copies', target_recheck=checked)
        released = copy.rpc('tjucta-a100', SOURCE_RELEASE, {'manifest': m, 'sha256': sha, 'target_recheck': checked})
        assert released['phase'] == 'closed_export_source_copies_released_to_verified_4T'
        assert released['source_files_removed'] == m['file_count']
        assert released['source_logical_bytes_released'] == m['total_logical_bytes']
        save(state, 'closed_export_copies_released_after_verified_4T', source_files_removed=m['file_count'], source_release=released)
    except BaseException as error:
        save(state, 'needs_inspection_verified_4T_originals_preserved', error=str(error))
        raise


if __name__ == '__main__':
    main()
