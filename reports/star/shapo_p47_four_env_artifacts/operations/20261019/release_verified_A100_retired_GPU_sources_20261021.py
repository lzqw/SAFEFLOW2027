#!/usr/bin/env python3
"""Release only exact cold source copies after a fresh verified4T archive check."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import archive_retired_GPU_A100_to_4T_20261021_v2 as copy

BASE = Path(__file__).resolve().parent
STATE = BASE / 'A100_retired_GPU_archive_20261021_source_release_state.json'
EXPECTED_SHA = 'eeddea5e98694aba67dde59a6de076758b9d78fcff729c436a7389f7fe49ac8c'
TARGET = '/home/carla/4T/SAFEFLOW2027_4080/retired_originals_from_a100/20261021/retired_gpu_workers'

TARGET_CHECK = r'''
import hashlib,json,sys,time
from pathlib import Path
q=json.load(sys.stdin);m=q['manifest'];root=Path(q['target'])
assert str(root)=='/home/carla/4T/SAFEFLOW2027_4080/retired_originals_from_a100/20261021/retired_gpu_workers'
assert root.resolve()==root
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='3b362c03-4ed3-4e03-aea4-cf7339356026'
archive=Path('/home/carla/4T/SAFEFLOW2027_4080');project=Path('/home/carla/SAFEFLOW2027_4080')
assert root.stat().st_dev==archive.stat().st_dev!=project.stat().st_dev
assert hashlib.sha256((root/'ARCHIVE_MANIFEST.json').read_bytes()).hexdigest()==q['manifest_sha256']
old=json.loads((root/'ARCHIVE_VERIFICATION.json').read_text())
assert old['source_manifest_sha256']==q['manifest_sha256'] and old['verified_regular_files']==598
assert old['verified_data_bytes']==8013297651
for r in m['files']:
 p=root/r['relative_path'];assert not p.is_symlink() and root in p.resolve().parents
 assert p.is_file() and p.stat().st_size==r['bytes'] and p.stat().st_dev==root.stat().st_dev
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 assert h.hexdigest()==r['sha256'],'Archive data changed; do not remove source'
print(json.dumps({'phase':'archive_reverified_immediately_before_source_release','verified_at_unix':time.time(),
 'target_root':str(root),'source_manifest_sha256':q['manifest_sha256'],'verified_regular_files':598,
 'verified_data_bytes':8013297651,'separate_4T_filesystem_verified':True}))
'''

SOURCE_RELEASE = r'''
import hashlib,json,os,shutil,stat,sys,time
from pathlib import Path
q=json.load(sys.stdin);m=q['manifest'];root=Path(m['source_root']);sha=q['manifest_sha256']
assert str(root)=='/home/lzqw/SAFEFLOW2027_A100/recovery_backups/retired_gpu_workers'
assert root.resolve()==root and root.is_dir()
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==m['source_boot_id']=='6928f333-1036-473c-92d5-4c482d5fef59'
assert hashlib.sha256((json.dumps(m,indent=2,sort_keys=True)+chr(10)).encode()).hexdigest()==sha
assert sha=='eeddea5e98694aba67dde59a6de076758b9d78fcff729c436a7389f7fe49ac8c'
assert q['target_check']['phase']=='archive_reverified_immediately_before_source_release'
assert q['target_check']['source_manifest_sha256']==sha and q['target_check']['verified_regular_files']==598
target='/home/carla/4T/SAFEFLOW2027_4080/retired_originals_from_a100/20261021/retired_gpu_workers'
assert q['target_check']['target_root']==target and 0<=time.time()-q['target_check']['verified_at_unix']<=300
assert sorted(p.name for p in root.iterdir())==sorted(m['scope_directories'])
assert len(q['active_workers'])>=3
for w in q['active_workers']:
 p=Path('/proc',str(w['pid']));s=(p/'stat').read_text().rsplit(') ',1)[1].split()
 assert int(s[19])==w['start_ticks'] and w['boot_id']==m['source_boot_id']
 args=[a.decode() for a in (p/'cmdline').read_bytes().split(bytes([0])) if a]
 assert 'scripts/star/train_star.py' in args and not any(str(root) in a for a in args)
 for fd in (p/'fd').iterdir():
  try:link=os.readlink(fd)
  except OSError:continue
  assert not link.startswith(str(root)+'/'),'An active owned worker is reading a cold source'
def cookie(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
expected={r['relative_path']:r for r in m['files']};seen=set();dirs=[]
for p in sorted(root.rglob('*')):
 s=p.lstat();relative=str(p.relative_to(root));assert not p.is_symlink() and root in p.resolve().parents
 if stat.S_ISDIR(s.st_mode):dirs.append(relative);continue
 assert stat.S_ISREG(s.st_mode) and relative in expected
 assert cookie(s)==expected[relative]['source_cookie']
 seen.add(relative)
assert seen==set(expected) and len(seen)==598 and sorted(dirs)==sorted(m['directories'])
ledger=root.parent/'retired_GPU_4T_source_release_20261021.json'
assert not ledger.exists() and not ledger.with_suffix('.tmp').exists(),'Inspect prior source release; do not repeat'
free_before=shutil.disk_usage(root).free;assert free_before>=2*1024**3
receipt={'phase':'releasing_exact_verified_source_copies','started_at_unix':time.time(),
 'source_root':str(root),'target_host':'4080','target_root':target,'source_manifest_sha256':sha,
 'source_files_removed':0,'source_logical_bytes_released':0,'pointer_directories':[],
 'filesystem_free_bytes_before':free_before,'training_processes_signalled':0,'other_projects_modified':False,
 'original_bytes_remain_on_verified_separate_4T':True,'no_original_parent_identity_aliases':True}
def save():
 part=ledger.with_suffix('.tmp');part.write_text(json.dumps(receipt,indent=2)+chr(10));part.replace(ledger)
save()
try:
 for r in m['files']:
  relative=Path(r['relative_path'])
  assert not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] in m['scope_directories']
  p=root/relative;assert root in p.resolve().parents and not p.is_symlink()
  assert cookie(p.stat())==r['source_cookie'];p.unlink()
  receipt['source_files_removed']+=1;receipt['source_logical_bytes_released']+=r['bytes']
 for relative in sorted(m['directories'],key=lambda x:len(Path(x).parts),reverse=True):
  if relative in m['scope_directories']:continue
  p=root/relative;assert root in p.resolve().parents;p.rmdir()
 for name in m['scope_directories']:
  d=root/name;assert d.is_dir() and not list(d.iterdir())
  pointer={'format':'safeflow-retired-original-storage-location-v1','original_archive_directory':str(d),
    'target_host':'4080','target_archive_directory':target+'/'+name,'source_manifest_sha256':sha,
    'manifest_at_target':target+'/ARCHIVE_MANIFEST.json','original_bytes_preserved':True,
    'restore_instruction':'Copy this exact archived directory from4080 back under '+str(root)+', preserving original filenames and checking the manifest SHA values.',
    'not_a_checkpoint_identity_alias_or_missing_2027_parent_repair':True}
  (d/'ARCHIVE_LOCATION.json').write_text(json.dumps(pointer,indent=2)+chr(10))
  receipt['pointer_directories'].append(str(d))
 receipt.update(phase='source_copies_released_to_verified_4T_archive',finished_at_unix=time.time(),
  filesystem_free_bytes_after=shutil.disk_usage(root).free,
  distinct_original_bytes_removed_from_all_locations=0,
  observed_filesystem_delta_includes_other_shared_filesystem_activity=True)
 save();print(json.dumps(receipt))
except BaseException as e:
 receipt.update(phase='needs_inspection_4T_originals_preserved',error=str(e),finished_at_unix=time.time());save();raise
'''


def save(state, phase, **updates):
    state.update(updates, phase=phase, updated_at_unix=time.time())
    part = STATE.with_suffix('.tmp')
    part.write_text(json.dumps(state, indent=2) + '\n')
    part.replace(STATE)
    print(json.dumps({'phase': phase, 'state': STATE.name}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-verified-source-copies', required=True, action='store_true')
    parser.parse_args()
    assert not STATE.exists(), 'Inspect the original source-release transaction, never repeat it'
    old = json.loads(copy.STATE.read_text())
    assert old['phase'] == 'verified_4T_archive_source_originals_still_untouched'
    assert old['source_files_removed'] == 0
    assert hashlib.sha256(Path(copy.__file__).read_bytes()).hexdigest() == old['controller_sha256']
    payload = copy.MANIFEST.read_bytes();assert hashlib.sha256(payload).hexdigest() == old['source_manifest_sha256'] == EXPECTED_SHA
    m = json.loads(payload)
    assert m['source_root'] == copy.SOURCE_ROOT and sorted(m['scope_directories']) == sorted(copy.DIRECTORIES)
    assert m['file_count'] == len(m['files']) == 598 and m['total_logical_bytes'] == 8013297651
    assert 'verify_retired_GPU_archive_relocation(BASE)' in (BASE / 'publish_final.py').read_text()
    state = {'format': 'verified-retired-original-source-release-v1', 'local_pid': os.getpid(),
             'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'copy_state_file': copy.STATE.name, 'source_manifest_file': copy.MANIFEST.name,
             'source_manifest_sha256': EXPECTED_SHA, 'source_files_removed': 0,
             'training_processes_signalled': 0, 'additional_scientific_rollouts': 0}
    save(state, 'rechecking_archive_before_any_source_removal')
    try:
        checked = copy.rpc('4080', TARGET_CHECK, {'manifest': m, 'manifest_sha256': EXPECTED_SHA, 'target': TARGET})
        raw = json.loads((BASE / 'gpu_capacity_state.json').read_text())
        assert 0 <= time.time() - raw['observed_at_unix'] <= 2100
        active = raw['hosts']['tjucta-a100']['training']
        save(state, 'releasing_only_manifest_listed_retired_source_copies', target_recheck=checked)
        released = copy.rpc('tjucta-a100', SOURCE_RELEASE,
                            {'manifest': m, 'manifest_sha256': EXPECTED_SHA,
                             'target_check': checked, 'active_workers': active})
        assert released['phase'] == 'source_copies_released_to_verified_4T_archive'
        assert released['source_files_removed'] == 598 and released['source_logical_bytes_released'] == 8013297651
        save(state, 'source_copies_released_after_4T_verification', source_files_removed=598, source_release=released)
    except BaseException as error:
        save(state, 'needs_inspection_verified_4T_originals_preserved', error=str(error))
        raise


if __name__ == '__main__':
    main()
