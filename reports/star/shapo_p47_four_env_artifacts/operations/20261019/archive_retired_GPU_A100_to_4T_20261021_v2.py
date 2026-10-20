#!/usr/bin/env python3
"""Python3.8-compatible copy-only successor; preserve the failed v1 record."""
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATE = BASE / 'A100_retired_GPU_archive_20261021_v2_state.json'
MANIFEST = BASE / 'A100_retired_GPU_archive_20261021_v2_source_manifest.json'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
SOURCE_ROOT = '/home/lzqw/SAFEFLOW2027_A100/recovery_backups/retired_gpu_workers'
TARGET_PARENT = '/home/carla/4T/SAFEFLOW2027_4080/retired_originals_from_a100/20261021'
DIRECTORIES = [
    'ant-2028_cpu5080_1792397493369363796',
    'ant-2029_cpu5080_1792397501825106309',
    'point-button-2027_balanced_MPS_5080_20261020_1792505069788259532',
    'point-button-2028_balanced_5080_20261020_0521_1792445670302638469',
    'point-button-2029_5080_after_reboot_1792394832217976708',
    'point-button-2030_1792041646634230940',
    'point-button-2031_1792106274391480995',
    'point-goal-2027_balanced_MPS_5080_20261020_1792494887972611747',
    'point-goal-2029_5080_1792393236238695773',
    'point-goal-2030_balanced_5080_20261020_0621_1792449339345817322',
    'point-goal-2031_5080_1792400319644619229',
    'walker2d-2028_cpu5080_1792397604877988208',
    'walker2d-2029_cpu5080_1792398162395213925',
]

SOURCE_CODE = r'''
import hashlib,json,os,shutil,stat,sys,time
from pathlib import Path
q=json.load(sys.stdin);root=Path(q['root'])
assert str(root)=='/home/lzqw/SAFEFLOW2027_A100/recovery_backups/retired_gpu_workers'
assert root.resolve()==root and root.is_dir()
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='6928f333-1036-473c-92d5-4c482d5fef59'
assert shutil.disk_usage(root).free>=2*1024**3
assert sorted(p.name for p in root.iterdir())==sorted(q['directories'])
for pid,ticks in [(14547,257250),(14550,257260),(2384563,9279674)]:
 p=Path('/proc',str(pid));s=(p/'stat').read_text().rsplit(') ',1)[1].split()
 assert int(s[19])==ticks
 args=[a.decode() for a in (p/'cmdline').read_bytes().split(bytes([0])) if a]
 assert 'scripts/star/train_star.py' in args and not any(str(root) in a for a in args)
 for f in (p/'fd').iterdir():
  try:target=os.readlink(f)
  except OSError:continue
  assert not target.startswith(str(root)+'/'),'Owned learner currently reads a retired original'
def cookie(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
items=[];directories=[]
for p in sorted(root.rglob('*')):
 s=p.lstat();relative=str(p.relative_to(root))
 assert not p.is_symlink() and (root in p.resolve().parents)
 if stat.S_ISDIR(s.st_mode):directories.append(relative);continue
 assert stat.S_ISREG(s.st_mode),'Unexpected file type in cold archive'
 if q['mode']=='inventory':
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
  assert cookie(p.stat())==cookie(s),'Cold file changed during hashing'
  items.append({'relative_path':relative,'bytes':s.st_size,'sha256':h.hexdigest(),
                'source_cookie':cookie(s),'source_mode':stat.S_IMODE(s.st_mode)})
 else:
  expected=q['files'].get(relative);assert expected is not None and cookie(s)==expected['source_cookie']
  items.append(relative)
if q['mode']=='inventory':
 result={'format':'safeflow-retired-GPU-originals-sha256-v1','origin':'tjucta-a100',
  'source_root':str(root),'source_boot_id':'6928f333-1036-473c-92d5-4c482d5fef59',
  'created_at_unix':time.time(),'scope_directories':q['directories'],'directories':directories,
  'files':items,'file_count':len(items),'total_logical_bytes':sum(r['bytes'] for r in items),
  'active_owned_learners_checked_no_reference':True,'no_original_parent_identity_aliases':True,
  'known_early2027_missing_full_parents_not_repaired':True}
else:
 assert set(items)==set(q['files'])
 assert sorted(directories)==sorted(q['manifest_directories'])
 result={'source_file_identity_cookies_unchanged':True,'file_count':len(items),'source_files_removed':0}
print(json.dumps(result))
'''

TARGET_PREPARE = r'''
import json,os,shutil,sys,tempfile
from pathlib import Path
q=json.load(sys.stdin);archive=Path('/home/carla/4T/SAFEFLOW2027_4080');project=Path('/home/carla/SAFEFLOW2027_4080')
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='3b362c03-4ed3-4e03-aea4-cf7339356026'
assert archive.resolve()==archive and archive.stat().st_dev!=project.stat().st_dev
parent=Path(q['parent']);assert str(parent)=='/home/carla/4T/SAFEFLOW2027_4080/retired_originals_from_a100/20261021'
parent.mkdir(parents=True,exist_ok=True);assert parent.resolve()==parent and (archive in parent.parents)
final=parent/'retired_gpu_workers';assert not final.exists(),'Inspect existing archive; never overwrite'
assert shutil.disk_usage(parent).free>=q['incoming_bytes']+100*1024**3
stage=Path(tempfile.mkdtemp(prefix='.retired_gpu_workers-incoming.',dir=parent))
assert stage.stat().st_dev==archive.stat().st_dev
print(json.dumps({'target_staging_directory':str(stage),'target_archive_directory':str(final),
 'separate_4T_filesystem_verified':True,'archive_helper_CPU_affinity':sorted(os.sched_getaffinity(0))}))
'''

TARGET_VERIFY = r'''
import hashlib,json,os,stat,sys,time
from pathlib import Path
q=json.load(sys.stdin);m=q['manifest'];stage=Path(q['stage']);final=Path(q['final'])
parent=Path('/home/carla/4T/SAFEFLOW2027_4080/retired_originals_from_a100/20261021')
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='3b362c03-4ed3-4e03-aea4-cf7339356026'
assert stage.parent==parent and stage.name.startswith('.retired_gpu_workers-incoming.')
assert stage.resolve()==stage and final==parent/'retired_gpu_workers' and not final.exists()
expected={r['relative_path']:r for r in m['files']};seen=set();dirs=[]
for p in sorted(stage.rglob('*')):
 s=p.lstat();relative=str(p.relative_to(stage));assert not p.is_symlink() and (stage in p.resolve().parents)
 if stat.S_ISDIR(s.st_mode):dirs.append(relative);continue
 assert stat.S_ISREG(s.st_mode) and relative in expected
 r=expected[relative];assert s.st_size==r['bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 assert h.hexdigest()==r['sha256'],'Archived original SHA mismatch'
 seen.add(relative)
assert seen==set(expected) and sorted(dirs)==sorted(m['directories'])
payload=json.dumps(m,indent=2,sort_keys=True)+'\n'
assert hashlib.sha256(payload.encode()).hexdigest()==q['manifest_sha256']
(stage/'ARCHIVE_MANIFEST.json').write_text(payload)
receipt={'format':'verified-retired-GPU-originals-4T-copy-v1','verified_at_unix':time.time(),
 'source_root':m['source_root'],'target_archive_directory':str(final),
 'source_manifest_sha256':q['manifest_sha256'],'verified_regular_files':len(seen),
 'verified_data_bytes':m['total_logical_bytes'],'source_files_removed':0,
 'separate_4T_filesystem_verified':True,'archive_helper_CPU_affinity':sorted(os.sched_getaffinity(0)),
 'no_early2027_parent_repair_or_checkpoint_identity_alias':True}
(stage/'ARCHIVE_VERIFICATION.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
stage.rename(final)
print(json.dumps(receipt))
'''


def save(state, phase, **updates):
    state.update(updates, phase=phase, updated_at_unix=time.time())
    part = STATE.with_suffix('.tmp')
    part.write_text(json.dumps(state, indent=2) + '\n')
    part.replace(STATE)
    print(json.dumps({'phase': phase, 'state': STATE.name}), flush=True)


def rpc(host, code, payload, timeout=1800):
    prefix = 'taskset -c 16-27 nice -n 10 ' if host == '4080' else 'nice -n 10 '
    result = subprocess.run([*SSH, host, prefix + 'python3 -c ' + shlex.quote(code)],
                            input=json.dumps(payload), capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(host + ' archive metadata operation failed: ' + result.stderr[-2500:])
    return json.loads(result.stdout)


def main():
    assert not STATE.exists() and not MANIFEST.exists(), 'Inspect original live operation; never replay'
    state = {'format': 'retired-GPU-originals-copy-operation-v1', 'local_pid': os.getpid(),
             'local_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'source_root': SOURCE_ROOT, 'target_parent': TARGET_PARENT,
             'source_files_removed': 0, 'training_processes_signalled': 0, 'other_projects_modified': False,
             'additional_scientific_rollouts': 0, 'source_cleanup_not_authorized_by_this_copy_program': True,
             'predecessor_failed_before_transfer': 'A100_retired_GPU_archive_20261021_state.json',
             'predecessor_failure_cause': 'Source system Python3.8 lacks Path.is_relative_to; replaced with parent membership checks'}
    save(state, 'building_source_SHA_manifest_source_untouched')
    try:
        manifest = rpc('tjucta-a100', SOURCE_CODE,
                       {'mode': 'inventory', 'root': SOURCE_ROOT, 'directories': DIRECTORIES})
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
        MANIFEST.write_text(encoded)
        sha = hashlib.sha256(encoded.encode()).hexdigest()
        save(state, 'source_manifest_ready_source_untouched', source_manifest=MANIFEST.name,
             source_manifest_sha256=sha, original_regular_files=manifest['file_count'],
             original_logical_bytes=manifest['total_logical_bytes'])
        target = rpc('4080', TARGET_PREPARE,
                     {'parent': TARGET_PARENT, 'incoming_bytes': manifest['total_logical_bytes']})
        save(state, 'streaming_retired_originals_to_4T_source_untouched', target=target)
        sender = ('nice -n 10 tar --format=posix -czf - -C ' + shlex.quote(SOURCE_ROOT)
                  + ' -- ' + ' '.join(map(shlex.quote, DIRECTORIES)))
        receiver = ('taskset -c 16-27 nice -n 10 tar -xzf - --no-same-owner -C '
                    + shlex.quote(target['target_staging_directory']))
        with subprocess.Popen([*SSH, 'tjucta-a100', sender], stdout=subprocess.PIPE) as producer:
            try:
                result = subprocess.run([*SSH, '4080', receiver], stdin=producer.stdout, timeout=14400)
            finally:
                producer.stdout.close()
                try:
                    producer.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    producer.terminate()
                    try:
                        producer.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        producer.kill()
                        producer.wait()
            assert producer.returncode == 0 and result.returncode == 0, 'Inspect same archive transfer'
        save(state, 'verifying_4T_copy_source_untouched')
        verified = rpc('4080', TARGET_VERIFY,
                       {'manifest': manifest, 'manifest_sha256': sha,
                        'stage': target['target_staging_directory'], 'final': target['target_archive_directory']})
        unchanged = rpc('tjucta-a100', SOURCE_CODE,
                        {'mode': 'verify', 'root': SOURCE_ROOT, 'directories': DIRECTORIES,
                         'files': {r['relative_path']: r for r in manifest['files']},
                         'manifest_directories': manifest['directories']})
        save(state, 'verified_4T_archive_source_originals_still_untouched',
             target_verification=verified, source_identity_recheck=unchanged)
    except BaseException as error:
        save(state, 'needs_inspection_source_originals_not_removed', error=str(error))
        raise


if __name__ == '__main__':
    main()
