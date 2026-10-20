#!/usr/bin/env python3
"""Keep source/4T originals; cache verified duplicate host inputs in RAM.

Canonical host paths stay usable. A rebooted/missing RAM cache is reconstructed
from the unchanged durable4T bytes before final accounting; no gate is waived.
"""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
MANIFEST = BASE / 'host_full_input_cache_20261020_manifest.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def manifest(base=BASE):
    m = json.loads((Path(base)/MANIFEST.name).read_text())
    assert m['archive_host']=='4080'
    assert m['host_root']=='/home/lzqw/SAFEFLOW2027_artifacts/recovered_a100_4080/mps_runtime_transitions'
    assert m['cache_root']=='/dev/shm/SAFEFLOW2027_full_input_cache_20261020'
    for row in m['entries']:
        host,cache,archive = map(Path,(row['host_path'],row['cache_path'],row['archive_path']))
        relative=host.relative_to(m['host_root'])
        assert '..' not in relative.parts and relative.name=='latest_full.pt'
        assert cache==Path(m['cache_root'])/relative
        assert archive==Path('/home/carla/4T/SAFEFLOW2027_4080/mps_runtime_transitions')/relative
        assert len(row['sha256'])==64
    return m


def verify_durable(m):
    code='''import hashlib,json\nfrom pathlib import Path\nrows=json.loads(%r)\nfor r in rows:\n p=Path(r['archive_path'])\n assert p.is_file() and not p.is_symlink()\n assert p.stat().st_dev!=Path('/home/carla/SAFEFLOW2027_4080').stat().st_dev\n h=hashlib.sha256()\n with p.open('rb') as f:\n  for b in iter(lambda:f.read(1048576),b''):h.update(b)\n assert h.hexdigest()==r['sha256']\nprint(json.dumps({'verified_durable4T_copies':len(rows)}))\n''' % json.dumps(m['entries'])
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','4080',
                      'python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=300)
    return json.loads(r.stdout)


def install_link(row):
    host,cache=Path(row['host_path']),Path(row['cache_path'])
    assert cache.is_file() and digest(cache)==row['sha256']
    if host.is_symlink():
        assert os.readlink(host)==str(cache)
        return
    assert host.is_file() and digest(host)==row['sha256']
    temporary=host.with_name(host.name+'.verified_ram_link.tmp')
    assert not temporary.exists() and not temporary.is_symlink()
    temporary.symlink_to(cache)
    # Atomic replacement removes only this verified redundant regular host copy.
    # Source and4T originals remain unchanged; the exact bytes exist in RAM too.
    os.replace(temporary,host)


def offload(base=BASE):
    m=manifest(base)
    proof=verify_durable(m)
    freed=0
    for row in m['entries']:
        host,cache=Path(row['host_path']),Path(row['cache_path'])
        assert host.is_file() and digest(host)==row['sha256']
        size=host.stat().st_size
        cache.parent.mkdir(parents=True,exist_ok=True)
        if not cache.exists():
            assert shutil.disk_usage(cache.parent).free>=size+512*1024**2
            part=cache.with_name(cache.name+'.partial');assert not part.exists()
            shutil.copyfile(host,part)
            with part.open('rb') as f:os.fsync(f.fileno())
            assert digest(part)==row['sha256']
            part.rename(cache)
        was_regular=not host.is_symlink()
        install_link(row)
        freed += size if was_regular else 0
        print(json.dumps({'job':row['job'],'host_cache_offloaded':True,'sha256':row['sha256']}),flush=True)
    receipt={**proof,'updated_at_unix':time.time(),'host_duplicate_bytes_released':freed,
             'host_filesystem_free_bytes_after':shutil.disk_usage(m['host_root']).free,
             'canonical_paths_preserved_as_verified_cache_links':True,
             'durable4T_and_source_originals_not_deleted':True,
             'RAM_cache_ephemeral_restore_required_after_reboot':True}
    p=Path(base)/'host_full_input_cache_20261020_offload_receipt.json'
    temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(p)
    print(json.dumps(receipt),flush=True)
    return receipt


def ensure_cache(base=BASE):
    m=manifest(base)
    for row in m['entries']:
        host,cache=Path(row['host_path']),Path(row['cache_path'])
        if host.is_file() and not host.is_symlink():
            assert digest(host)==row['sha256']
            continue
        assert host.is_symlink() and os.readlink(host)==str(cache)
        if not cache.is_file():
            cache.parent.mkdir(parents=True,exist_ok=True)
            assert shutil.disk_usage(cache.parent).free>=512*1024**2+300*1024**2
            part=cache.with_name(cache.name+'.restore.partial');assert not part.exists()
            subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=12',
                            '4080:'+row['archive_path'],str(part)],check=True,timeout=3600)
            assert digest(part)==row['sha256']
            part.rename(cache)
        assert digest(cache)==row['sha256'] and digest(host)==row['sha256']
    return {'host_inputs_materialized_and_verified':len(m['entries'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['offload','ensure'])
    args=parser.parse_args()
    print(json.dumps(offload() if args.action=='offload' else ensure_cache()),flush=True)
