#!/usr/bin/env python3
"""Extend the existing read-only half-hour monitor to A100's archive disk.

No new sampling loop, extra SSH wave, training signal or scientific change.
The unchanged base main preserves the next due time across a service restart.
"""
import hashlib
import importlib.util
from pathlib import Path

BASE=Path(__file__).resolve().parent
SOURCE=BASE/'watch_gpu_capacity.py'


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='9756ae014bd9a98382a2da3361107d45d92c154aeacd14a903bcbb092411ec82'
    spec=importlib.util.spec_from_file_location('capacity_base',SOURCE)
    watcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(watcher)
    assert watcher.POLL_SECONDS==watcher.STABLE_SECONDS==1800
    watcher.ARCHIVE_STORAGE=dict(watcher.ARCHIVE_STORAGE,
        **{'tjucta-a100':'/storage_0/SAFEFLOW2027_lzqw_uid1007'})
    old="    locations.append(('archive_4t_filesystem',Path(sys.argv[2]),100))"
    new="""    archive_path=Path(sys.argv[2])
    label=('archive_data_filesystem' if archive_path==Path('/storage_0/SAFEFLOW2027_lzqw_uid1007')
           else 'archive_4t_filesystem')
    locations.append((label,archive_path,100))"""
    assert watcher.REMOTE.count(old)==1
    watcher.REMOTE=watcher.REMOTE.replace(old,new)
    old="label=='archive_4t_filesystem' and not separate"
    assert watcher.REMOTE.count(old)==1
    watcher.REMOTE=watcher.REMOTE.replace(old,
        "label in ('archive_4t_filesystem','archive_data_filesystem') and not separate")
    compile(watcher.REMOTE,str(__file__)+':remote','exec')
    watcher.log('Read-only archive disk coverage extended to A100 data disk; unchanged1800-second cadence and no startup probe')
    watcher.main()


if __name__=='__main__':main()
