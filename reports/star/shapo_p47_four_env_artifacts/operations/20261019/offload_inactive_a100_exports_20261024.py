#!/usr/bin/env python3
"""Archive six obsolete host-local export sets on A100's separate data disk.

Uses the established byte-preserving exporter-maintenance implementation.
Only exports whose canonical container is absent are eligible. Existing old
notes and checkpoints are preserved together; real export parents stay put.
"""
import hashlib
import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parent
SIDE = 'inactive_a100_export_offload_20261024'
SOURCE = BASE / 'offload_inactive_4080_exports_20261024.py'


def main():
    spec = importlib.util.spec_from_file_location('inactive_export_base', SOURCE)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    remote = helper.REMOTE
    replacements = {
        '/home/carla/SAFEFLOW2027_4080': '/home/lzqw/SAFEFLOW2027_A100',
        '/home/carla/4T/SAFEFLOW2027_4080/recovery_backups/inactive_exports_20261024':
            '/storage_0/SAFEFLOW2027_lzqw_uid1007/recovery_backups/inactive_exports_20261024',
        "jobs=['point-button-2031','point-goal-2028','walker2d-2028']":
            "jobs=['ant-2027','ant-2028','walker2d-2027','walker2d-2029','point-button-2028','point-button-2030']",
        "assert os.getuid()==1000 and os.path.ismount('/home/carla/4T')":
            "assert os.getuid()==1007 and os.path.ismount('/storage_0')\nnamespace=Path('/storage_0/SAFEFLOW2027_lzqw_uid1007')\nassert namespace.is_dir() and not namespace.is_symlink() and namespace.stat().st_uid==1007 and namespace.stat().st_mode&0o077==0",
        "Path('/home/carla/4T')": "Path('/storage_0')",
        "shutil.disk_usage('/home/carla/4T')": "shutil.disk_usage('/storage_0')",
        'st.st_uid==1000': 'st.st_uid==1007',
        'archiving_three_inactive_export_copies': 'archiving_six_inactive_export_copies',
        'three_inactive_export_copies_archived_and_primary_replicas_released':
            'six_inactive_export_copies_archived_and_primary_replicas_released',
        'original_bytes_retained_on_separate_4T': 'original_bytes_retained_on_separate_A100_data_disk',
        'historical_inactive_4080_export_copy_not_current_backup': 'historical_inactive_a100_export_copy_not_current_backup',
        'Exact old bytes remain on4T.': 'Exact old bytes remain on the separate A100 data disk.',
    }
    for old, new in replacements.items():
        assert old in remote, old
        remote = remote.replace(old, new)
    compile(remote, str(__file__) + ':remote', 'exec')
    helper.REMOTE = remote
    helper.STATE = BASE / (SIDE + '_state.json')
    helper.RECEIPT = BASE / (SIDE + '_receipt.json')
    helper.REMOTE_RECEIPT = '/storage_0/SAFEFLOW2027_lzqw_uid1007/recovery_backups/inactive_exports_20261024/relocation_receipt.json'
    # Reuse the one-shot orchestration, but bind its endpoint and expected phase
    # explicitly. No native training or source containers are altered.
    import json
    import shlex
    import subprocess
    import time
    assert not helper.STATE.exists() and not helper.RECEIPT.exists(), 'Inspect prior one-shot operation'
    state = dict(phase='running', helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 base_helper_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), started_at_unix=time.time())
    helper.STATE.write_text(json.dumps(state, indent=2) + '\n')
    try:
        subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'tjucta-a100',
                        'taskset -c 16-27 nice -n 19 ionice -c 3 python3 -c ' + shlex.quote(remote)], check=True, timeout=600)
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', 'tjucta-a100:' + helper.REMOTE_RECEIPT,
                        str(helper.RECEIPT)], check=True, timeout=60)
        record = json.loads(helper.RECEIPT.read_text())
        assert record['phase'] == 'six_inactive_export_copies_archived_and_primary_replicas_released'
        state.update(phase='completed', finished_at_unix=time.time())
    except BaseException as error:
        state.update(phase='needs_inspection_no_automatic_retry', error=str(error))
        raise
    finally:
        helper.STATE.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
