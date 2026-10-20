#!/usr/bin/env python3
"""Deduplicate only byte-identical A100 backup copies with our backup writer idle."""
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
RECEIPT = BASE / 'A100_backup_duplicate_only_deduplication_20261020_receipt.json'
UNIT = 'safeflow-recovered-checkpoint-backup.service'
ROOT = '/home/lzqw/SAFEFLOW2027_A100/recovery_backups/from_4080'


def identity(pid):
    p = Path('/proc') / str(pid)
    stat = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
    args = (p / 'cmdline').read_bytes().split(bytes([0]))
    assert str(BASE / 'backup_two_hosts.py').encode() in args
    return int(stat[19]), stat[0]


def main():
    assert not RECEIPT.exists(), 'Never replay an already-recorded maintenance operation'
    pid = int(subprocess.check_output(['systemctl', '--user', 'show', UNIT, '-p', 'MainPID', '--value'], text=True))
    ticks, initial_state = identity(pid)
    assert initial_state not in ('T', 't'), 'Do not resume a process paused by someone else'
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    helper = (BASE / 'deduplicate_backup_copies.py').read_text()
    receipt = {'kind': 'duplicate_only_backup_maintenance_not_training_or_scientific_audit',
               'started_at_unix': time.time(), 'phase': 'waiting_for_own_backup_writer_idle',
               'local_backup_pid': pid, 'local_backup_start_ticks': ticks, 'local_boot_id': boot,
               'remote_host': 'tjucta-a100', 'exact_remote_root': ROOT,
               'helper_sha256': hashlib.sha256(helper.encode()).hexdigest(),
               'remote_training_processes_signalled': 0, 'other_projects_modified': False,
               'distinct_checkpoints_removed': 0, 'additional_scientific_rollouts': 0}
    paused = False
    try:
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            assert identity(pid)[0] == ticks
            children = Path(f'/proc/{pid}/task/{pid}/children')
            if not children.read_text().strip():
                os.kill(pid, signal.SIGSTOP)
                paused = True
                for _ in range(20):
                    if identity(pid)[1] in ('T', 't'):
                        break
                    time.sleep(.05)
                assert identity(pid) == (ticks, 'T')
                if not children.read_text().strip():
                    break
                os.kill(pid, signal.SIGCONT)
                paused = False
            time.sleep(5)
        else:
            raise RuntimeError('Own backup writer remains busy; no remote deduplication attempted')
        receipt['phase'] = 'own_idle_backup_writer_held_for_duplicate_only_operation'
        receipt['backup_writer_held_at_unix'] = time.time()
        remote = (
            'import json,shutil,sys\n'
            + 'root=' + repr(ROOT) + '\n'
            + 'print(json.dumps({"filesystem_free_bytes_before":shutil.disk_usage(root).free}),flush=True)\n'
            + 'sys.argv=["deduplicate_backup_copies.py","--root",root,"--apply"]\n'
            + 'scope={"__name__":"__main__"}\n'
            + 'exec(compile(' + repr(helper) + ',"duplicate_only_helper","exec"),scope)\n'
            + 'print(json.dumps({"filesystem_free_bytes_after":shutil.disk_usage(root).free}),flush=True)\n')
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                                 '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                                 'tjucta-a100', 'python3 -c ' + shlex.quote(remote)],
                                capture_output=True, text=True, timeout=180)
        receipt.update(remote_returncode=result.returncode, remote_stdout=result.stdout, remote_stderr=result.stderr)
        result.check_returncode()
        records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        summary = next(r for r in records if 'deduplicated_copies' in r)
        receipt.update(phase='duplicate_only_operation_completed', summary=summary,
                       duplicate_paths_byte_content_preserved=True, originals_and_unique_checkpoints_preserved=True)
    except BaseException as error:
        receipt.update(phase='needs_inspection_no_automatic_retry', error=str(error))
        raise
    finally:
        if paused:
            assert identity(pid)[0] == ticks, 'Never signal a reused PID'
            os.kill(pid, signal.SIGCONT)
            receipt['original_backup_writer_resumed_at_unix'] = time.time()
        receipt['finished_at_unix'] = time.time()
        RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({'receipt': RECEIPT.name, 'phase': receipt['phase'],
                          'summary': receipt.get('summary'), 'backup_writer_resumed': paused}), flush=True)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise SystemExit('Interrupted; resume only our held backup writer')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
