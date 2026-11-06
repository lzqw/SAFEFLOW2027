"""One-off stop of two owned v1 pilots superseded by a task API correction.

Keep all native files and last full checkpoints. Never signal another project,
never manufacture a completed receipt, and count possible unlogged work.
"""
import csv
import json
import os
from pathlib import Path
import signal
import time


def main():
    root = Path('/home/lzqw/SAFEFLOW2027_ARM/runs/obstacle_development')
    specifications = [(231341, 'sac-lag-seed2027-v1'), (234861, 'flow-iso-seed2027-v1')]
    frozen = []
    try:
        for pid, name in specifications:
            proc = Path('/proc') / str(pid)
            args = [a.decode() for a in (proc / 'cmdline').read_bytes().split(b'\0') if a]
            assert 'scripts.star.arm_safe_entropy.train' in args
            assert args[args.index('--output') + 1] == str(root / name)
            assert args[args.index('--source-commit') + 1] == '6c64aeb0ca46f4cfeaab853d2fc73462df14101c'
            directory = root / name
            assert (directory / 'latest_full.pt').is_file()
            assert not (directory / 'engineering_stop.json').exists()
            while True:
                os.kill(pid, signal.SIGSTOP)
                frozen.append(pid)
                if not (directory / 'latest_full.tmp').exists():
                    break
                os.kill(pid, signal.SIGCONT)
                frozen.remove(pid)
                time.sleep(.1)
            native_status = json.loads((directory / 'status.json').read_text())
            with (directory / 'progress.csv').open(newline='') as stream:
                logged = list(csv.DictReader(stream))
            lower = max(native_status['steps'], int(logged[-1]['steps']))
            receipt = dict(kind='owned_pilot_superseded_by_environment_api_fix', pid=pid,
                job=name, stopped_at_unix=time.time(), source_commit=args[args.index('--source-commit') + 1],
                native_status_before_stop=native_status, last_logged_training_transitions=lower,
                conservative_training_transition_upper_bound=lower + 1000,
                unlogged_tail_not_assumed_zero=True,
                preserved_full_checkpoint=str(directory / 'latest_full.pt'),
                original_files_deleted=0, scientific_completion_claimed=False,
                reason='v1 moving-state FK cache lags q/dq; v2 also fixes complete control-interval minimum labels',
                replacement_scope='fresh matched v2 task for all algorithms; no v1/v2 efficacy pooling')
            (directory / 'engineering_stop.json').write_text(json.dumps(receipt, indent=2) + '\n')
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
            frozen.remove(pid)
            print(json.dumps(receipt), flush=True)
    finally:
        # Never leave an unsuccessfully processed worker suspended.
        for pid in frozen:
            os.kill(pid, signal.SIGCONT)


if __name__ == '__main__':
    main()
