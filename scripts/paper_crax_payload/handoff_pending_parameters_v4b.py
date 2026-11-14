"""One finite, identity-checked scheduler handoff. Never signal a learner.

Specific to the registered CTAlab queues; preserves all worker processes and
archives queue state before retiring only three waiting/scheduling controllers.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import time


ROOT = Path('/storage_0/SAFEFLOW2027_lzqw_uid1007/nonhardware_v4')
OPERATIONS = ROOT / 'resource-handoff-v4b'


def command(pid):
    return (Path('/proc') / str(pid) / 'cmdline').read_bytes()


def check():
    if platform.node() != 'CTAlab' or os.getuid() != 1007:
        raise ValueError('Wrong host or identity for this one-off transfer')
    specifications = [
        ('queue-gpu0.json', 1825419, 'component_queue_a100_gpu0_v4.json', 1825421, 'full'),
        ('queue-gpu1.json', 1825545, 'component_queue_a100_gpu1_v4.json', 1825547, 'noise-only'),
        ('e08-fsrl/formal-queue-v1.json', 1831775, 'cvpo_queue_a100_v4.json', None, None)]
    snapshots = []
    for path, pid, plan, worker, arm in specifications:
        state = json.loads((ROOT / path).read_text())
        cmd = command(pid)
        if state['pid'] != pid or b'run_queue' not in cmd or plan.encode() not in cmd:
            raise ValueError('Original scheduler identity changed; inspect, do not signal')
        expected = [worker] if worker else []
        if state['live_native_pids'] != expected or state['pending'] != (1 if worker else 4):
            raise ValueError('Queue changed or pending jobs may have started')
        if worker:
            if (b'train_component_study' not in command(worker)
                    or str(ROOT / 'runs' / arm).encode() not in command(worker)
                    or (ROOT / 'runs' / arm / 'receipt.json').exists()):
                raise ValueError('Original worker changed or finished; inspect first')
        elif state['jobs']:
            raise ValueError('Do not retire CVPO if a scientific job started')
        snapshots.append(dict(path=path, pid=pid, worker=worker, arm=arm, state=state))
    if any((ROOT / 'runs' / arm).exists() for arm in ('full-rho010', 'full-rho050')):
        raise ValueError('Never relocate a parameter run that already started')
    if OPERATIONS.exists():
        raise FileExistsError('One-off handoff already staged; no blind retry')
    return snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    snapshots = check()
    if not args.apply:
        print(json.dumps(dict(status='preflight_only', controller_pids=[r['pid'] for r in snapshots])))
        return
    OPERATIONS.mkdir()
    (OPERATIONS / 'before.json').write_text(json.dumps(snapshots, indent=2) + '\n')
    # All targets passed their exact manifest/command checks before any signal.
    for row in snapshots:
        os.kill(row['pid'], signal.SIGTERM)
    time.sleep(.2)
    for row in snapshots:
        path = Path('/proc') / str(row['pid'])
        if path.exists() and (path / 'stat').read_text().split()[2] != 'Z':
            raise RuntimeError('Controller termination not confirmed; do not launch a replacement')
        if row['worker'] and (b'train_component_study' not in command(row['worker'])
                or str(ROOT / 'runs' / row['arm']).encode() not in command(row['worker'])):
            raise RuntimeError('Original training handle no longer matches; inspect, never restart')
    # Existing CVPO observer keeps reading this stable state path after the
    # replacement controller starts; its exact old state remains recoverable.
    (ROOT / 'e08-fsrl/formal-queue-v1.json').rename(OPERATIONS / 'formal-queue-v1.retired.json')
    receipt = dict(status='only_original_schedulers_retired_workers_preserved',
        time_utc=datetime.now(timezone.utc).isoformat(),
        retired_controller_pids=[r['pid'] for r in snapshots],
        preserved_worker_pids=[r['worker'] for r in snapshots if r['worker']],
        parameter_runs_were_never_started=True, foreign_processes_signaled=False,
        training_workers_signaled=False, training_restarts=0,
        science_budget_unchanged=True, replacement_controllers_not_yet_launched=True)
    (OPERATIONS / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
