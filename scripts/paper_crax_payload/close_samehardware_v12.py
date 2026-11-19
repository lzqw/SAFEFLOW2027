"""Finite30min receipt wait, then one strict offline summary; no training or git writes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/home/carla/4T/SAFEFLOW2027_SIMULATION_V12')
RUNS = {
    'A-full-P5-2034': ROOT/'runs-samehardware/A-full-P5-2034',
    'A-Iso-matchedLR-2034': ROOT/'runs-samehardware/A-Iso-matchedLR-2034',
    'A-no-lateLR-2034': ROOT/'runs-matched-offload/A-no-lateLR-2034',
    'A-Metric-matchedLR-2034': ROOT/'runs-matched-offload/A-Metric-matchedLR-2034',
}


def terminal_state(queue):
    if queue['status'] == 'completed_with_failures':
        raise RuntimeError('Native training failed; preserve evidence and inspect, never retry')
    return queue['status'] == 'completed' and queue['pending'] == 0 and not queue['live_native_pids']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maximum-checks', type=int, default=24)
    args = p.parse_args()
    if not 1 <= args.maximum_checks <= 48:
        raise ValueError('Bound the observer lifetime to at most24h')
    operation = ROOT/'samehardware-closure'
    operation.mkdir(exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    for check in range(args.maximum_checks):
        queue = json.loads((ROOT/'queue_samehardware.json').read_text())
        complete = terminal_state(queue)
        snapshot = dict(check=check+1, unix_time=time.time(), native_queue_state=queue['status'],
            interval_seconds=1800, ready_for_offline_summary=complete, training_launches=0,
            automatic_retry=False, foreign_processes_changed=False)
        (operation/f'check_{check+1:02d}.json').write_text(json.dumps(snapshot, indent=2)+'\n')
        print(json.dumps(snapshot), flush=True)
        if complete:
            inputs = operation/'inputs'; inputs.mkdir()
            for name, directory in RUNS.items():
                inputs.joinpath(name).symlink_to(directory.resolve(strict=True), target_is_directory=True)
            subprocess.run([sys.executable, '-m', 'scripts.paper_crax_payload.summarize_matched_ablation_v12',
                '--same-hardware', '--runs', str(inputs), '--output', str(ROOT/'samehardware-analysis')],
                cwd=source, check=True)
            print('All four native250K controls summarized; no candidate selected or test run.', flush=True)
            return
        if check+1 < args.maximum_checks:
            time.sleep(1800)
    raise TimeoutError('Finite observation ended without full receipts; training is untouched')


if __name__ == '__main__':
    main()
