"""One explicitly queued development run; never launches the full matrix.

Historical filename retained; dependency budget is explicit for later stages.
The caller makes the scientific foundation decision before adding a run.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dependency', type=Path, required=True)
    parser.add_argument('--method', choices=['sac_lag', 'flow_iso', 'flow_oracle', 'flow_learned'], required=True)
    parser.add_argument('--dependency-steps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--oracle-backend', choices=['exact', 'controller_jacobian'], default='exact')
    parser.add_argument('--oracle-workers', type=int, default=8)
    args = parser.parse_args()
    assert not args.output.exists() and not args.state.exists()
    base = Path(__file__).resolve().parents[3]
    args.state.parent.mkdir(parents=True, exist_ok=True)

    def record(**state):
        payload = dict(method=args.method, seed=args.seed, dependency=str(args.dependency),
                       output=str(args.output), source_commit=args.source_commit,
                       observed_at_unix=time.time(), **state)
        temporary = args.state.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        temporary.replace(args.state)

    record(phase='waiting_for_dependency_completion', queued_transitions=300000)
    while not (args.dependency / 'receipt.json').exists():
        if (args.dependency / 'engineering_stop.json').exists():
            record(phase='dependency_superseded_no_restart')
            return
        status = args.dependency / 'status.json'
        if status.exists() and json.loads(status.read_text()).get('state') == 'engineering_interruption':
            record(phase='dependency_engineering_failure_no_restart')
            return
        time.sleep(30)
    receipt = json.loads((args.dependency / 'receipt.json').read_text())
    assert receipt['status'] == 'completed' and receipt['steps'] == args.dependency_steps
    while True:
        output = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
            '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        free, utilization = map(float, output.strip().split(','))
        cpu_headroom = os.getloadavg()[0] < .6 * len(os.sched_getaffinity(0))
        disk = os.statvfs(args.dependency.parent)
        if free > 12 * 1024 and utilization < 85 and cpu_headroom and disk.f_bavail * disk.f_frsize > 8 * 1024**3:
            break
        record(phase='waiting_for_shared_capacity', free_mib=free, utilization=utilization)
        time.sleep(1800)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.state.with_suffix('.log')
    with log.open('ab') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.star.arm_safe_entropy.train',
            '--method', args.method, '--steps', '300000', '--seed', str(args.seed), '--num-envs', '2',
            '--eval-every', '100000', '--eval-episodes', '30', '--output', str(args.output),
            '--source-commit', args.source_commit, '--oracle-backend', args.oracle_backend,
            '--oracle-workers', str(args.oracle_workers)], cwd=base, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    record(phase='launched_one_development_run_only', pid=process.pid, log=str(log), queued_transitions=300000,
           oracle_backend=args.oracle_backend if args.method == 'flow_oracle' else None)


if __name__ == '__main__':
    main()
