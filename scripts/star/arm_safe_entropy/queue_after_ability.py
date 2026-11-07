"""One explicitly queued development/control run; never launches the full matrix.

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
    parser.add_argument('--method', choices=['sac_lag', 'flow_iso', 'flow_oracle', 'flow_learned',
                                            'flow_iso_low', 'flow_random_aniso'], required=True)
    parser.add_argument('--dependency-steps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--oracle-backend', choices=['exact', 'controller_jacobian'], default='exact')
    parser.add_argument('--oracle-workers', type=int, default=8)
    parser.add_argument('--minimum-free-mib', type=int, default=12288)
    parser.add_argument('--maximum-gpu-utilization', type=float, default=85.)
    parser.add_argument('--mechanism-plan', type=Path)
    parser.add_argument('--reference-variance', type=float)
    parser.add_argument('--reference-match-run')
    args = parser.parse_args()
    control = args.method in {'flow_iso_low', 'flow_random_aniso'}
    if control:
        if args.mechanism_plan is None:
            parser.error('A control requires the evidence-reviewed bounded mechanism plan')
        plan = json.loads(args.mechanism_plan.read_text())
        if (plan.get('status') != 'stage_evidence_reviewed_ready_for_requested_bounded_controls_not_main'
                or args.method not in plan['methods'] or args.seed not in plan['training_seeds']
                or plan['steps_per_seed'] != 300000 or plan['evaluation_episodes'] != 30):
            parser.error('Use only the requested reviewed control cohort and fixed budget')
        if args.method == 'flow_iso_low':
            if (args.reference_variance != plan['calibration']['iso_low_variance']
                    or args.reference_match_run != plan['calibration']['measured_from_diagnostic']):
                parser.error('Use the predeclared measured Iso-Low variance and calibration artifact')
        elif args.reference_variance is not None or args.reference_match_run is not None:
            parser.error('Random-Aniso retains the original .04 reference variance')
    elif args.mechanism_plan is not None or args.reference_variance is not None or args.reference_match_run is not None:
        parser.error('These calibration options are only for the bounded mechanism controls')
    assert not args.output.exists() and not args.state.exists()
    base = Path(__file__).resolve().parents[3]
    args.state.parent.mkdir(parents=True, exist_ok=True)

    def record(**state):
        payload = dict(method=args.method, seed=args.seed, dependency=str(args.dependency),
                       output=str(args.output), source_commit=args.source_commit,
                       stage='mechanism_control' if control else 'obstacle_development',
                       reference_variance=args.reference_variance,
                       reference_match_run=args.reference_match_run,
                       minimum_free_mib=args.minimum_free_mib,
                       maximum_gpu_utilization=args.maximum_gpu_utilization,
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
        if free > args.minimum_free_mib and utilization <= args.maximum_gpu_utilization and cpu_headroom and disk.f_bavail * disk.f_frsize > 8 * 1024**3:
            break
        record(phase='waiting_for_shared_capacity', free_mib=free, utilization=utilization)
        time.sleep(1800)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.state.with_suffix('.log')
    command = [sys.executable, '-u', '-m', 'scripts.star.arm_safe_entropy.train',
        '--method', args.method, '--steps', '300000', '--seed', str(args.seed), '--num-envs', '2',
        '--eval-every', '100000', '--eval-episodes', '30', '--eval-seed-start', '52027',
        '--output', str(args.output), '--source-commit', args.source_commit,
        '--oracle-backend', args.oracle_backend, '--oracle-workers', str(args.oracle_workers)]
    if control:
        command += ['--phase', 'mechanism_control']
    if args.reference_variance is not None:
        command += ['--reference-variance', str(args.reference_variance),
                    '--reference-match-run', args.reference_match_run]
    with log.open('ab') as stream:
        process = subprocess.Popen(command, cwd=base, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    record(phase='launched_one_mechanism_control_run_only' if control else 'launched_one_development_run_only',
           pid=process.pid, log=str(log), queued_transitions=300000,
           oracle_backend=args.oracle_backend if args.method == 'flow_oracle' else None)


if __name__ == '__main__':
    main()
