"""Resource-only handoff for explicitly listed existing frozen-main runs.

Never train, resume, select a checkpoint, overwrite evidence or retry a failed
evaluation. Keep the evaluator in its original immutable source directory.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def read_json(path):
    return json.loads(path.read_text())


def admit(project, gpu):
    value = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True, timeout=15)
    free_mib, utilization = map(int, value.strip().split(','))
    load_limit = .6 * len(os.sched_getaffinity(0))
    return (free_mib >= 6144 and utilization <= 85
            and os.getloadavg()[0] <= load_limit
            and shutil.disk_usage(project).free >= 8 * 2**30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--interval-seconds', type=int, default=1800)
    parser.add_argument('--maximum-checks', type=int, default=48)
    args = parser.parse_args()
    if args.interval_seconds < 1800 or args.maximum_checks < 1:
        parser.error('Use half-hour-or-longer checks and a bounded observation window')
    source = args.project / 'code' / args.source_commit
    frozen_path = source / 'configs/star/arm_safe_entropy/main_plan.json'
    frozen = read_json(frozen_path)
    pending = read_json(args.jobs)['jobs']
    identities = [(j['method'], j['seed']) for j in pending]
    if not pending or len(set(identities)) != len(identities):
        parser.error('List each intended existing main run exactly once')
    for job in pending:
        config = read_json(Path(job['run']) / 'run_config.json')
        if (config['method'] != job['method'] or config['training_seed'] != job['seed']
                or config.get('stage') != 'main' or config.get('frozen_config') != frozen
                or job['method'] not in frozen['methods']
                or job['seed'] not in frozen['training_seeds']):
            parser.error('Only dispatch the explicitly identified frozen native main runs')
    emit('dispatcher_started', pid=os.getpid(), jobs=identities,
         interval_seconds=args.interval_seconds, additional_training_transitions=0)
    for observation in range(args.maximum_checks):
        for job in list(pending):
            run, output, log = (Path(job[key]) for key in ('run', 'output', 'log'))
            receipt_path = run / 'receipt.json'
            if not receipt_path.exists():
                try:
                    command = (Path('/proc') / str(job['native_pid']) / 'cmdline').read_bytes()
                except FileNotFoundError as error:
                    raise RuntimeError('Native handle missing without1M receipt; do not restart') from error
                if b'scripts.star.arm_safe_entropy.train' not in command or str(run).encode() not in command:
                    raise RuntimeError('Native handle identity changed; do not restart')
                emit('waiting_native', method=job['method'], seed=job['seed'],
                     native_pid=job['native_pid'])
                continue
            native = read_json(receipt_path)
            if native.get('status') != 'completed' or native.get('steps') != 1000000:
                raise RuntimeError('Do not evaluate an interrupted or selected partial run')
            if output.exists() or log.exists():
                raise RuntimeError('Prior evaluation evidence exists; inspect it, never overwrite/retry')
            if not admit(args.project, args.gpu):
                emit('waiting_shared_resources', method=job['method'], seed=job['seed'])
                continue
            command = [str(args.project / 'env/bin/python'), '-u', '-m',
                'scripts.star.arm_safe_entropy.evaluate_main_robustness',
                '--frozen-config', str(frozen_path), '--plan',
                str(source / 'configs/star/arm_safe_entropy/formal_robustness_plan.json'),
                '--run', str(run), '--layout-validity',
                str(args.project / 'analysis/formal-v3-layout-validity-20261109-r2/layout_validity.json'),
                '--output', str(output), '--source-commit', args.source_commit, '--device', 'cuda:0']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with log.open('xb') as stream:
                process = subprocess.Popen(command, cwd=source, env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
                emit('evaluation_launched', method=job['method'], seed=job['seed'],
                     pid=process.pid, run=str(run), output=str(output), log=str(log), gpu=args.gpu)
                code = process.wait()
            if code:
                raise RuntimeError(f'Evaluation exited{code}; preserve outputs and inspect{log}')
            receipt = read_json(output / 'receipt.json')
            if (receipt.get('status') != 'completed' or receipt.get('condition_rows') != 20
                    or receipt.get('policy_episodes') != 2000
                    or receipt.get('native_nominal_episodes_reused') != 400
                    or receipt.get('new_stress_episodes') != 1600
                    or receipt.get('training_transitions') != 0
                    or receipt.get('adaptation_updates') != 0):
                raise RuntimeError('Incomplete fixed test; never automatically retry or hide its evidence')
            emit('evaluation_completed', method=job['method'], seed=job['seed'],
                 output=str(output), condition_rows=20, policy_episodes=2000)
            pending.remove(job)
        if not pending:
            emit('all_explicitly_queued_frozen_tests_completed')
            return
        if observation + 1 < args.maximum_checks:
            time.sleep(args.interval_seconds)
    raise RuntimeError('Observation window expired; native training was not stopped or restarted')


if __name__ == '__main__':
    main()
