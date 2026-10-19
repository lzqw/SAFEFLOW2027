#!/usr/bin/env python3
"""Resume one existing Point cell on spare4080 CPU, preserving full training."""
import argparse
import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/carla/SAFEFLOW2027_4080')
BASE = ROOT / 'deploy'
CODE = ROOT / 'code'
PYTHON = ROOT / 'env/bin/python'
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
SIDE = '4080_gpu_cpu_rebalance_20261019_2330'
JOB = 'point-goal-2028'
D = ROOT / 'runs/shapo_p47_four_env/runs' / JOB
PLAN = BASE / (SIDE + '.json')
STATE = BASE / (JOB + '_' + SIDE + '_state.json')
SESSION = 'safeflow_point_goal_2028_cpu_balance_20261019_2330'


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def workers():
    result = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = [v.decode(errors='replace') for v in path.read_bytes().split(b'\0') if v]
            if 'scripts/star/train_star.py' not in args or args[args.index('--config') + 1] != str(D / 'resolved_formal_config.yaml'):
                continue
            fields = (path.parent / 'stat').read_text().rsplit(') ', 1)[1].split()
            env = dict(v.split(b'=', 1) for v in (path.parent / 'environ').read_bytes().split(b'\0') if b'=' in v)
            result.append({'pid': int(path.parent.name), 'start_ticks': int(fields[19]),
                           'device': args[args.index('--device') + 1], 'state': fields[0],
                           'cuda_visible_devices': env.get(b'CUDA_VISIBLE_DEVICES', b'').decode()})
        except (OSError, ValueError, IndexError):
            continue
    return result


def progress():
    path = max(D.glob('training/**/progress.csv'), key=lambda p: p.stat().st_mtime_ns)
    row = {}
    for row in csv.DictReader(path.open()):
        pass
    return path, row


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cpu_capacity():
    import shutil
    cores = len(os.sched_getaffinity(0))
    load = os.getloadavg()[0]
    available = next(int(line.split()[1]) * 1024 for line in
                     Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    free = shutil.disk_usage(D).free
    assert cores >= 24 and load < cores * .7, 'CPU headroom changed; source untouched'
    assert available >= 8 * 1024**3 and free >= 3 * 1024**3
    return {'cpu_affinity_count': cores, 'load_1m': load,
            'memory_available_bytes': available, 'project_free_bytes': free,
            'cpu_load_limit_fraction': .7, 'minimum_memory_available_bytes': 8 * 1024**3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    assert plan['source_commit'] == SOURCE and plan['job'] == JOB
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['controller_sha256']
    assert plan['total_training_budget'] == 10000000
    assert plan['source_gpu_index'] == 0 and plan['target_device'] == 'cpu'
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert boot == plan['expected_boot_id']
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain'], text=True).strip()

    if args.worker:
        receipt = json.loads((BASE / (JOB + '_' + SIDE + '_receipt.json')).read_text())
        assert not any(w['pid'] != receipt['source_identity']['pid'] for w in workers())
        checkpoint = Path(receipt['checkpoint'])
        assert digest_file(checkpoint) == receipt['checkpoint_sha256']
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1',
               'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(CODE)}
        command = [str(PYTHON), 'scripts/star/train_star.py', '--config', str(D / 'resolved_formal_config.yaml'),
                   '--seed', '2028', '--steps', '10000000', '--device', 'cpu', '--vector-envs', '1',
                   '--batch-size', '256', '--replay-size', '400000', '--steps-per-epoch', '5000',
                   '--log-dir', str(D / 'training'), '--result', str(D / 'receipt.json'),
                   '--no-tensorboard', '--resume', str(checkpoint)]
        with (D / ('train_' + SIDE + '.log')).open('x') as log:
            result = subprocess.run(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
        (D / 'exit_code.txt').write_text(str(result.returncode) + '\n')
        raise SystemExit(result.returncode)

    assert not STATE.exists(), 'Existing transaction requires inspection, not duplicate retry'
    assert plan['runtime_evidence']['historical_same_config_cpu_fps'] > plan['runtime_evidence']['last_shared_gpu_fps']
    assert plan['minimum_resume_steps'] == 2950000
    current = workers()
    assert len(current) == 1 and current[0]['pid'] == plan['expected_source_pid']
    identity = current[0]
    assert identity['start_ticks'] == plan['expected_source_start_ticks']
    assert identity['device'] == 'cuda:0' and identity['cuda_visible_devices'] == '0'
    source_args = [v.decode() for v in (Path('/proc') / str(identity['pid']) / 'cmdline').read_bytes().split(b'\0') if v]
    for flag, value in {'--seed': '2028', '--steps': '10000000', '--vector-envs': '1',
                        '--batch-size': '256', '--replay-size': '400000', '--steps-per-epoch': '5000'}.items():
        assert source_args[source_args.index(flag) + 1] == value
    capacity = cpu_capacity()
    state = {**plan, 'source_identity': identity, 'source_host': '4080', 'target_host': '4080',
             'source_device': 'cuda:0', 'target_device': 'cpu', 'target_cuda_visible_devices': '-1',
             'boot_id': boot, 'phase': 'preparing', 'target_capacity_at_launch': capacity,
             'scientific_parameters_changed': False, 'additional_evaluation': False,
             'original_gpu_logs_preserved': True, 'bitwise_equivalence_claimed': False}

    def save():
        state['updated_at_unix'] = time.time()
        write(STATE, state)
        write(D / (SIDE + '_state.json'), state)

    def source_signal(kind):
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot
        observed = next(w for w in workers() if w['pid'] == identity['pid'])
        assert all(observed[k] == identity[k] for k in ('pid', 'start_ticks', 'device', 'cuda_visible_devices'))
        os.kill(identity['pid'], kind)

    def targets():
        return [w for w in workers() if w['pid'] != identity['pid']]

    save()
    paused = retired = launch_requested = False
    exit_path = D / 'exit_code.txt'
    try:
        source_signal(signal.SIGSTOP)
        paused = True
        state.update(phase='source_paused_verifying_full_checkpoint', source_paused=True,
                     launch_requested=False)
        save()
        for _ in range(50):
            if next(w for w in workers() if w['pid'] == identity['pid'])['state'] == 'T':
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Owned source pause not confirmed')
        import torch
        torch.set_num_threads(1)
        assert str(torch.__version__) == '2.7.1+cu118'
        checkpoint = max(D.glob('training/**/checkpoints/latest_full.pt'), key=lambda p: p.stat().st_mtime_ns)
        checkpoint.resolve().relative_to(D.resolve())
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert 'rng' in payload and 'replay' in payload
        resume = int(payload['counters']['env_steps'])
        resume_cost = float(payload['counters']['cumulative_training_cost'])
        digest = digest_file(checkpoint)
        old_csv, row = progress()
        steps, cost = int(float(row['TotalEnvSteps'])), float(row['Train/CumulativeCost'])
        assert resume >= plan['minimum_resume_steps']
        assert 0 <= steps - resume < 50000 and cost >= resume_cost
        state.update(phase='source_paused_target_starting', checkpoint=str(checkpoint), checkpoint_sha256=digest,
                     resume_steps=resume, resume_cumulative_training_cost=resume_cost,
                     last_logged_source_steps=steps, last_logged_source_cumulative_training_cost=cost,
                     known_repeated_steps=steps-resume, known_additional_cost=cost-resume_cost,
                     full_replay_and_rng_present=True, unknown_unlogged_tail_not_zero=True,
                     physical_repeated_steps_and_cost_unknown_not_zero=True,
                     prior_csvs=[str(p) for p in D.glob('training/**/progress.csv')])
        write(BASE / (JOB + '_' + SIDE + '_receipt.json'), state)
        write(D / (SIDE + '.json'), state)
        if exit_path.exists():
            exit_path.rename(D / ('exit_code.before_' + SIDE + '.txt'))
        if (D / 'receipt.json').exists():
            (D / 'receipt.json').rename(D / ('receipt.before_' + SIDE + '.json'))
        cpu_capacity()
        command = shlex.join([str(PYTHON), str(Path(__file__).resolve()), '--worker'])
        command += ' >>' + shlex.quote(str(BASE / (JOB + '_' + SIDE + '_launcher.log'))) + ' 2>&1'
        launch_requested = True
        state.update(phase='target_launch_requested_source_paused', launch_requested=True,
                     target_session=SESSION)
        save()
        subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, command], check=True)
        state.update(phase='target_starting_source_paused', target_session=SESSION)
        save()
        deadline = time.monotonic() + 3600
        started = time.monotonic()
        while time.monotonic() < deadline:
            live = targets()
            path, row = progress()
            n = int(float(row.get('TotalEnvSteps', -1)))
            if len(live) == 1 and live[0]['device'] == 'cpu' and live[0]['cuda_visible_devices'] == '-1' and str(path) not in state['prior_csvs'] and n >= resume + 5000:
                break
            if not live and time.monotonic() - started > 180 and subprocess.run(['tmux', 'has-session', '-t', SESSION], capture_output=True).returncode:
                raise RuntimeError('Target terminal before first epoch')
            time.sleep(20)
        else:
            raise RuntimeError('No first target epoch yet; do not restart a live target')
        target = live[0]
        assert target['device'] == 'cpu' and target['cuda_visible_devices'] == '-1'
        state.update(phase='first_target_epoch_verified_retiring_source', target_identity=target,
                     target_cpu_placement_verified=True,
                     first_verified_target_steps=n, first_verified_target_fps=float(row['Time/FPS']),
                     first_verified_target_csv=str(path),
                     first_verified_target_cumulative_training_cost=float(row['Train/CumulativeCost']))
        save()
        source_signal(signal.SIGTERM)
        source_signal(signal.SIGCONT)
        paused = False
        for _ in range(60):
            if all(w['pid'] != identity['pid'] for w in workers()):
                break
            time.sleep(1)
        else:
            raise RuntimeError('Source retirement not confirmed')
        retired = True
        # The old wrapper writes its exit marker after the source trainer exits.
        # Wait for that write before archiving it, so it cannot appear late.
        for _ in range(60):
            if exit_path.is_file():
                break
            time.sleep(1)
        else:
            raise RuntimeError('Retired source wrapper exit marker not observed')
        assert any(w['pid'] == target['pid'] and w['start_ticks'] == target['start_ticks'] for w in targets())
        exit_path.rename(D / ('exit_code.retired_source_' + SIDE + '.txt'))
        if (D / 'receipt.json').exists():
            (D / 'receipt.json').rename(D / ('receipt.retired_source_' + SIDE + '.json'))
        state.update(phase='handoff_completed_first_target_epoch_verified', source_retired=True,
                     source_paused=False,
                     target_identity=target, first_verified_target_steps=n, first_verified_target_fps=float(row['Time/FPS']),
                     first_verified_target_csv=str(path), canonical_container_unchanged=True)
        save()
    except Exception as error:
        live = targets()
        session_absent = subprocess.run(['tmux', 'has-session', '-t', SESSION], capture_output=True).returncode != 0
        if paused and not live and (not launch_requested or session_absent):
            if launch_requested:
                if exit_path.exists():
                    exit_path.rename(D / ('exit_code.failed_target_' + SIDE + '.txt'))
                if (D / 'receipt.json').exists():
                    (D / 'receipt.json').rename(D / ('receipt.failed_target_' + SIDE + '.json'))
            try:
                source_signal(signal.SIGCONT)
                paused = False
            except Exception as resume_error:
                state['source_resume_error'] = str(resume_error)
        state.update(phase='needs_inspection_after_rebalance_error', source_paused=paused,
                     source_retired=retired, target_workers=live, launch_requested=launch_requested, error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
