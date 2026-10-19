#!/usr/bin/env python3
"""Move one existing Point cell between A100 cards after stable capacity evidence."""
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

ROOT = Path('/home/lzqw/SAFEFLOW2027_A100')
BASE = ROOT / 'deploy'
CODE = ROOT / 'code/training-62ca739'
PYTHON = ROOT / 'env/bin/python'
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
SIDE = 'a100_gpu_rebalance_20261019_1951'
JOB = 'point-goal-2030'
D = ROOT / 'runs/shapo_p47_four_env/runs' / JOB
PLAN = BASE / (SIDE + '.json')
STATE = BASE / (JOB + '_' + SIDE + '_state.json')
SESSION = 'safeflow_point_goal_2030_gpu1_balance_20261019_1951'


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


def spare_gpu():
    def query(option):
        return list(csv.reader(subprocess.check_output(
            ['nvidia-smi', option, '--format=csv,noheader,nounits'], text=True, timeout=20).splitlines()))
    row = next([v.strip() for v in r] for r in query('--query-gpu=index,uuid,memory.free,utilization.gpu') if int(r[0]) == 1)
    assert int(row[2]) >= 16384 and int(row[3]) <= 85, 'Target capacity changed; source untouched'
    for gpu, pid, _ in query('--query-compute-apps=gpu_uuid,pid,used_memory'):
        if gpu.strip() != row[1]:
            continue
        args = [v.decode(errors='replace') for v in (Path('/proc') / pid.strip() / 'cmdline').read_bytes().split(b'\0') if v]
        assert 'scripts/star/train_star.py' in args
        Path(args[args.index('--config') + 1]).resolve().relative_to(ROOT.resolve())
    return {'physical_gpu_index': 1, 'uuid': row[1], 'free_mib': int(row[2]),
            'utilization_percent': int(row[3]), 'foreign_gpu_processes': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    assert plan['source_commit'] == SOURCE and plan['job'] == JOB
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['controller_sha256']
    assert plan['total_training_budget'] == 10000000
    assert plan['source_gpu_index'] == 0 and plan['target_gpu_index'] == 1
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert boot == plan['expected_boot_id']
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain'], text=True).strip()

    if args.worker:
        receipt = json.loads((BASE / (JOB + '_' + SIDE + '_receipt.json')).read_text())
        assert not any(w['pid'] != receipt['source_identity']['pid'] for w in workers())
        checkpoint = Path(receipt['checkpoint'])
        with checkpoint.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['checkpoint_sha256']
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '1', 'OMP_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PYTHONNOUSERSITE': '1',
               'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(CODE)}
        command = [str(PYTHON), 'scripts/star/train_star.py', '--config', str(D / 'resolved_formal_config.yaml'),
                   '--seed', '2030', '--steps', '10000000', '--device', 'cuda:0', '--vector-envs', '1',
                   '--batch-size', '256', '--replay-size', '400000', '--steps-per-epoch', '5000',
                   '--log-dir', str(D / 'training'), '--result', str(D / 'receipt.json'),
                   '--no-tensorboard', '--resume', str(checkpoint)]
        with (D / ('train_' + SIDE + '.log')).open('x') as log:
            result = subprocess.run(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
        (D / 'exit_code.txt').write_text(str(result.returncode) + '\n')
        raise SystemExit(result.returncode)

    assert not STATE.exists(), 'Existing transaction requires inspection, not duplicate retry'
    qualification = plan['capacity_qualification']
    assert qualification['status'] == 'available_for_rebalance'
    assert qualification['observed_at_unix'] - qualification['candidate_since_unix'] >= 1800
    assert 0 <= time.time() - qualification['observed_at_unix'] < 180
    current = workers()
    assert len(current) == 1 and current[0]['pid'] == plan['expected_source_pid']
    identity = current[0]
    assert identity['device'] == 'cuda:0' and identity['cuda_visible_devices'] == '0'
    source_args = [v.decode() for v in (Path('/proc') / str(identity['pid']) / 'cmdline').read_bytes().split(b'\0') if v]
    for flag, value in {'--seed': '2030', '--steps': '10000000', '--vector-envs': '1',
                        '--batch-size': '256', '--replay-size': '400000', '--steps-per-epoch': '5000'}.items():
        assert source_args[source_args.index(flag) + 1] == value
    capacity = spare_gpu()
    state = {**plan, 'source_identity': identity, 'source_host': 'tjucta-a100', 'target_host': 'tjucta-a100',
             'source_device': 'cuda:0', 'target_device': 'cuda:0', 'target_cuda_visible_devices': '1',
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
        checkpoint = max(D.glob('training/**/checkpoints/latest_full.pt'), key=lambda p: p.stat().st_mtime_ns)
        checkpoint.resolve().relative_to(D.resolve())
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert 'rng' in payload and 'replay' in payload
        resume = int(payload['counters']['env_steps'])
        resume_cost = float(payload['counters']['cumulative_training_cost'])
        with checkpoint.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        old_csv, row = progress()
        steps, cost = int(float(row['TotalEnvSteps'])), float(row['Train/CumulativeCost'])
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
        spare_gpu()
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
            if len(live) == 1 and live[0]['device'] == 'cuda:0' and live[0]['cuda_visible_devices'] == '1' and str(path) not in state['prior_csvs'] and n >= resume + 5000:
                break
            if not live and time.monotonic() - started > 180 and subprocess.run(['tmux', 'has-session', '-t', SESSION], capture_output=True).returncode:
                raise RuntimeError('Target terminal before first epoch')
            time.sleep(20)
        else:
            raise RuntimeError('No first target epoch yet; do not restart a live target')
        target = live[0]
        gpu_rows = csv.reader(subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
            text=True, timeout=20).splitlines())
        assert any(uuid.strip() == capacity['uuid'] and int(pid.strip()) == target['pid'] for uuid, pid in gpu_rows)
        state.update(phase='first_target_epoch_verified_retiring_source', target_identity=target,
                     target_gpu_uuid_verified=capacity['uuid'],
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
