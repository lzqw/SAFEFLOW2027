#!/usr/bin/env python3
"""Two disposable update-only clients: default scheduling versus private MPS.

Does not change GPU compute mode, existing processes, scientific source, or checkpoints.
The private MPS daemon is stopped only after both disposable clients exit normally.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path('/home/autodrive/LZQW/SAFEFLOW2027')
BASE = ROOT / 'operations/performance_20261020'
GPU = 'GPU-ac315b09-0bb7-2072-efd1-cdfc5fae96a5'
BOOT = '5f46bb0c-b4ec-4802-8f6e-01ac611c0f2f'
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
CHECKPOINT = ROOT / 'runs/shapo_p47_four_env/runs/point-goal-2028/training/STAR-{SafetyPointGoal1-v0}/seed-2028-2026-10-19-23-39-24/checkpoints/latest_full.pt'
SHA = '754b15ee61bac6329c48615efb2d42b6c60d77a4475af595f6c822a239fbcb62'
EXPECTED_GPU_PIDS = {28355, 31275, 51246, 135035, 144563, 155591, 166944, 186026}


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def worker(args):
    import torch
    import yaml
    import omnisafe
    os.sched_setaffinity(0, set(range(16)))
    torch.set_num_threads(1)
    cfg = yaml.safe_load((ROOT / 'runs/shapo_p47_four_env/runs/point-goal-2028/resolved_formal_config.yaml').read_text())
    custom = cfg['custom_cfgs']
    custom['seed'] = 2028
    custom.setdefault('train_cfgs', {}).update(device='cuda:0', torch_threads=1,
        vector_env_nums=1, parallel=1, total_steps=10000000)
    custom.setdefault('algo_cfgs', {}).update(size=400000, batch_size=256,
        steps_per_epoch=5000, resume_from=str(CHECKPOINT))
    custom.setdefault('logger_cfgs', {}).update(log_dir=str(args.directory / ('logs_' + str(args.index))),
        use_tensorboard=False, use_wandb=False)
    agent = omnisafe.Agent('STAR', cfg['env_id'], custom_cfgs=custom).agent
    before = (agent.env_steps, agent.run_env_steps, agent.update_count)
    for _ in range(2):
        agent._update()
    torch.cuda.synchronize()
    save(args.directory / ('ready_' + str(args.index) + '.json'), {'pid': os.getpid()})
    while not (args.directory / 'start.json').exists():
        time.sleep(.05)
    start = json.loads((args.directory / 'start.json').read_text())
    while time.monotonic() < start['start_monotonic']:
        time.sleep(.01)
    begin = time.monotonic()
    updates = 0
    try:
        while time.monotonic() < start['end_monotonic']:
            agent._update()
            updates += 1
    finally:
        torch.cuda.synchronize()
    elapsed = time.monotonic() - begin
    assert (agent.env_steps, agent.run_env_steps) == before[:2]
    assert agent.update_count == before[2] + 2 + updates
    save(args.directory / ('result_' + str(args.index) + '.json'), {
        'pid': os.getpid(), 'updates': updates, 'seconds': elapsed,
        'updates_per_second': updates / elapsed, 'environment_steps': 0,
        'formal_training_data': False, 'MPS_pipe_directory': os.environ['CUDA_MPS_PIPE_DIRECTORY']})


def gpu_state():
    import csv
    rows = list(csv.reader(subprocess.check_output(['nvidia-smi',
        '--query-gpu=uuid,memory.free,compute_mode', '--format=csv,noheader,nounits'], text=True).splitlines()))
    row = next(r for r in rows if r[0].strip() == GPU)
    assert int(row[1]) >= 8192 and row[2].strip() == 'Default'
    rows = csv.reader(subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True).splitlines())
    pids = {int(r[1]) for r in rows if r[0].strip() == GPU}
    assert pids == EXPECTED_GPU_PIDS, 'Cohort changed; no benchmark or daemon start'
    identities = []
    for pid in sorted(pids):
        proc = Path('/proc') / str(pid)
        argv = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        assert 'scripts/star/train_star.py' in argv
        assert argv[argv.index('--config')+1].startswith(str(ROOT) + '/runs/')
        identities.append({'pid': pid, 'start_ticks': int((proc/'stat').read_text().rsplit(') ',1)[1].split()[19])})
    return identities


def control(env, command):
    return subprocess.run(['nvidia-cuda-mps-control'], input=command + '\n', env=env,
                          text=True, capture_output=True, check=True, timeout=15).stdout


def phase(directory, env, mps):
    directory.mkdir()
    children = []
    handles = []
    for index in range(2):
        log = (directory / ('worker_' + str(index) + '.stdout')).open('w')
        handles.append(log)
        children.append(subprocess.Popen([sys.executable, __file__, '--worker',
            '--directory', str(directory), '--index', str(index)], env=env,
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT / 'code'))
    deadline = time.monotonic() + 150
    while not all((directory / ('ready_' + str(i) + '.json')).exists() for i in range(2)):
        assert all(c.poll() is None for c in children), 'Disposable initialization failed; inspect logs'
        assert time.monotonic() < deadline, 'Disposable initialization timed out; inspect same children, do not restart'
        time.sleep(.2)
    proof = {}
    if mps:
        servers = control(env, 'get_server_list')
        server_pids = [int(v) for v in servers.split() if v.isdigit()]
        assert len(server_pids) == 1, 'MPS server connection unproven'
        clients = control(env, 'get_client_list ' + str(server_pids[0]))
        client_pids = {int(v) for v in clients.split() if v.isdigit()}
        assert {c.pid for c in children} <= client_pids, 'Both MPS client connections must be proven'
        proof = {'server_list': servers, 'client_list': clients,
                 'both_disposable_client_PIDs_proven': True}
    begin = time.monotonic() + 1
    save(directory / 'start.json', {'start_monotonic': begin, 'end_monotonic': begin + 12})
    # No client is force-terminated. Each client synchronizes CUDA and exits normally.
    for child in children:
        code = child.wait(timeout=60)
        assert code == 0, 'Disposable worker failed; inspect logs, never change formal trainers'
    for h in handles:
        h.close()
    results = [json.loads((directory / ('result_' + str(i) + '.json')).read_text()) for i in range(2)]
    return {'workers': results, 'aggregate_updates_per_second': sum(r['updates_per_second'] for r in results),
            'MPS_connection_proof': proof}


def main():
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == BOOT
    assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT/'code', text=True).strip() == SOURCE
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == SHA
    before = gpu_state()
    out = Path(tempfile.mkdtemp(prefix='disposable_mps_comparison_', dir=BASE))
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONPATH=str(ROOT/'code'),
               CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['CUDA_MPS_PIPE_DIRECTORY'] = str(out/'nonexistent_no_mps')
    result = {'kind': 'engineering_update_only_comparison_not_scientific_result',
              'output_directory': str(out), 'source_commit': SOURCE,
              'input_sha256': SHA, 'formal_workers_before': before, 'environment_steps': 0,
              'compute_mode_changed': False, 'formal_trainers_restarted_or_modified': False}
    result['default'] = phase(out/'default', env, False)
    gpu_state()
    pipe, log = out/'private_mps_pipe', out/'private_mps_log'
    pipe.mkdir(); log.mkdir()
    env.update(CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(log))
    daemon_env = dict(env, CUDA_VISIBLE_DEVICES=GPU)
    start = subprocess.run(['nvidia-cuda-mps-control','-d'], env=daemon_env,
                           text=True, capture_output=True, timeout=20)
    result['daemon_start'] = {'returncode': start.returncode, 'stdout': start.stdout, 'stderr': start.stderr}
    if start.returncode == 0:
        try:
            result['mps'] = phase(out/'mps', env, True)
        finally:
            # Only shut down OUR private daemon after all its clients have exited.
            clients_exited = all((out/'mps'/('result_'+str(i)+'.json')).exists() for i in range(2))
            result['MPS_disposable_clients_exited'] = clients_exited
            if clients_exited:
                result['private_daemon_quit_stdout'] = control(env, 'quit')
            else:
                result['private_daemon_left_for_identity_checked_recovery'] = True
    if 'mps' in result:
        result['MPS_over_default_update_throughput_ratio_not_full_training_guarantee'] = (
            result['mps']['aggregate_updates_per_second'] / result['default']['aggregate_updates_per_second'])
    result['observed_at_unix'] = time.time()
    result['formal_workers_after'] = gpu_state()
    assert result['formal_workers_after'] == before
    save(out/'comparison.json', result)
    print('SAFEFLOW_MPS_COMPARISON=' + json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        main()
