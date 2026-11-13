"""Bounded Piper profiles then the fixed paired100K actor-only transfer.

Receipts and one task status file are authoritative; no retraining on restart.
Continuation/curriculum/geometry experiments require a later explicit phase.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.star.arm_safe_entropy.train import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--assets-receipt', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    plan = json.loads(Path('configs/paper_crax_payload/robot_gap_v2.json').read_text())
    base = json.loads(Path('configs/paper_crax_payload/piper_sim_v1.json').read_text())
    root = args.root.resolve()
    if root != Path(plan['data_root']):
        raise ValueError('Use the declared A100 data-disk root')
    if (root/'campaign_status.json').exists():
        raise FileExistsError('Do not duplicate or silently restart an existing campaign')
    if os.stat(root).st_dev == os.stat('/home/lzqw').st_dev:
        raise ValueError('Large outputs must be on a separate data filesystem')
    status = dict(protocol=plan['protocol'], source_commit=args.source_commit,
        controller_pid=os.getpid(), started_unix=time.time(), state='running',
        cap=plan['new_training_cap'], launched_training_target=0,
        observed_completed_training=0, physical_robot_interactions=0,
        pointgoal_new_training=0, jobs=[], later_phases_not_launched=True)
    (root/'configs').mkdir(exist_ok=True)
    (root/'logs').mkdir(exist_ok=True)
    (root/'runs').mkdir(exist_ok=True)

    def save():
        status['updated_unix'] = time.time()
        atomic_json(root/'campaign_status.json', status)

    def launch(name, config, method, device, steps, engineering=False, parent=None):
        if shutil.disk_usage(root).free < plan['storage_headroom_bytes']:
            raise RuntimeError('Data disk headroom below40GiB; preserve existing data')
        cfg = root/'configs'/f'{name}.json'
        atomic_json(cfg, config)
        out = root/'runs'/name
        if out.exists():
            raise FileExistsError(f'Existing run must be inspected, not replaced: {out}')
        command = [sys.executable, '-u', '-m', 'scripts.paper_crax_payload.piper_train', 'train',
            '--config', str(cfg), '--assets-receipt', str(args.assets_receipt), '--output', str(out),
            '--device', device, '--task', 'Payload', '--method', method, '--steps', str(steps),
            '--seed', str(base['development_seed']), '--source-commit', args.source_commit]
        if engineering:
            command.append('--engineering')
        if parent:
            command.extend(['--init-from', parent, '--task-transfer', 'Reach_to_Payload'])
        job = dict(name=name, command=command, target_steps=steps, output=str(out),
            status='running', engineering=engineering, started_unix=time.time())
        status['launched_training_target'] += steps
        if status['launched_training_target'] > plan['new_training_cap']:
            raise ValueError('Campaign training reservation exceeds total cap')
        with (root/'logs'/f'{name}.log').open('x') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        job['pid'] = process.pid
        status['jobs'].append(job); save()
        return process, job

    def finish(process, job):
        code = process.wait()
        job.update(exit_code=code, finished_unix=time.time())
        if code:
            job['status'] = 'engineering_failure_no_automatic_restart'; save()
            raise RuntimeError(f'{job["name"]} failed; inspect actual prefix before any retry')
        receipt = json.loads((Path(job['output'])/'receipt.json').read_text())
        if receipt['newly_collected_transitions'] != job['target_steps'] or receipt['status'] != 'completed':
            raise ValueError('Full budget receipt mismatch')
        job.update(status='completed', actual_steps=receipt['newly_collected_transitions'],
            complete_training_sps=receipt['complete_training_sps'], timing=receipt['timing'],
            learner_updates=receipt['learner_updates'], reward_updates=receipt['reward_updates'],
            actor_updates=receipt['actor_updates'])
        status['observed_completed_training'] += receipt['newly_collected_transitions']; save()

    try:
        profiles = []
        for profile in plan['profiles']:
            config = deepcopy(base)
            config.update(num_envs=profile['num_envs'], parallel_environments=profile['parallel_environments'],
                profile_full_geometry=True, synchronized_profile_timing=True,
                checkpoint_fractions=[], checkpoint_every=plan['profile_steps_each'])
            config['shared_learner'].update(batch_size=profile['batch_size'],
                updates_per_vector_step=profile['updates_per_vector_step'],
                batched_geometry=profile['batched'], batched_targets=profile['batched'])
            config['evaluation']['conditions'] = [base['evaluation']['conditions'][0]]
            name = 'profile-'+profile['name']
            process, job = launch(name, config, 'EA', 'cuda:0', plan['profile_steps_each'], engineering=True)
            finish(process, job)
            evaluation = root/'runs'/name/'profile_nominal_evaluation.json'
            subprocess.run([sys.executable, '-m', 'scripts.paper_crax_payload.piper_train', 'evaluate',
                '--config', str(root/'configs'/f'{name}.json'), '--assets-receipt', str(args.assets_receipt),
                '--checkpoint', str(root/'runs'/name/'latest_full.pt'), '--device', 'cuda:0',
                '--split', 'development', '--episodes', str(plan['profile_nominal_episodes_each']),
                '--output', str(evaluation)], check=True)
            score = json.loads(evaluation.read_text())['conditions'][0]
            job.update(profile_nominal_safe_success=score['safe_success'],
                       profile_nominal_return=score['episode_return'])
            profiles.append((profile, job)); save()
        best_success = max(job['profile_nominal_safe_success'] for _, job in profiles)
        eligible = [(p, j) for p, j in profiles if j['profile_nominal_safe_success'] >= best_success-.1-1e-9]
        chosen, measurement = max(eligible, key=lambda pair: pair[1]['complete_training_sps'])
        status['profile_decision'] = dict(selected=chosen, rule=plan['profile_selection'],
            all_short_profiles_zero_success=best_success==0,
            full_geometry_forced_in_engineering_profiles=True,
            shared_foreign_GPU_load_not_controlled=True,
            no_claim_of_optimization_equivalence=True)
        save()
        transfer_config = deepcopy(base)
        transfer_config.update(num_envs=chosen['num_envs'],
            parallel_environments=chosen['parallel_environments'],
            profile_full_geometry=False, synchronized_profile_timing=False)
        transfer_config['shared_learner'].update(batch_size=chosen['batch_size'],
            updates_per_vector_step=chosen['updates_per_vector_step'],
            batched_geometry=chosen['batched'], batched_targets=chosen['batched'])
        # Identical task/config/data/update budgets; only the declared EA vs Iso
        # geometry differs. Parent remains fixed, irrespective of Payload scores.
        active = []
        for method in plan['transfer_methods']:
            active.append(launch('transfer-'+method.lower()+'-100k', transfer_config, method,
                f'cuda:{plan["transfer_gpu_ids"][method]}', plan['transfer_steps_each'],
                parent=plan['transfer_parent']['path']))
        for process, job in active:
            finish(process, job)
        status['state'] = 'initial_phase_completed_pending_scientific_decision'
        status['remaining_later_phase_budget_not_automatically_spent'] = (
            plan['new_training_cap']-status['launched_training_target'])
        save()
    except BaseException as error:
        status.update(state='needs_inspection_no_automatic_restart', error=str(error))
        # Do not kill a healthy paired sibling if one job fails.
        save()
        raise


if __name__ == '__main__':
    main()
