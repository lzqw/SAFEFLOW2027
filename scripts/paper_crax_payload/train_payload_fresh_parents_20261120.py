"""Fixed native Payload300K parents with two real seeds; no new method scan."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import shutil
from types import SimpleNamespace

import torch

from scripts.paper_crax_payload.piper_train import train
from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import sha256
from scripts.paper_crax_payload.train_reach_crossseed_v12 import remaining_output_reserve
from scripts.star.arm_safe_entropy.train import atomic_json


PROJECT = Path(__file__).resolve().parents[2]
PLAN = PROJECT/'configs/paper_crax_payload/simulation_v12_payload_fresh_parents_20261120.json'


def configuration(plan, seed):
    if plan != json.loads(PLAN.read_text()) or seed not in plan['training_seeds']:
        raise ValueError('Only the two prospectively registered native Payload parents')
    base = PROJECT/plan['base_config']
    if sha256(base) != plan['base_config_sha256']:
        raise ValueError('Keep the original native parent recipe and learner')
    config = deepcopy(json.loads(base.read_text()))
    config['development_seed'] = seed
    config['evaluation']['development_layout_seed_start'] = plan['development_layout_seed_start']
    config['evaluation']['action_sampling_seed'] = plan['action_sampling_seed']
    return config


def required_free_bytes(plan, job, root):
    own = plan['projected_output_gib_per_parent']*1024**3
    if job['existing_crossseed_registration']:
        registration = json.loads((root/job['existing_crossseed_registration']).read_text())
        if (registration['status'] != 'admitted_fixed_crossseed900K_development'
                or registration['runner_source_commit'] != '77a517d5'
                or registration['data_roots_by_host'].get(job['host']) != str(root)):
            raise ValueError('Account for the exact existing A100 campaign, not a substitute registration')
        jobs = [j for j in registration['jobs'] if j['host'] == job['host']]
        if len(jobs) != 3:
            raise ValueError('Reserve all three registered A100 outputs including completed arms')
        other = remaining_output_reserve(root, jobs)
    else:
        other = job['reserved_other_output_gib']*1024**3
    # Do not credit own growing output: preserve the full2GiB at every save.
    return plan['minimum_free_floor_gib']*1024**3+own+other


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'assets-receipt', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    config = configuration(plan, args.seed)
    job = next(j for j in plan['jobs'] if j['seed'] == args.seed)
    root = Path(job['data_root']).resolve(strict=True)
    output = root/'runs-payload-parents-20261120'/f'Payload-EA-{args.seed}'
    if (platform.node() != job['host'] or not Path(job['mount']).is_mount()
            or root.stat().st_dev != Path(job['mount']).stat().st_dev
            or args.output.resolve() != output or args.output.exists()
            or args.device != 'cuda:0' or os.environ.get('CUDA_VISIBLE_DEVICES') != job['visible_devices']):
        raise ValueError('Use the exact new data-disk output and one registered shared GPU')

    def storage_guard():
        if shutil.disk_usage(root).free < required_free_bytes(plan, job, root):
            raise OSError('Preserve40GiB plus all pending existing outputs and full Payload output budget')

    storage_guard()
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible GPU; saved generator index0')
    torch.cuda.set_per_process_memory_fraction(.025, torch.device(args.device))
    directory = root/f'payload-parent-configuration-{args.seed}-20261120'
    directory.mkdir(exist_ok=False)
    path = directory/'configuration.json'
    atomic_json(path, config)
    atomic_json(directory/'registration.json', dict(plan=plan, training_seed=args.seed,
        source_commit=args.source_commit, configuration_sha256=sha256(path),
        plan_sha256=sha256(args.plan), host=platform.node(), output=str(output),
        initialization='fresh_networks_optimizers_replay_dual_environment_rng_no_parent_import',
        saved_cuda_rng_index=0, required_free_bytes=required_free_bytes(plan, job, root),
        new_training_budget=300000, independent_test_used=False,
        scientific_early_stopping=False, automatic_retry=False, downstream_training_started=False))
    native = SimpleNamespace(config=path, assets_receipt=args.assets_receipt, task='Payload', method='EA',
        seed=args.seed, steps=plan['training_steps_each'], source_commit=args.source_commit,
        device=args.device, output=output, engineering=False, resume=None, init_from=None,
        task_transfer=None, adapt_from=None)
    train(native, storage_guard=storage_guard)


if __name__ == '__main__':
    main()
