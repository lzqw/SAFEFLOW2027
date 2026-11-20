"""One prospective third independent parent; preserve the first two recipes."""
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
from scripts.paper_crax_payload.train_reach_fresh_parent_v12 import configuration as reference_configuration
from scripts.star.arm_safe_entropy.train import atomic_json


PROJECT = Path(__file__).resolve().parents[2]
PLAN = PROJECT/'configs/paper_crax_payload/simulation_v12_reach_third_parent_20261120.json'


def configuration(plan):
    if plan != json.loads(PLAN.read_text()):
        raise ValueError('Use the prospective fixed third-seed recipe')
    reference = json.loads((PROJECT/plan['reference_plan']).read_text())
    if (reference['base_config_sha256'] != plan['base_config_sha256']
            or reference['training_steps_each'] != plan['training_steps']
            or reference['development_layout_seed_start'] != plan['development_layout_seed_start']
            or reference['action_sampling_seed'] != plan['action_sampling_seed']
            or plan['training_seed'] != 41030 or plan['training_seed'] in reference['training_seeds']):
        raise ValueError('Third seed must preserve the first two parent recipes and DEV identities')
    config = deepcopy(reference_configuration(reference, reference['training_seeds'][0]))
    config['development_seed'] = plan['training_seed']
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'assets-receipt', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    config = configuration(plan)
    root = Path(plan['data_root']).resolve(strict=True)
    if (platform.node() != plan['host'] or not Path('/home/carla/4T').is_mount()
            or root.stat().st_dev != Path('/home/carla/4T').stat().st_dev
            or not args.output.resolve().is_relative_to(root) or args.output.resolve() == root
            or args.output.exists() or args.device != 'cuda:0'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0'):
        raise ValueError('New output on the admitted4080 data disk and one explicit shared GPU only')
    if shutil.disk_usage(root).free < (plan['minimum_free_floor_gib']+plan['projected_output_gib'])*1024**3:
        raise OSError('Preserve40GiB floor plus the full prospective output budget')
    if torch.cuda.device_count() != 1:
        raise ValueError('Bind exactly one visible GPU; preserve saved generator index0')
    torch.cuda.set_per_process_memory_fraction(.025, torch.device(args.device))
    directory = root/'third-parent-configurations-20261120'
    directory.mkdir(exist_ok=False)
    config_path = directory/'reach-EA-41030.json'
    atomic_json(config_path, config)
    atomic_json(directory/'registration.json', dict(plan=plan, source_commit=args.source_commit,
        plan_sha256=sha256(args.plan), reference_plan_sha256=sha256(PROJECT/plan['reference_plan']),
        config_sha256=sha256(config_path), output=str(args.output), host=platform.node(),
        saved_cuda_rng_index=0, changed_recipe_fields=['development_seed'],
        independent_test_used=False, downstream_training_started=False, automatic_retry=False))
    native = SimpleNamespace(config=config_path, assets_receipt=args.assets_receipt, task='Reach', method='EA',
        seed=plan['training_seed'], steps=plan['training_steps'], source_commit=args.source_commit,
        device=args.device, output=args.output, engineering=False, resume=None, init_from=None,
        task_transfer=None, adapt_from=None)
    train(native)


if __name__ == '__main__':
    main()
