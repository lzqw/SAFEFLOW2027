"""Third real training seed, exact same150K learner and same-mode DEV recipe."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import shutil

from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import sha256, TEMPLATE
from scripts.paper_crax_payload.train_reach_crossseed_v12 import (
    CrossseedPiperLearner, evaluate)
from scripts.paper_crax_payload.train_piper_comparison_v12 import attach as attach_native
from scripts.paper_crax_payload.train_native_p5_adaptation import train
from scripts.paper_crax_payload.train_native_p5_warmup_v12 import TrainingTracker


PROJECT = Path(__file__).resolve().parents[2]
PLAN = PROJECT/'configs/paper_crax_payload/simulation_v12_third_seed_confirmation_20261121.json'
FORMAT = 'native_reach_third_seed_confirmation150K_v12_full'


def configuration(spec):
    if spec != json.loads(PLAN.read_text()):
        raise ValueError('Keep the exact registered third400K parent and fixed150K recipe')
    template = json.loads(TEMPLATE.read_text())
    plan = deepcopy(template)
    for key in ('training_seeds','parent_steps','total_new_training_steps','development_layout_seed_start',
                'action_sampling_seed','conditions','parent_rng_cuda_index','training_radius_values'):
        plan.pop(key)
    plan['compiled_training_seed'] = 41030
    plan['template_sha256'] = sha256(TEMPLATE)
    plan['third_seed_registration'] = spec
    plan['lines'] = {'piper_reach':dict(robot='Piper', task='Reach', seed=41030, parent_steps=400000,
        parent_format='paper_native_piper_sim_v1_full', parent_path=spec['parent_path'],
        parent_sha256=spec['parent_sha256'], parent_source_commit=spec['parent_source_commit'],
        parent_rng_cuda_index=0, training_radius_values=[.045,.05,.055],
        development_layout_seed_start=template['development_layout_seed_start'],
        action_sampling_seed=template['action_sampling_seed'], conditions=deepcopy(template['conditions']))}
    return plan


def attach(learner, plan, line, arm):
    attach_native(learner, plan, line, arm)
    learner.__class__ = CrossseedPiperLearner
    learner.crossseed_identity = dict(seed=41030, template_sha256=sha256(TEMPLATE),
        parent_sha256=plan['lines'][line]['parent_sha256'], new_training_budget=150000,
        modes=deepcopy(plan['same_mode_final_deployment_review']), independent_test_used=False)
    return learner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    spec = json.loads(PLAN.read_text())
    plan = configuration(spec)
    root = Path(spec['data_root']).resolve(strict=True)
    parent = Path(spec['parent_path'])
    output = root/'runs-third-confirmation-20261121'/f'Reach-41030-{args.arm}'
    receipt = json.loads((parent.parent/'receipt.json').read_text())
    manifest = json.loads((parent.parent/'run_config.json').read_text())
    if (platform.node() != '447' or not Path('/home/carla/4T').is_mount()
            or args.arm not in plan['arms'] or args.output.resolve() != output or args.output.exists()
            or os.environ.get('CUDA_VISIBLE_DEVICES') != '0'
            or sha256(parent) != spec['parent_sha256']
            or receipt['status'] != 'completed' or receipt['training_steps'] != 400000
            or receipt['initial_steps'] != 300000 or receipt['newly_collected_transitions'] != 100000
            or receipt['training_seed'] != 41030 or receipt['training_source_commit'] != spec['parent_source_commit']
            or manifest['source_commit'] != spec['parent_source_commit']
            or manifest['training_seed'] != 41030):
        raise ValueError('Require the actual complete fixed400K parent on the admitted4080 data disk')
    runtime = root/f'third-confirmation-plan-{args.arm}-20261121.json'
    with runtime.open('x') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')
    def validate(p, native):
        if p != plan or native.resume or native.arm != args.arm:
            raise ValueError('Do not change third-seed scientific settings or silently resume')
    def guard(data_root):
        if data_root != root or shutil.disk_usage(root).free < 48*1024**3:
            raise OSError('Preserve40GiB floor and entire8GiB three-arm plus recovery growth budget')
    args.plan = runtime
    args.parent = parent
    args.assets_receipt = Path('/home/carla/4T/SAFEFLOW2027_PAPER/analysis/piper/assets-ac41fcb.json')
    args.data_root = root
    args.line = 'piper_reach'
    args.resume = None
    args.device = 'cuda:0'
    args.cuda_memory_fraction = .025
    train(args, protocol_validator=validate, adaptation_attacher=attach, format_name=FORMAT,
        piper_evaluator=evaluate, tracker_factory=TrainingTracker, storage_guard=guard)


if __name__ == '__main__':
    main()
