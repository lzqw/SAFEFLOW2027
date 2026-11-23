"""One prospective holdout of retrospectively DEV-selected models; no training or selection loop."""
import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import platform
import shutil

import torch

from scripts.paper_crax_payload.compare_piper_deployment_modes_v12 import (
    comparison, metrics, validate_records)
from scripts.paper_crax_payload.evaluate_piper_deployment_modes_v12 import deployment_mode
from scripts.paper_crax_payload.native_p5_adaptation import (
    actor_digest, build_native, evaluation_config, target_deployment)
from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import sha256
from scripts.paper_crax_payload.summarize_simulation_v12 import output
from scripts.paper_crax_payload.train_native_p5_warmup_v12 import evaluate
from scripts.star.arm_safe_entropy.train import atomic_json


PROJECT = Path(__file__).resolve().parents[2]
PLAN = PROJECT/'configs/paper_crax_payload/selected_seed_frozen_holdout_20261123.json'


def validate_spec(spec,task):
    if spec != json.loads(PLAN.read_text()) or task not in ('Reach','Payload'):
        raise ValueError('Only the exact prospective models, mode, budgets and new layouts')
    if (spec['deployment_mode'] != 'zero_noise' or spec['episodes_per_condition'] != 200
            or spec['holdout_used_for_selection'] or spec['repeat_after_outcome_for_model_selection']):
        raise ValueError('No outcome-dependent checkpoint, seed, mode or layout selection')
    return spec['tasks'][task]


def load_case(root,case,task,seed):
    path = root/case['relative_run']/case['checkpoint_file']
    directory = path.parent
    if sha256(path) != case['checkpoint_sha256']:
        raise ValueError('Selected fixed endpoint changed; never substitute another checkpoint')
    saved = torch.load(path,map_location='cpu',weights_only=False)
    manifest = json.loads((directory/'run_config.json').read_text())
    receipt = json.loads((directory/'receipt.json').read_text())
    module = importlib.import_module(case['module'])
    line_name = 'piper_reach' if task == 'Reach' else 'piper_payload'
    plan = saved['plan']
    line = plan['lines'][line_name]
    if (saved['format'] != module.FORMAT or saved['steps'] != case['total_steps']
            or saved['line'] != line_name or saved['arm'] != case['arm']
            or line['seed'] != seed or line['task'] != task or saved['plan'] != manifest['plan']
            or saved['source_commit'] != manifest['source_commit']
            or receipt['source_commit'] != saved['source_commit'] or receipt['status'] != 'completed'
            or receipt['steps'] != case['total_steps']
            or receipt['new_training_steps'] != case['new_training_steps']):
        raise ValueError('Require the exact native task/seed/arm/receipt and original full-state identity')
    learner = build_native(saved['configuration'],line,'cuda:0')
    learner.set_transitions(line['parent_steps'])
    module.attach(learner,plan,line_name,case['arm'])
    learner.load_state_dict(saved['learner'])
    cfg = evaluation_config(saved['configuration'],plan,line)
    return learner,cfg,dict(checkpoint_sha256=sha256(path),receipt_sha256=sha256(directory/'receipt.json'),
        manifest_sha256=sha256(directory/'run_config.json'),source_commit=saved['source_commit'],
        historical_training_counts=receipt['segment_counts'],
        historical_training_metrics=receipt.get('training_metrics'),
        training_task=task,training_seed=seed,checkpoint_steps=saved['steps'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('Reach','Payload'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    spec=json.loads(PLAN.read_text());task=validate_spec(spec,args.task)
    root=Path(spec['data_root']).resolve(strict=True)
    expected=root/'selected-seed-frozen-holdout-20261123'/args.task
    if (platform.node()!=spec['host'] or not Path('/home/carla/4T').is_mount()
            or root.stat().st_dev!=Path('/home/carla/4T').stat().st_dev
            or args.output.resolve()!=expected or args.output.exists()
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1):
        raise ValueError('Exact new4080 data-disk output; one sharedGPU only')
    if shutil.disk_usage(root).free<42*1024**3:
        raise OSError('Preserve40GiB plus entire two-task compact output allowance')
    torch.cuda.set_per_process_memory_fraction(.025,torch.device('cuda:0'))
    assets_path=Path('/home/carla/4T/SAFEFLOW2027_PAPER/analysis/piper/assets-ac41fcb.json')
    assets=json.loads(assets_path.read_text())
    if (sha256(Path(assets['source_urdf']))!=assets['source_sha256']
            or sha256(Path(assets['resolved_urdf']))!=assets['resolved_sha256']):
        raise ValueError('Original official simulation assets changed')
    args.output.mkdir(parents=True,exist_ok=False)
    atomic_json(args.output/'run_config.json',dict(protocol=spec,task=args.task,
        source_commit=args.source_commit,plan_sha256=sha256(PLAN),native_pid=os.getpid(),
        host=platform.node(),retrospective_DEV_model_selection=True,prospective_seed_disjoint_holdout=True,
        independent_test_used=True,training_launches=0,physical_robot_interactions=0))
    summaries,episodes,provenance,records=[],[],[],{}
    for case in task['cases']:
        learner,cfg,identity=load_case(root,case,args.task,task['training_seed'])
        cfg=deepcopy(cfg)
        cfg['evaluation']['action_sampling_seed']=task['action_seed_start']
        cfg['evaluation']['development_layout_seed_start']=task['layout_seed_start']
        before=(actor_digest(learner.actor),actor_digest(learner.target_actor),learner.transitions,
                learner.updates,learner.actor_updates,learner.safety_multiplier)
        with target_deployment(learner),deployment_mode(learner,spec['deployment_mode']):
            result=evaluate(learner,cfg,assets,args.task,spec['episodes_per_condition'],task['layout_seed_start'])
        after=(actor_digest(learner.actor),actor_digest(learner.target_actor),learner.transitions,
               learner.updates,learner.actor_updates,learner.safety_multiplier)
        if before!=after:
            raise ValueError('Holdout may not mutate selected models or training counters')
        # The native routine defaults to DEV metadata; the split is a caller
        # protocol choice. Explicitly record this prospective frozen split,
        # not a new policy, training change or retrospective reuse of old tests.
        result.update(case=case['name'],deployment_mode='zero_noise',independent_test_used=True,
            evaluation_split='prospectively_frozen_selected_seed_holdout_20261123',
            retrospective_DEV_model_selection=True,holdout_used_for_selection=False,
            population_robustness_claimed=False,new_training_transitions=0)
        atomic_json(args.output/(case['name']+'.json'),result)
        provenance.append(dict(case=case['name'],**identity))
        for group in result['conditions']:
            rs=group['records'];validate_records(rs,task['layout_seed_start'],200)
            records[case['name'],group['condition']]=rs
            summaries.append(dict(case=case['name'],task=args.task,condition=group['condition'],
                mode='zero_noise',training_seed=task['training_seed'],**metrics(rs)))
            episodes.extend(dict(case=case['name'],task=args.task,condition=group['condition'],
                **{k:v for k,v in r.items() if k!='trajectory'}) for r in rs)
        atomic_json(args.output/'status.json',dict(state='running',evaluated_episodes=len(episodes),
            planned_episodes=1600,new_training_steps=0,independent_test_used=True))
        del learner;torch.cuda.empty_cache()
    candidate,baseline=[c['name'] for c in task['cases']]
    comparisons=[]
    for condition in [c['name'] for c in cfg['evaluation']['conditions']]:
        comparisons.append(dict(task=args.task,condition=condition,mode='zero_noise',
            candidate=candidate,baseline=baseline,
            **comparison(records[candidate,condition],records[baseline,condition])))
    if len(summaries)!=8 or len(episodes)!=1600:
        raise ValueError('All selected models, four conditions and every registered episode required')
    for name,rows in [('summary.csv',summaries),('episodes.csv',episodes),('paired_against_metric.csv',comparisons)]:
        output(args.output/name,rows)
    atomic_json(args.output/'receipt.json',dict(status='completed',task=args.task,source_commit=args.source_commit,
        plan_sha256=sha256(PLAN),evaluated_episodes=1600,conditions=8,provenance=provenance,
        new_training_steps=0,optimizer_updates=0,independent_test_used=True,holdout_used_for_selection=False,
        retrospective_DEV_model_selection=True,training_seed_count=1,
        scope=spec['claim_scope'],all_seed_failures_remain_in_evidence=True,
        global_objective_complete=False,physical_robot_interactions=0))
    atomic_json(args.output/'status.json',dict(state='completed',evaluated_episodes=1600,
        independent_test_used=True,new_training_steps=0))


if __name__=='__main__': main()
