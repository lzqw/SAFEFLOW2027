"""Finite paired whole-policy DEV evaluation; no training or postfailure actions."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import platform

import torch

from scripts.paper_crax_payload.analyze_piper_failures_v12 import paired
from scripts.paper_crax_payload.diagnose_piper_noise_v12 import load
from scripts.paper_crax_payload.native_p5_adaptation import target_deployment
from scripts.paper_crax_payload.summarize_simulation_v12 import output
from scripts.paper_crax_payload.train_native_p5_adaptation import evaluation_config, sha256, storage_check
from scripts.paper_crax_payload.train_native_p5_warmup_v12 import evaluate
from scripts.star.arm_safe_entropy.train import atomic_json

PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_piper_deployment_modes.json'
ROOT = Path('/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12')


@contextmanager
def deployment_mode(learner, mode):
    if mode not in ('sampled','zero_noise'):
        raise ValueError('Use the two predeclared complete-policy modes')
    original = learner.action
    inherited = 'action' not in learner.__dict__
    learner.action = lambda obs: original(obs,deterministic=mode=='zero_noise')
    try:
        yield
    finally:
        if inherited:
            del learner.action
        else:
            learner.action = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True); p.add_argument('--assets-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--source-commit',required=True)
    p.add_argument('--device',default='cuda:0')
    args = p.parse_args(); torch.set_num_threads(1)
    spec = json.loads(args.plan.read_text())
    if spec != json.loads(PLAN.read_text()):
        raise ValueError('Retain all frozen cases/modes/layouts;no outcome-dependent selection')
    if not args.output.resolve().is_relative_to(ROOT.resolve(strict=True)):
        raise ValueError('Keep outputs on the admitted A100 data disk')
    storage_check(ROOT,growth_gib=2)
    device = torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(.025,device)
    assets = json.loads(args.assets_receipt.read_text())
    if sha256(assets['source_urdf'])!=assets['source_sha256'] or sha256(assets['resolved_urdf'])!=assets['resolved_sha256']:
        raise ValueError('Pinned official simulation assets changed')
    args.output.mkdir(exist_ok=False)
    atomic_json(args.output/'run_config.json',dict(protocol=spec,source_commit=args.source_commit,
        plan_sha256=sha256(args.plan),host=platform.node(),native_pid=os.getpid(),device=str(device),
        new_training_steps=0,independent_test_used=False,physical_robot_interactions=0))
    summaries = []; evaluations = {}; provenance = []; work = 0
    for case in spec['cases']:
        directory = ROOT/'deployment-mode-inputs'/case['relative_run']
        resolved = dict(case,checkpoint=str(directory/'full_new_100000.pt'),receipt=str(directory/'receipt.json'))
        learner,native,line = load(resolved,device=args.device)
        if native['model_commit']!=assets['model_commit']:
            raise ValueError('Native checkpoint and official assets must match')
        training = json.loads(Path(case['plan']).read_text())
        cfg = evaluation_config(native,training,line)
        cfg['evaluation']['action_sampling_seed'] = spec['action_seed_start'][case['line']]
        cfg['evaluation']['development_layout_seed_start'] = spec['layout_seed_start'][case['line']]
        provenance.append(dict(case=case['name'],checkpoint_sha256=case['checkpoint_sha256'],
            native_receipt_sha256=sha256(resolved['receipt']),training_plan_sha256=sha256(case['plan'])))
        for mode in spec['deployment_modes']:
            with target_deployment(learner),deployment_mode(learner,mode):
                result = evaluate(learner,cfg,assets,line['task'],spec['episodes_per_condition'],
                    spec['layout_seed_start'][case['line']])
            result.update(case=case['name'],label=case['label'],line=case['line'],deployment_mode=mode,
                interpretation='zero_noise is a separate deployment policy,not the original stochastic-policy result')
            atomic_json(args.output/(case['name']+'__'+mode+'.json'),result)
            for group in result['conditions']:
                records = group['records']; work += sum(r['length'] for r in records)
                evaluations[case['name'],mode,group['condition']] = records
                summaries.append(dict(case=case['name'],label=case['label'],line=case['line'],mode=mode,
                    condition=group['condition'],episodes=len(records),
                    safe_successes=group['safe_success_count'],violations=group['violation_count'],
                    timeouts=group['safe_timeout_count'],stall_count=group['stall_count'],
                    ordinary_terminal_task_successes=sum(r['ordinary_terminal_task_success'] for r in records),
                    mean_return=group['episode_return'],mean_capped_seconds=group['time_to_goal_capped_seconds']))
            atomic_json(args.output/'status.json',dict(state='running',native_pid=os.getpid(),
                evaluated_conditions=len(summaries),planned_conditions=42,
                evaluated_episodes=sum(r['episodes'] for r in summaries),new_simulation_controls=work,new_training_steps=0))
    comparisons = []
    for case in spec['cases']:
        conditions = [r['condition'] for r in summaries if r['case']==case['name'] and r['mode']=='sampled']
        for condition in conditions:
            left = next(r for r in summaries if r['case']==case['name'] and r['mode']=='zero_noise' and r['condition']==condition)
            right = next(r for r in summaries if r['case']==case['name'] and r['mode']=='sampled' and r['condition']==condition)
            comparisons.append(dict(case=case['name'],line=case['line'],condition=condition,
                **paired(evaluations[case['name'],'zero_noise',condition],evaluations[case['name'],'sampled',condition]),
                **{'delta_'+key:left[key]-right[key] for key in ('safe_successes','violations','timeouts','stall_count','mean_return','mean_capped_seconds')}))
    output(args.output/'summary.csv',summaries); output(args.output/'paired_mode_changes.csv',comparisons)
    atomic_json(args.output/'receipt.json',dict(status='completed',source_commit=args.source_commit,provenance=provenance,
        conditions=len(summaries),evaluated_episodes=sum(r['episodes'] for r in summaries),new_simulation_controls=work,
        new_training_steps=0,optimizer_updates=0,independent_test_used=False,training_seed_count=1,
        whole_policy_trajectories=True,best_checkpoint_selected=False,physical_robot_interactions=0,
        scope='Development-only deployment modes;does not change training risk or establish a frozen winner'))
    atomic_json(args.output/'status.json',dict(state='completed',evaluated_episodes=sum(r['episodes'] for r in summaries),
        evaluated_conditions=len(summaries),new_training_steps=0))


if __name__=='__main__': main()
