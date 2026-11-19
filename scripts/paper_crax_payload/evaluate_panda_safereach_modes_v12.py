"""Frozen25D/45D deployment comparisons using native loaders and native physics."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from scripts.paper_crax_payload.compare_piper_deployment_modes_v12 import comparison, metrics, validate_records
from scripts.paper_crax_payload.evaluate_piper_deployment_modes_v12 import deployment_mode
from scripts.paper_crax_payload.native_p5_adaptation import build_native, target_deployment, actor_digest
from scripts.paper_crax_payload.summarize_simulation_v12 import output
from scripts.paper_crax_payload.train_matched_ablation_v12 import build as build_reach
from scripts.paper_crax_payload.train_native_p5_adaptation import evaluation_config, sha256, storage_check
from scripts.paper_crax_payload.train_panda_reference_v12 import attach as attach_panda, evaluate as evaluate_panda
from scripts.paper_crax_payload.train_safereach_early_v12 import deployed as evaluate_reach
from scripts.star.arm_safe_entropy.train import atomic_json

PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_panda_safereach_deployment_modes.json'
ROOT = Path('/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12')


def load(case,device):
    directory = ROOT/'panda-safereach-mode-inputs'/case['relative_run']
    path = directory/case['checkpoint_file']; receipt_path = directory/'receipt.json'
    if sha256(path)!=case['checkpoint_sha256']:
        raise ValueError('Require exactly the registered native fixed endpoint')
    saved = torch.load(path,map_location='cpu',weights_only=False)
    receipt = json.loads(receipt_path.read_text()); plan = json.loads(Path(case['plan']).read_text())
    if (saved['format']!=case['format'] or saved['steps']!=case['steps']
            or not saved['source_commit'].startswith(case['training_source'])
            or receipt['status']!='completed' or receipt['steps']!=case['steps']
            or receipt['source_commit']!=saved['source_commit']):
        raise ValueError('Native receipt/format/source/budget must match')
    if case['line']=='panda_payload':
        line = plan['lines'][case['line']]
        if (saved['plan']!=plan or saved['arm']!=case['arm'] or saved['line']!=case['line']
                or line['seed']!=case['training_seed'] or receipt['new_training_steps']!=100000
                or saved['replay']['data']['obs'].shape[-1]!=45):
            raise ValueError('Require native45D Panda full same-parent reference')
        learner = build_native(saved['configuration'],line,device)
        learner.set_transitions(line['parent_steps']); attach_panda(learner,plan,case['line'],case['arm'])
        learner.load_state_dict(saved['learner']); cfg = evaluation_config(saved['configuration'],plan,line)
    else:
        arm = next(a for a in plan['arms'] if a['name']==case['arm'])
        if (saved['configuration']!=plan or saved['arm']!=arm or saved['seed']!=case['training_seed']
                or receipt['new_training_transitions']!=250000 or receipt['engineering_only']
                or saved['replay']['data']['obs'].shape[-1]!=25):
            raise ValueError('Require the native25D matched full scientific checkpoint')
        learner = build_reach(plan,arm,case['training_seed'],device)
        learner.load_state_dict(saved['learner']); cfg = deepcopy(plan)
    return learner,cfg,dict(case=case['name'],checkpoint_sha256=case['checkpoint_sha256'],
        receipt_sha256=sha256(receipt_path),original_training_source=saved['source_commit'],
        native_training_seed=case['training_seed'])


@contextmanager
def timed_actions(learner):
    original = learner.action; inherited = 'action' not in learner.__dict__; timings = []
    def action(obs):
        if learner.device.type=='cuda': torch.cuda.synchronize(learner.device)
        tick = time.perf_counter_ns(); result = original(obs)
        if learner.device.type=='cuda': torch.cuda.synchronize(learner.device)
        timings.append((time.perf_counter_ns()-tick)/1e6)
        return result
    learner.action = action
    try:
        yield timings
    finally:
        if inherited: del learner.action
        else: learner.action = original


def normalize_reach(result,timings):
    cursor = 0
    for group in result['conditions']:
        for row in group['records']:
            durations = timings[cursor:cursor+row['length']]; cursor += row['length']
            if len(durations)!=row['length']:
                raise ValueError('Every native action needs its actual measured timing')
            row.update(seed=row['layout_seed'],trajectory=row['trajectory_tcp_xyz'],
                time_to_goal_capped_seconds=row['capped_seconds'],
                inference_p50_ms=float(np.percentile(durations,50)),inference_p95_ms=float(np.percentile(durations,95)))
    if cursor!=len(timings):
        raise ValueError('Do not report unused inference calls as behavior actions')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True); p.add_argument('--device',default='cuda:0')
    args = p.parse_args(); spec = json.loads(args.plan.read_text()); torch.set_num_threads(1)
    if spec!=json.loads(PLAN.read_text()): raise ValueError('Use every registered case/mode/layout')
    if not args.output.resolve().is_relative_to(ROOT.resolve(strict=True)):
        raise ValueError('Keep output on the admitted data disk')
    storage_check(ROOT,growth_gib=2)
    device = torch.device(args.device)
    if device.type=='cuda': torch.cuda.set_per_process_memory_fraction(.025,device)
    args.output.mkdir(exist_ok=False)
    atomic_json(args.output/'run_config.json',dict(protocol=spec,source_commit=args.source_commit,
        plan_sha256=sha256(args.plan),host=platform.node(),native_pid=os.getpid(),device=str(device),
        new_training_steps=0,independent_test_used=False))
    summaries = []; episodes = []; evaluations = {}; provenance = []; controls = 0
    for case in spec['cases']:
        learner,cfg,source = load(case,args.device); provenance.append(source)
        cfg['evaluation']['development_layout_seed_start'] = spec['layout_seed_start'][case['line']]
        action_key = 'action_sampling_seed' if case['line']=='panda_payload' else 'action_seed'
        cfg['evaluation'][action_key] = spec['action_seed_start'][case['line']]
        if case['line']=='safereach': cfg['evaluation']['development_episodes_each']=100
        before = (learner.transitions,learner.updates,learner.actor_updates,learner.safety_multiplier,
                  actor_digest(learner.actor),actor_digest(learner.target_actor))
        for mode in spec['deployment_modes']:
            with deployment_mode(learner,mode):
                if case['line']=='panda_payload':
                    with target_deployment(learner): result = evaluate_panda(learner,cfg,100)
                else:
                    with timed_actions(learner) as durations: result = evaluate_reach(learner,cfg,case['steps'])
                    result = normalize_reach(result,durations)
            after = (learner.transitions,learner.updates,learner.actor_updates,learner.safety_multiplier,
                     actor_digest(learner.actor),actor_digest(learner.target_actor))
            if before!=after: raise ValueError('Frozen evaluation changed native models/counters/dual')
            result.update(case=case['name'],label=case['label'],line=case['line'],deployment_mode=mode,
                evaluation_split='new_development_only_not_final_guard_or_independent_test',independent_test_used=False,
                new_training_steps=0,inference_timing_scope='sharedGPU synchronized full action call;not physical robot')
            atomic_json(args.output/(case['name']+'__'+mode+'.json'),result)
            for group in result['conditions']:
                rs = group['records']; validate_records(rs,spec['layout_seed_start'][case['line']],100)
                controls += sum(r['length'] for r in rs)
                evaluations[case['line'],case['label'],mode,group['condition']] = rs
                summaries.append(dict(case=case['name'],label=case['label'],line=case['line'],mode=mode,
                    condition=group['condition'],**metrics(rs)))
                episodes.extend(dict(case=case['name'],line=case['line'],mode=mode,condition=group['condition'],
                    **{k:v for k,v in r.items() if k not in ('trajectory','trajectory_tcp_xyz','layout','control_details')}) for r in rs)
            atomic_json(args.output/'status.json',dict(state='running',native_pid=os.getpid(),
                evaluated_conditions=len(summaries),planned_conditions=34,evaluated_episodes=len(episodes),
                new_simulation_controls=controls,new_training_steps=0))
        del learner
        if device.type=='cuda': torch.cuda.empty_cache()
    mode_changes = []; baselines = []; guard = []
    for row in summaries:
        key = (row['line'],row['label'],row['mode'],row['condition'])
        if row['mode']=='zero_noise':
            other = evaluations[row['line'],row['label'],'sampled',row['condition']]
            mode_changes.append(dict(case=row['case'],condition=row['condition'],**comparison(evaluations[key],other)))
            if row['line']=='panda_payload' and row['label']=='P5':
                ref = metrics(other)
                guard.append(dict(condition=row['condition'],zero_noise_successes=row['safe_successes'],
                    sampled_successes=ref['safe_successes'],drop_no_more2pp=row['safe_successes']>=ref['safe_successes']-2,
                    zero_noise_timeouts=row['timeouts'],sampled_timeouts=ref['timeouts'],
                    zero_noise_stalls=row['stall_count'],sampled_stalls=ref['stall_count'],
                    task_mean95percent_target=row['safe_successes']>=95,independent_confirmation=False))
        if row['label']=='P5':
            for label in ('Metric','Iso') if row['line']=='safereach' else ('Metric',):
                other = evaluations[row['line'],label,row['mode'],row['condition']]
                baselines.append(dict(case=row['case'],mode=row['mode'],condition=row['condition'],
                    baseline=label+'_same_mode',**comparison(evaluations[key],other)))
    if len(summaries)!=34 or len(episodes)!=3400: raise ValueError('All3400 registered episodes required')
    for name,rows in (('summary.csv',summaries),('episodes.csv',episodes),('paired_mode_changes.csv',mode_changes),
                      ('paired_against_baselines_same_mode.csv',baselines),('panda_development_regression.csv',guard)):
        output(args.output/name,rows)
    atomic_json(args.output/'receipt.json',dict(status='completed',source_commit=args.source_commit,provenance=provenance,
        cases=5,modes=2,conditions=34,evaluated_episodes=3400,new_simulation_controls=controls,
        new_training_steps=0,optimizer_updates=0,physical_robot_interactions=0,independent_test_used=False,
        winner_confirmation=False,training_risk_improved=False,
        scope='Per task one training seed;Panda guard DEV only;complete deployment trajectories not training changes'))
    atomic_json(args.output/'status.json',dict(state='completed',evaluated_conditions=34,evaluated_episodes=3400,new_training_steps=0))


if __name__=='__main__': main()
