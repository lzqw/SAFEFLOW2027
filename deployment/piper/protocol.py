"""Generate the planned interleaved real experiment; never execute it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


METRICS = ['safe_success', 'unassisted_safe_success', 'violation', 'hard_collision',
           'protection_intervention_rate', 'action_modification_m',
           'minimum_distance_m', 'capped_completion_s',
           'inference_p50_s', 'inference_p95_s', 'inference_p99_s', 'deadline_miss_rate']


def reach90_schedule():
    """Current first-stage plan; no Payload/online RL is silently added."""
    methods = ['Iso','Iso-Low','EA']
    rows = []
    for repeat in range(10):
        for ci, condition in enumerate(('Nominal','UnseenGeometry','ExecutionVariation')):
            offset = (repeat+ci) % len(methods)
            for method in methods[offset:]+methods[:offset]:
                rows.append(dict(order=len(rows)+1,task='Reach',condition=condition,
                    method=method,repeat=repeat+1,status='planned_not_executed'))
    return rows


def fixed_policy_schedule(seed=72027):
    rng=random.Random(seed)
    methods=['Iso','Iso-Low','EA']
    groups=[(task,condition) for task in ['Reach','Payload']
            for condition in ['Nominal','UnseenGeometry','UnseenExecutionOrPayload']]
    rows=[]
    for repeat in range(20):
        rng.shuffle(groups)
        for task,condition in groups:
            order=methods.copy()
            rng.shuffle(order)
            for method in order:
                rows.append(dict(order=len(rows)+1,task=task,condition=condition,
                                 method=method,repeat=repeat+1,seed=seed,
                                 status='planned_not_executed'))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=72027)
    parser.add_argument('--protocol',choices=['reach90','legacy360'],default='reach90')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Preserve earlier experiment protocol')
    rows = reach90_schedule() if args.protocol == 'reach90' else fixed_policy_schedule(args.seed)
    result=dict(schema_version=2,protocol=args.protocol,status='planned_not_authorized_for_execution',
                physical_motion_authorized=False,real_interactions=0,
                fixed_policy_episodes=len(rows),rows=rows,
                metrics=METRICS,
                activation_requires='on-site TCP/frame/period/workspace/speed calibration and explicit operator confirmation; no stress fixture automatically copied from simulation',
                finetune=dict(status='conditional_not_launched',
                    requires='separate on-site operator enable and retained adaptation-efficiency claim',
                    task_count=1,branches=['Iso-FT','Iso-Low-FT','EA-FT'],
                    episodes_per_branch=50,desired_independent_repeats=2,
                    evaluation_nodes=[0,10,25,50],frozen_pretrained_reference=True,
                    original_and_measured_executed_actions_required=True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(planned_fixed_policy_episodes=len(rows),real_interactions=0)))


if __name__=='__main__':
    main()
