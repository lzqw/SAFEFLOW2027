"""Generate the planned interleaved real experiment; never execute it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


METRICS = ['safe_success', 'unassisted_safe_success', 'violation', 'hard_collision',
           'protection_intervention_rate', 'action_modification_m',
           'minimum_distance_m', 'capped_completion_s',
           'inference_p50_s', 'inference_p95_s', 'deadline_miss_rate']


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
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Preserve earlier experiment protocol')
    result=dict(schema_version=1,status='planned_not_authorized_for_execution',
                physical_motion_authorized=False,real_interactions=0,
                fixed_policy_episodes=360,rows=fixed_policy_schedule(args.seed),
                metrics=METRICS,
                finetune=dict(status='conditional_not_launched',
                    requires='separate on-site operator enable and retained adaptation-efficiency claim',
                    task_count=1,branches=['Iso-FT','Iso-Low-FT','EA-FT'],
                    episodes_per_branch=50,desired_independent_repeats=2,
                    evaluation_nodes=[0,10,25,50],frozen_pretrained_reference=True,
                    original_and_measured_executed_actions_required=True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(planned_fixed_policy_episodes=360,real_interactions=0)))


if __name__=='__main__':
    main()
