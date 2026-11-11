"""Reuse existing oracle labels to diagnose frozen margin transfer;zero physics.

Historical source actions are not new-policy actions. This retrospective
probe cannot replace the required final common-state/spectrum mechanism.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.star.arm_safe_entropy.stability_runtime import build_learner
from scripts.star.arm_safe_entropy.train import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--native-root',type=Path,required=True)
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--truth',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text());pool_document=json.loads(args.pool.read_text())
    pool=pool_document['records'];truth_document=json.loads(args.truth.read_text())
    truth=np.asarray(truth_document['source_oracle'],dtype=np.float32)
    if (args.output.exists() or len(plan['cases'])!=5 or len(pool)!=64 or truth.shape!=(64,5) or
            not pool_document['episode_isolated_from_training'] or
            plan['training_seed']!=2028 or plan['near_boundary_width_m']!=.02 or
            truth_document['training_transitions']!=0):
        raise ValueError('Use only the declared bounded historical-label reuse;never overwrite a result')
    observations=np.asarray([r['observation'] for r in pool],dtype=np.float32)
    actions=np.asarray([r['source_action'] for r in pool],dtype=np.float32)
    if observations.shape!=(64,25) or actions.shape!=(64,3):raise ValueError('Unexpected state/action schema')
    torch.set_num_threads(1)
    obs=torch.from_numpy(observations);action=torch.from_numpy(actions)
    summaries=[];records=[]
    for case in plan['cases']:
        folder=args.native_root/case['run'];config=json.loads((folder/'run_config.json').read_text())
        step=case['checkpoint_steps']
        if (config['stage']!='ea_stability_v2_development' or config['training_seed']!=2028 or
                config['variant'] not in ('s_ea','s_iso') or step not in (300000,600000,1000000)):
            raise ValueError('Only the explicit same-seed native development policies are in scope')
        if step==1000000 and json.loads((folder/'receipt.json').read_text()).get('steps')!=step:
            raise ValueError('Do not label an incomplete policy as the final endpoint')
        learner=build_learner(config['configuration'],config['variant'],'cpu')
        learner.load_state_dict(torch.load(folder/f'policy_{step}.pt',map_location='cpu',weights_only=False))
        for model_name,model in (('geometry_ema',learner.margin_geometry),('online',learner.margin_online)):
            model.requires_grad_(False)
            normal,_,diagnostic=model.geometry(obs,action,return_diagnostics=True)
            with torch.no_grad():predicted=model(obs,action).mean(0).numpy()
            valid=(truth[:,4]>1e-7)&diagnostic['gradient_valid'].numpy()
            cosine=(normal.numpy()*truth[:,1:4]).sum(-1)
            agreement=diagnostic['agreement'].numpy()
            for condition in sorted({r['condition'] for r in pool}):
                indices=np.asarray([i for i,r in enumerate(pool) if r['condition']==condition])
                actual=truth[indices,0];estimate=predicted[indices];near=np.abs(actual)<=.02
                predicted_safe=estimate<0;direction=indices[valid[indices]]
                summaries.append(dict(variant=config['variant'],training_seed=2028,checkpoint_steps=step,
                    model=model_name,condition=condition,source_states=len(indices),
                    margin_mae_m=float(np.abs(estimate-actual).mean()),bias_predicted_minus_true_m=float((estimate-actual).mean()),
                    near_boundary_states=int(near.sum()),near_boundary_mae_m=float(np.abs(estimate[near]-actual[near]).mean()) if near.any() else None,
                    predicted_safe_states=int(predicted_safe.sum()),true_unsafe_states=int((actual>0).sum()),
                    false_safe_joint_fraction=float(((actual>0)&predicted_safe).mean()),
                    false_safe_given_predicted_safe=float((actual[predicted_safe]>0).mean()) if predicted_safe.any() else None,
                    valid_direction_states=len(direction),direct_action_normal_cosine=float(cosine[direction].mean()) if len(direction) else None,
                    head_agreement=float(agreement[indices].mean()),native_source_commit=config['source_commit']))
            records.append(dict(run=case['run'],checkpoint_steps=step,model=model_name,
                predicted_margin_m=predicted.tolist(),valid_direction=valid.tolist(),
                direct_normal_cosine=[float(cosine[i]) if valid[i] else None for i in range(64)],
                head_agreement=agreement.tolist()))
    atomic_json(args.output,dict(source_commit=args.source_commit,plan=plan,
        pool=str(args.pool),truth=str(args.truth),historical_layout_seed_start=pool_document['layout_seed_start'],
        source_pool_not_new_policy_visitation=True,not_a_formal_unseen_test=True,
        not_a_final_mechanism_substitute=True,training_transitions=0,new_physical_control_intervals=0,
        reused_source_oracle_records=64,summaries=summaries,records=records))
    print(json.dumps(dict(status='completed_frozen_existing_label_probe',summary_rows=len(summaries),
                         training_transitions=0,new_physical_control_intervals=0)),flush=True)


if __name__=='__main__':main()
