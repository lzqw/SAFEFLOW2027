"""One frozen-state boundary step, no learner update or scientific training."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jp
import numpy as np
import torch
from crax.io.torch import jax_to_torch
from scripts.paper_crax_payload.crax_env import PointGoalTorch
from scripts.paper_crax_payload.crax_models import CumulativeFlowLearner

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.cuda.set_device(0)
    saved=torch.load(args.checkpoint,map_location='cuda:0',weights_only=False)
    env=PointGoalTorch(saved['profile']['E'],saved['seed'],'cuda:0')
    env.load_state_dict(saved['environment'])
    # Artificial elapsed boundary ONLY; native physics/horizon unchanged.
    env.state.info['steps']=jp.full((env.num_envs,),999.,dtype=jp.float32)
    action=torch.zeros((env.num_envs,env.action_dim),device='cuda:0')
    obs,reward,done,info=env.step(action)
    # Do not synchronize before consumers: actual training is asynchronous.
    observed=torch.stack((done.sum(),info['truncated'].sum(),info['terminated'].sum()))
    partial=saved['partial'];partial['cost']+=info['cost']
    settings=saved['learner']['settings']
    learner=CumulativeFlowLearner(env.obs_dim,env.action_dim,'stable_iso','cuda:0',
        width=settings['width'],dual_rule='native_per_step',episode_length=1000,lagrangian_lr=.1)
    learner.load_state_dict(saved['learner'])
    before=done.sum().clone()
    learner.record_completed_episodes(partial['cost'],done)
    after=done.sum().clone()
    for value in partial.values():value.mul_(~done)
    torch.cuda.synchronize()
    result=dict(physical_controls=env.num_envs,new_training_transitions=0,
        real_interactions=0,synthetic_elapsed_boundary=True,
        asynchronous_consumers=True,before=observed.tolist(),
        done_dtype=str(done.dtype),done_unique=done.unique().tolist(),
        reset_steps=np.asarray(env.state.info['steps']).tolist()[:4],
        before_record=before.item(),after_record=after.item(),
        recorded_episode_count=learner.dual_episodes.item(),
        final_partial_length=partial['length'][:4].tolist())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
