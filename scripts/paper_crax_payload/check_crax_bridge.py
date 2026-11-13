"""One bounded actual-GPU semantic/roundtrip check; no candidate training.

Engineering truncation probes set only wrapper elapsed counters to horizon-1;
they do not claim1000 genuine collected steps or change the native task horizon.
All duplicated next-step probes count as physical simulator queries.
"""
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
from jax import numpy as jp
import torch
from crax import envs
from scripts.paper_crax_payload.crax_env import PointGoalTorch,DeviceReplay
from scripts.paper_crax_payload.crax_models import CumulativeFlowLearner
from scripts.paper_crax_payload.native_sac_reference import CompleteReset
from scripts.star.arm_safe_entropy.stability_runtime import capture_rng,restore_rng


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-commit',required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve bridge evidence')
    args.output.mkdir(parents=True);started=time.monotonic();controls=0
    def progress(stage):
        (args.output/'status.json').write_text(json.dumps(dict(stage=stage,physical_controls=controls,
            elapsed_seconds=time.monotonic()-started,pid=os.getpid(),source_commit=args.source_commit),indent=2)+'\n')
        print(stage,controls,flush=True)
    progress('initializing_native_gpu')
    torch.cuda.set_device(0);torch.set_num_threads(2)
    env=PointGoalTorch(2,41991,'cuda:0');obs=env.reset();torch.cuda.synchronize()
    assert obs.device.type=='cuda' and jax.devices()[0].platform=='gpu'
    action=torch.zeros((2,env.action_dim),device='cuda')
    obs,reward,done,info=env.step(action);controls+=2;torch.cuda.synchronize();progress('first_native_step')
    assert info['cost'].device.type=='cuda' and env.native.episode_length==1000
    assert info['terminal_observation'].shape==obs.shape and not done.any()
    # Engineering-only boundary probe; exactly two real physics controls follow.
    env.state.info['steps']=jp.full((2,),999.,dtype=jp.float32)
    learner=CumulativeFlowLearner(env.obs_dim,env.action_dim,'ea','cuda:0',width=32,critic_only_updates=0)
    learner.geometry.profile_full_geometry=True;learner.geometry.set_margin_updates(0)
    replay=DeviceReplay(16,env.obs_dim,env.action_dim,'cuda:0',41992)
    replay.add(obs=obs,action=action,reward=reward,cost=info['cost'],next_obs=info['terminal_observation'],
        terminated=info['terminated'],truncated=info['truncated'],margin=info['margin'])
    for _ in range(2):learner.update(replay.sample(2))
    assert learner.actor_updates==1 and learner.margin_updates==2
    snapshot=dict(environment=env.state_dict(),learner=learner.state_dict(),replay=replay.state_dict(),rng=capture_rng())
    torch.save(snapshot,args.output/'bridge_full.pt')
    policyobs,rew,ended,outcome=env.step(action);controls+=2;torch.cuda.synchronize();progress('actual_truncation_before_restore')
    assert ended.all() and outcome['truncated'].bool().all() and not outcome['terminated'].bool().any()
    assert not torch.equal(policyobs,outcome['terminal_observation'])
    saved=torch.load(args.output/'bridge_full.pt',weights_only=False,map_location='cuda:0')
    env.load_state_dict(saved['environment']);learner.load_state_dict(saved['learner']);replay.load_state_dict(saved['replay']);restore_rng(saved['rng'])
    restored_obs,restored_rew,restored_done,restored_info=env.step(action);controls+=2;torch.cuda.synchronize()
    assert torch.equal(restored_obs,policyobs) and torch.equal(restored_rew,rew) and torch.equal(restored_done,ended)
    assert torch.equal(restored_info['terminal_observation'],outcome['terminal_observation'])
    assert learner.updates==2 and learner.actor_updates==1 and replay.size==2
    progress('actual_flow_full_roundtrip_passed')
    native=CompleteReset(envs.get_environment('safe_goal_point',level=1),1000)
    reset=jax.jit(native.reset);step=jax.jit(native.step)
    state=reset(jax.random.split(jax.random.PRNGKey(41993),2))
    state.info['steps']=jp.full((2,),999.,dtype=jp.float32)
    terminal=step(state,jp.zeros((2,env.action_dim)));controls+=2;jax.block_until_ready(terminal.obs)
    assert bool(jp.all(terminal.done)) and bool(jp.all(terminal.info['truncation']))
    assert bool(jp.any(terminal.obs!=terminal.info['terminal_observation']))
    assert bool(jp.all(terminal.info['paper_total_cost']==terminal.info['cost']))
    again=step(terminal,jp.zeros((2,env.action_dim)));controls+=2;jax.block_until_ready(again.obs)
    assert bool(jp.all(again.info['steps']==1)) and not bool(jp.any(again.done))
    assert bool(jp.all(again.info['episode_metrics']['length']==1))
    progress('native_reference_reset_and_counters_passed')
    result=dict(status='completed',source_commit=args.source_commit,new_training_transitions=0,
        synthetic_learner_updates=2,actual_nontraining_simulator_controls=controls,new_real_robot_interactions=0,
        JAX_Torch_same_gpu=True,device_conversion_no_cpu=True,full_save_restore_passed=True,
        native_horizon_preserved=1000,truncation_elapsed_counter_injected_for_engineering_only=True,
        partial_state_replay_optimizer_RNG_restored=True,effective_config=env.effective_config(),
        elapsed_seconds=time.monotonic()-started,torch_version=str(torch.__version__),jax_version=jax.__version__)
    (args.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
