"""Necessary CPU contracts; not a substitute for the real MJX/DLPack check.

The replay and reset-mask definitions are AST-loaded directly from crax_env.py
to test their actual implementation without requiring JAX/MuJoCo on the local
host.  The masked PyTree test uses NumPy leaves; GPU conversion/JIT still must
be verified once on the actual server before reported training.
"""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.paper_crax_payload.crax_models import (
    CumulativeFlowLearner, FlowActor, QEnsemble, UpdateGeometry, cost_target)
from scripts.star.arm_safe_entropy.endpoint_geometry import geometry_ramp


def _load_env_definition(name, namespace):
    path=Path(__file__).resolve().parents[2]/'scripts/paper_crax_payload/crax_env.py'
    syntax=ast.parse(path.read_text())
    node=next(node for node in syntax.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name)
    module=ast.Module(body=[node],type_ignores=[])
    exec(compile(module,str(path),'exec'),namespace)
    return namespace[name]


def _tree_map(function, reset, terminal):
    if isinstance(reset,dict):
        return {key:_tree_map(function,reset[key],terminal[key]) for key in reset}
    if isinstance(reset,tuple):
        return tuple(_tree_map(function,new,old) for new,old in zip(reset,terminal))
    return function(reset,terminal)


def test_pre_reset_terminal_observation_is_not_masked_policy_start():
    merge=_load_env_definition('_merge_reset',dict(jp=np,jax=SimpleNamespace(tree=SimpleNamespace(map=_tree_map))))
    terminal=dict(obs=np.array([[11.,12.],[21.,22.],[31.,32.]]),
        info=dict(steps=np.array([1000,30,1000]),cost=np.array([3.,0.,1.])),
        shared_scalar=np.array(5.))
    reset=dict(obs=np.array([[101.,102.],[201.,202.],[301.,302.]]),
        info=dict(steps=np.zeros(3),cost=np.zeros(3)),shared_scalar=np.array(0.))
    original=deepcopy(terminal)
    policy=merge(np.array([True,False,True]),reset,terminal)
    np.testing.assert_array_equal(policy['obs'],[[101.,102.],[21.,22.],[301.,302.]])
    np.testing.assert_array_equal(policy['info']['steps'],[0,30,0])
    np.testing.assert_array_equal(terminal['obs'],original['obs'])
    np.testing.assert_array_equal(terminal['info']['cost'],[3,0,1])
    assert policy['shared_scalar']==5


def test_device_replay_vector_ring_retains_raw_cost_and_terminal_fields():
    Replay=_load_env_definition('DeviceReplay',dict(torch=torch))
    replay=Replay(5,2,2,'cpu',17)
    def block(values):
        values=torch.tensor(values,dtype=torch.float32)
        return dict(obs=values[:,None].expand(-1,2),action=torch.zeros(len(values),2),
            reward=values,cost=values+30,next_obs=values[:,None].expand(-1,2)+1000,
            terminated=torch.zeros(len(values)),truncated=torch.ones(len(values)),margin=-values)
    replay.add(**block([0,1,2]))
    replay.add(**block([3,4,5]))
    assert replay.size==5 and replay.position==1
    assert sorted(replay.data['reward'].tolist())==[1,2,3,4,5]
    assert torch.equal(replay.data['cost'],replay.data['reward']+30)
    assert torch.equal(replay.data['next_obs'][:,0],replay.data['reward']+1000)
    assert torch.equal(replay.data['truncated'],torch.ones(5))
    restored=Replay(5,2,2,'cpu',999)
    restored.load_state_dict(replay.state_dict())
    first=replay.sample(16);second=restored.sample(16)
    assert all(torch.equal(first[k],second[k]) for k in first)


def test_cumulative_cost_target_unbounded_and_time_limit_bootstraps():
    raw=torch.tensor([30.,40.,50.])
    following=torch.tensor([100.,100.,100.])
    # Middle case is a true termination; the third represents a time-limit
    # transition with terminated=0,truncated=1, so it must retain bootstrap.
    terminated=torch.tensor([0.,1.,0.])
    target=cost_target(raw,following,terminated,.99,'cumulative')
    assert target.tolist()==pytest.approx([129.,40.,149.])
    binary=cost_target(torch.tensor([1.]),torch.tensor([.7]),torch.tensor([1.]),.99,'episodic_binary')
    assert binary.item()==1
    with pytest.raises(ValueError):
        cost_target(raw,following,terminated,.99,'implicit')
    q=QEnsemble(3,2,width=8,probability=False)
    with torch.no_grad():
        for head in q.heads:
            for parameter in head.parameters():parameter.zero_()
            head[-1].bias.fill_(7.)
    assert torch.equal(q(torch.zeros(2,3),torch.zeros(2,2)),torch.full((2,2),7.))


@pytest.mark.parametrize('action_dim',[2,6])
def test_flow_dimensions_energy_units_and_sampler_rng(action_dim):
    actor=FlowActor(5,action_dim,width=8)
    actor.velocity=lambda obs,action,time:torch.full_like(action,.2)
    obs=torch.zeros(4,5)
    action,energy,_=actor(obs,deterministic=True)
    assert action.shape==(4,action_dim)
    assert torch.allclose(action,torch.full_like(action,.2))
    assert torch.allclose(energy,torch.full_like(energy,.5*action_dim*.2**2/.04))
    rng=torch.Generator().manual_seed(123)
    state=torch.random.get_rng_state().clone()
    first=actor(obs,generator=rng)[0]
    assert torch.equal(state,torch.random.get_rng_state())
    second=actor(obs,generator=torch.Generator().manual_seed(123))[0]
    assert torch.equal(first,second)
    learner=CumulativeFlowLearner(5,action_dim,device='cpu',width=8)
    assert learner.energy_target==pytest.approx(action_dim*.25/.04)
    assert learner.settings['cost_budget']==25
    assert learner.settings['task_mode']=='cumulative'


def test_dual_uses_completed_full_episodes_only_and_not_preheat():
    learner=CumulativeFlowLearner(3,2,device='cpu',width=8)
    costs=torch.tensor([1000.,50.]); completed=torch.tensor([False,True])
    learner.record_completed_episodes(costs,completed)
    assert learner.dual_episodes.item()==0
    learner.actor_updates=1
    learner.record_completed_episodes(costs,completed)
    assert learner.dual_episodes.item()==1
    assert learner.episode_cost_ema.item()==50
    assert learner.multiplier.item()==pytest.approx(.25)
    before=(learner.episode_cost_ema.clone(),learner.multiplier.clone())
    learner.record_completed_episodes(torch.tensor([100000.,100000.]),torch.tensor([False,False]))
    assert torch.equal(before[0],learner.episode_cost_ema)
    assert torch.equal(before[1],learner.multiplier)


def test_geometry_ramp_is_effective_update_age_and_restored_exactly():
    geometry=UpdateGeometry(torch.nn.Identity(),'endpoint_pullback')
    for updates,expected in [(0,0.),(999,0.),(1000,0.),(3000,.5),(5000,1.),(7000,1.)]:
        geometry.set_margin_updates(updates)
        assert geometry_ramp(geometry.transitions)==expected
    learner=CumulativeFlowLearner(3,2,device='cpu',width=8)
    learner.margin_updates=3000
    learner.geometry.set_margin_updates(3000)
    restored=CumulativeFlowLearner(3,2,device='cpu',width=8)
    restored.load_state_dict(deepcopy(learner.state_dict()))
    assert restored.margin_updates==3000
    assert geometry_ramp(restored.geometry.transitions)==.5
def test_warmup_metrics_do_not_require_a_loss_or_fake_zero_learning():
    from scripts.paper_crax_payload.crax_models import CumulativeFlowLearner
    learner=CumulativeFlowLearner(4,2,device='cpu',width=8)
    metrics=learner.metrics()
    assert metrics['updates']==metrics['actor_updates']==metrics['margin_updates']==0
    assert 'actor_loss' not in metrics and 'reward_loss' not in metrics

