import io
import random

import numpy as np
import torch

from scripts.star.arm_safe_entropy.stability_models import StableArmLearner


def batch(size=8):
    return dict(obs=torch.randn(size,25),next_obs=torch.randn(size,25),
        action=torch.rand(size,3)*2-1,reward=torch.randn(size),done=torch.zeros(size),
        cost=torch.zeros(size),margin=torch.randn(size)*.02)


def test_shared_package_and_dual_accumulation_wait_for_actor_learning():
    torch.set_num_threads(1)
    learners=[StableArmLearner(variant=v,device='cpu',width=16,critic_only_updates=2)
              for v in ('s_iso','s_local','s_ea')]
    assert all(l.settings==learners[0].settings for l in learners)
    for learner in learners:
        data=batch()
        learner.record_episode(1.)
        assert learner.episode_cost_ema is None and learner.safety_multiplier==0
        learner.update(data);learner.update(data)
        assert learner.actor_updates==0 and learner.reward_updates==4
        learner.record_episode(1.)
        assert learner.episode_cost_ema is None
        learner.update(data);learner.update(data)
        assert learner.actor_updates==1 and learner.reward_updates==8
        learner.record_episode(1.)
        assert learner.episode_cost_ema==1. and learner.safety_multiplier>0
        assert learner.warmup_safety_episodes==2


def test_geometry_snapshot_is_fixed_across_entire_round_then_one_ema():
    torch.set_num_threads(1)
    learner=StableArmLearner(device='cpu',width=16,critic_only_updates=0)
    learner.set_transitions(50000)
    before=[p.detach().clone() for p in learner.margin_geometry.parameters()]
    seen=[]
    original=learner.geometry.query
    def query(*args,**kwargs):
        seen.append(next(learner.margin_geometry.parameters()).detach().clone())
        return original(*args,**kwargs)
    learner.geometry.query=query
    learner.update(batch())
    assert len(seen)>=12
    assert all(torch.equal(p,before[0]) for p in seen)
    for old,online,geometry in zip(before,learner.margin_online.parameters(),learner.margin_geometry.parameters()):
        torch.testing.assert_close(geometry,old*.995+online.detach()*.005)
        assert not geometry.requires_grad and geometry.grad is None
    assert learner.metrics()['geometry_beta']==1.


def test_ema_optimizer_counter_and_rng_save_load_preserves_next_updates():
    torch.set_num_threads(1)
    torch.manual_seed(22);np.random.seed(22);random.seed(22)
    learner=StableArmLearner(device='cpu',width=16,critic_only_updates=2)
    learner.set_transitions(20000)
    data=batch()
    for _ in range(4): learner.update(data)
    learner.record_episode(1.)
    payload=dict(learner=learner.state_dict(),rng=dict(torch=torch.get_rng_state(),
        numpy=np.random.get_state(),python=random.getstate()))
    stream=io.BytesIO();torch.save(payload,stream)
    expected_random=(np.random.rand(),random.random())
    learner.update(data);learner.update(data)
    expected=learner.state_dict()
    stream.seek(0);saved=torch.load(stream,weights_only=False)
    restored=StableArmLearner(device='cpu',width=16,critic_only_updates=2)
    restored.load_state_dict(saved['learner'])
    torch.set_rng_state(saved['rng']['torch']);np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python'])
    assert (np.random.rand(),random.random())==expected_random
    restored.update(data);restored.update(data)
    actual=restored.state_dict()
    for module in ('actor','target_actor','q','qt','cost_q','cost_qt','margin'):
        for key,value in expected[module].items():
            torch.testing.assert_close(actual[module][key],value,rtol=0.,atol=0.)
    for key in expected['ea_stability_v2']['margin_geometry']:
        torch.testing.assert_close(actual['ea_stability_v2']['margin_geometry'][key],
                                   expected['ea_stability_v2']['margin_geometry'][key],rtol=0.,atol=0.)
    assert restored.updates==learner.updates and restored.actor_updates==learner.actor_updates
    assert restored.transitions==20000 and restored.geometry.transitions==20000
    assert restored.episode_cost_ema==learner.episode_cost_ema


def test_undeclared_settings_change_is_not_a_native_resume():
    learner=StableArmLearner(device='cpu',width=16)
    other=StableArmLearner(device='cpu',width=16,margin_ema_tau=.001)
    try:
        other.load_state_dict(learner.state_dict())
    except ValueError as error:
        assert 'explicit fork' in str(error)
    else:
        raise AssertionError('A changed candidate must not masquerade as a native resume')
