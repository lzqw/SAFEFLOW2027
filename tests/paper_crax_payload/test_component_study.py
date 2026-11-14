import torch
import pytest

from scripts.paper_crax_payload.component_models import ComponentLearner
from scripts.star.arm_safe_entropy.stability_models import StableArmLearner


class FixedGeometry:
    def query(self, actor, obs, midpoint, proposal, step, diagnostics):
        normal=torch.zeros_like(proposal); normal[:,0]=1
        weight=torch.ones_like(proposal[:,0])
        return normal,weight,{'zero_gradient':torch.zeros_like(weight,dtype=torch.bool)}


def test_switches_separate_sampling_and_energy_and_match_initialization():
    obs=torch.zeros(4,25); noise=torch.ones(4,4,3)*.2
    results={}
    for component in ['iso','noise_only','metric_only','full']:
        torch.manual_seed(2027)
        model=ComponentLearner(component)
        results[component]=model.actor(obs,geometry=FixedGeometry(),noise_increments=noise)
        if component=='iso': reference=model.actor.state_dict()
        else:
            assert all(torch.equal(v,reference[k]) for k,v in model.actor.state_dict().items())
    assert torch.equal(results['iso'][0],results['metric_only'][0])
    assert torch.equal(results['noise_only'][0],results['full'][0])
    assert not torch.equal(results['iso'][0],results['full'][0])
    assert not torch.equal(results['iso'][1],results['metric_only'][1])
    assert not torch.equal(results['noise_only'][1],results['full'][1])


def test_full_retains_legacy_path_and_checkpoint_identity():
    torch.manual_seed(2027); full=ComponentLearner('full')
    torch.manual_seed(2027); legacy=StableArmLearner(device='cpu')
    obs=torch.zeros(3,25); noise=torch.ones(4,3,3)*.2
    a=full.actor(obs,geometry=FixedGeometry(),noise_increments=noise)
    b=legacy.actor(obs,geometry=FixedGeometry(),noise_increments=noise)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    with pytest.raises(ValueError,match='reinterpret'):
        ComponentLearner('noise_only').load_state_dict(full.state_dict())


def test_all_update_paths_use_component_actor_and_keep_common_temperature_rule():
    obs=torch.zeros(8,25)
    batch=dict(obs=obs,next_obs=obs,action=torch.zeros(8,3),reward=torch.ones(8),
               done=torch.zeros(8),cost=torch.zeros(8),margin=torch.zeros(8))
    for component in ['iso','noise_only','metric_only','full']:
        torch.manual_seed(2027)
        model=ComponentLearner(component,critic_only_updates=0,actor_interval=1)
        model.set_transitions(31000)
        seen=[]
        hooks=[actor.register_forward_hook(lambda module,args,result:seen.append(
            (module.sampling_directional,module.energy_directional))) for actor in (model.actor,model.target_actor)]
        try:model.update(batch)
        finally:
            for hook in hooks:hook.remove()
        expected={'iso':(False,False),'noise_only':(True,False),'metric_only':(False,True),'full':(True,True)}[component]
        assert len(seen)==4 and set(seen)=={expected}  # cost + two Bellman groups + actor
        assert model.reward_updates==2 and model.actor_updates==1 and model.energy_target==18.75
        assert all(torch.isfinite(v).all() for v in model.actor.parameters())
