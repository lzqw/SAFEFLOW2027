import pytest
import torch
from scripts.paper_crax_payload.crax_models import CumulativeFlowLearner


def test_native_dual_uses_perstep_units_not_samplingblock_as_episode():
    learner=CumulativeFlowLearner(4,2,device='cpu',width=8,dual_rule='native_per_step',episode_length=1000)
    costs=torch.full((256,),.05)
    learner.update_native_dual(costs)
    assert learner.multiplier==0 # no Actor updates during preheat
    learner.actor_updates=1
    learner.update_native_dual(costs)
    assert learner.multiplier.item()==pytest.approx((.05-.025)*(.01/1000),rel=1e-6)
    before=learner.multiplier.clone()
    learner.record_completed_episodes(torch.tensor([100.]),torch.tensor([True]))
    assert torch.equal(before,learner.multiplier) # complete episode is recorded,not applied twice
    assert learner.episode_cost_ema==100 and learner.dual_episodes==1
    assert learner.settings['cost_budget']==25
