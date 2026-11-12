import copy
import torch
import pytest
from scripts.paper_crax_payload.recover_episode_accounting import recover_episode_accounting

def test_full_prefix_recovers_real_episodes_without_policy_or_cost_change():
    s=dict(steps=8,profile=dict(E=2),environment=dict(config=dict(reward_distance_scale=0,reward_goal=1)),
        learner=dict(settings=dict(dual_rule='native_per_step'),episode_cost_ema=torch.tensor(0.),
                     dual_episodes=torch.tensor(0.),multiplier=torch.tensor(.47)),
        replay=dict(size=8,capacity=10,data=dict(reward=torch.tensor([1.,0.,0.,1.,1.,1.,0.,0.]),
            cost=torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.]),terminated=torch.zeros(8),
            truncated=torch.tensor([0.,0.,1.,1.,0.,0.,0.,0.]))),
        partial={k:torch.ones(2)*4 for k in ('return','cost','goals','length')},
        totals=torch.tensor([36.,0.,0.,0.,0.,0.]))
    before=copy.deepcopy(s)
    report,rows=recover_episode_accounting(s)
    assert report['recovered_complete_episodes']==2
    assert s['partial']['length'].tolist()==[2.,2.]
    assert s['partial']['cost'].tolist()==[12.,14.]
    assert s['totals'].tolist()==[36.,2.,2.,10.,2.,2.]
    assert len(rows)==2 and s['learner']['episode_cost_ema']==5
    assert torch.equal(s['learner']['multiplier'],before['learner']['multiplier'])
    for k in s['replay']['data']:assert torch.equal(s['replay']['data'][k],before['replay']['data'][k])
    s['replay']['size']=7
    with pytest.raises(ValueError):recover_episode_accounting(s)
