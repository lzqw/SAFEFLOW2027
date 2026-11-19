from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.paper_crax_payload.native_p5_adaptation import build_native
from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import TEMPLATE,RADIUS_PLAN,bind
from scripts.paper_crax_payload.train_piper_comparison_v12 import attach as original_attach
from scripts.paper_crax_payload.train_reach_crossseed_v12 import attach,evaluate


def plan():
    template=json.loads(TEMPLATE.read_text());radius=json.loads(RADIUS_PLAN.read_text())
    rows=[dict(seed=r['seed'],parent_steps=300000,total_steps=400000,new_training_steps=100000,
        saved_cuda_rng_index=0,downstream_training_started=False,parent_checkpoint_sha256=r['checkpoint_sha256'],
        checkpoint_sha256=str(i)*64) for i,r in enumerate(radius['parents'],1)]
    summary=dict(status='complete_independent_radius200K_summary',completed_runs=2,
        new_completed_training_transitions=200000,training_seeds=[41028,41029],development_episodes=180,
        independent_test_used=False,downstream_training_started=False,
        provenance=[dict(seed=r['seed'],checkpoint_sha256=r['checkpoint_sha256'],source_commit='10ba914c') for r in rows])
    return bind(template,summary,rows,41028,Path('/data/full_400000.pt'))


@pytest.mark.parametrize('arm',['P5-fixed','P5-activation2mm','Metric-only'])
def test_reuses_exact_existing_weights_optimizers_and_loss_math(arm):
    torch.set_num_threads(1);p=plan();line=p['lines']['piper_reach']
    native=json.loads(Path('configs/paper_crax_payload/piper_sim_v1.json').read_text())
    torch.manual_seed(41028);old=build_native(native,line,'cpu');old.set_transitions(400000)
    original_attach(old,p,'piper_reach',arm)
    old_rng=torch.random.get_rng_state()
    torch.manual_seed(41028);new=build_native(native,line,'cpu');new.set_transitions(400000)
    opt=new.actor_opt;attach(new,p,'piper_reach',arm)
    assert torch.equal(old_rng,torch.random.get_rng_state()) and new.actor_opt is opt
    obs=torch.randn(8,34)
    torch.manual_seed(700);a=old.actor(obs,geometry=old.geometry,return_reference=True)
    torch.manual_seed(700);b=new.actor(obs,geometry=new.geometry,return_reference=True)
    assert all(torch.equal(l,r) for l,r in zip(a[:3],b[:3]))
    left,right=old.actor_regularizer(a[3]),new.actor_regularizer(b[3])
    if isinstance(left,float):assert left==right==0.
    else:torch.testing.assert_close(left,right,rtol=0,atol=0)
    for step in (500000,550000):
        old.set_transitions(step);new.set_transitions(step)
        assert old.actor_opt.param_groups[0]['lr']==new.actor_opt.param_groups[0]['lr']==.00003
    broken=deepcopy(new.state_dict());broken.pop('reach_crossseed_v12')
    with pytest.raises(ValueError,match='Retain'):
        new.load_state_dict(broken)


def test_all_arms_receive_both_modes_and_all_extra_physics_is_counted(monkeypatch):
    import scripts.paper_crax_payload.train_reach_crossseed_v12 as module
    learner=SimpleNamespace(transitions=500000,adaptation=dict(parent_steps=400000),updates=1,
        actor_updates=1,safety_multiplier=0.,actor=torch.nn.Linear(1,1),target_actor=torch.nn.Linear(1,1),
        crossseed_identity=dict(seed=41028))
    learner.action=lambda obs,deterministic=False:deterministic
    calls=[]
    def fake_evaluate(model,config,assets,task,episodes,seed_start):
        calls.append(model.action(None))
        return dict(conditions=[dict(condition='nominal',records=[dict(length=3)])],
            new_training_transitions=0,physical_robot_interactions=0)
    monkeypatch.setattr(module,'evaluate_native',fake_evaluate)
    result=evaluate(learner,{},None,'Reach',100,12020000)
    assert calls==[False,True]
    assert result['new_simulation_controls']==6
    assert [g['deployment_mode'] for g in result['conditions']]==['sampled','zero_noise']
    assert result['independent_test_used'] is False and learner.action(None) is False
