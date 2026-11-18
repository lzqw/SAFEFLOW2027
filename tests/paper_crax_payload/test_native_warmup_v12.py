from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts.paper_crax_payload.native_p5_adaptation import build_native, attach_adaptation
from scripts.paper_crax_payload.train_native_p5_warmup_v12 import attach, TrainingTracker


def configuration():
    plan=json.loads(Path('configs/paper_crax_payload/simulation_v12_piper_round1.json').read_text())
    manifest=json.loads(Path('reports/performance_gaps_v11/first_stage_complete/a100/piper_reach-p5/run_config.json').read_text())
    return plan,manifest['configuration']


def test_native_fixed_baseline_identity_and_new_step_ramp_gradient():
    torch.set_num_threads(1);plan,native=configuration();line=plan['lines']['piper_reach']
    torch.manual_seed(9901);old=build_native(native,line,'cpu');old.set_transitions(400000)
    baseline=deepcopy(plan);baseline['arms']=['p5'];attach_adaptation(old,baseline,'piper_reach','p5')
    torch.manual_seed(9901);new=build_native(native,line,'cpu');new.set_transitions(400000)
    attach(new,plan,'piper_reach','R0-fixed')
    for name in ('actor','target_actor','q','cost_q','margin_geometry'):
        assert all(torch.equal(v,getattr(new,name).state_dict()[k]) for k,v in getattr(old,name).state_dict().items())
    ramp=build_native(native,line,'cpu');ramp.set_transitions(400000)
    attach(ramp,plan,'piper_reach','R1-ramp50k');ramp.set_transitions(425000)
    velocity=torch.ones(2,4,3,requires_grad=True);normals=torch.zeros_like(velocity);normals[...,0]=1
    weights=torch.full((2,4),.5)
    trace=dict(velocities=velocity,normals=normals,weights=weights,
        vn=torch.full_like(weights,.025),vt=torch.full_like(weights,.04))
    a=new.actor_regularizer(trace);b=ramp.actor_regularizer(trace)
    ga=torch.autograd.grad(a,velocity,retain_graph=True)[0];gb=torch.autograd.grad(b,velocity)[0]
    assert torch.allclose(gb,.5*ga)
    assert ramp.metrics()['direction_weight_effective']==.5
    for step in (440000,460000,470000,480000,500000):
        old.set_transitions(step);new.set_transitions(step)
        assert old.actor_opt.param_groups[0]['lr']==new.actor_opt.param_groups[0]['lr']
    saved=deepcopy(ramp.state_dict());restored=build_native(native,line,'cpu');restored.set_transitions(400000)
    attach(restored,plan,'piper_reach','R1-ramp50k');restored.load_state_dict(saved)
    assert restored.transitions==425000 and restored.metrics()['direction_weight_effective']==.5
    wrong=build_native(native,line,'cpu');wrong.set_transitions(400000)
    attach(wrong,plan,'piper_reach','R1-ramp75k')
    with pytest.raises(ValueError,match='arm/ramp'):
        wrong.load_state_dict(saved)


def test_stall_is_episode_scoped_and_does_not_change_terminal_classes():
    tracker=TrainingTracker(2)
    for _ in range(11):tracker.observe(0,1.,False,dict(distance_to_goal=.2))
    tracker.observe(0,-10.,True,dict(distance_to_goal=.2))
    assert tracker.metrics()['training_stall_episodes']==1
    tracker.observe(0,10.,True,dict(distance_to_goal=.02))
    assert tracker.metrics()['training_stall_episodes']==1
    assert tracker.metrics()['ordinary_terminal_task_successes']==1
    restored=TrainingTracker(2);restored.load_state_dict(tracker.state_dict())
    assert restored.state_dict()==tracker.state_dict()
