from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts.paper_crax_payload.direction_optimization import build, cosine_value, validate_config
from scripts.paper_crax_payload.direction_annealing import AnnealingLearner, schedule_values


def config():
    return json.loads(Path('configs/paper_crax_payload/direction_optimization_v7.json').read_text())


def test_v7_schedules_and_actual_first_round_budget():
    cfg=config(); validate_config(cfg); p1,p2,p3=cfg['candidates']
    assert sum(250000-cfg['parents'][p['parent']]['steps'] for p in cfg['candidates'])==350000
    assert cosine_value(150000,p1['direction'],1.)==.3
    assert cosine_value(175000,p2['direction'],1.)==1.
    assert cosine_value(200000,p2['direction'],1.)==pytest.approx(.55)
    assert cosine_value(225000,p2['direction'],1.)==.1
    assert cosine_value(150000,p3['actor_lr'],.0001)==.0001
    assert cosine_value(175000,p3['actor_lr'],.0001)==pytest.approx(.000065)
    assert cosine_value(200000,p3['actor_lr'],.0001)==.00003
    for p,schedule in [(p2,'fixed'),(p3,'loss_only')]:
        for step in range(100000,150001):
            assert cosine_value(step,p['direction'],1.)==schedule_values(schedule,step)[0]
            assert cosine_value(step,p['actor_lr'],.0001)==.0001


def test_v7_full_learner_import_and_lr_resume_keep_original_state():
    torch.set_num_threads(1)
    cfg=config(); candidate=cfg['candidates'][2]
    parent=AnnealingLearner('loss_only',device='cpu'); parent.set_transitions(150000)
    # Nonempty native optimizer state, not a weights-only checkpoint.
    for parameter in parent.actor.parameters(): parameter.grad=torch.ones_like(parameter)*.001
    parent.actor_opt.step(); parent.actor_opt.zero_grad(set_to_none=True)
    state=deepcopy(parent.state_dict()); child=build(cfg,candidate,'cpu'); child.import_parent(state)
    for name in ('actor','target_actor','q','qt','cost_q','cost_qt','margin_geometry'):
        assert all(torch.equal(value,getattr(child,name).state_dict()[key])
                   for key,value in getattr(parent,name).state_dict().items())
    assert child.actor_opt.state and child.current_direction_weight==.1
    assert child.actor_opt.param_groups[0]['lr']==.0001
    child.set_transitions(200000)
    assert child.actor_opt.param_groups[0]['lr']==.00003
    assert child.q_opt.param_groups[0]['lr']==.0003
    saved=deepcopy(child.state_dict()); restored=build(cfg,candidate,'cpu'); restored.load_state_dict(saved)
    assert restored.transitions==200000 and restored.actor_opt.param_groups[0]['lr']==.00003
    assert restored.actor.rho==restored.target_actor.rho==.25
    assert parent.actor_opt.param_groups[0]['lr']==.0001
    with pytest.raises(ValueError,match='identical v7'):
        build(cfg,cfg['candidates'][0],'cpu').load_state_dict(saved)
    with pytest.raises(ValueError,match='same v6'):
        AnnealingLearner('loss_only',device='cpu').load_state_dict(saved)
