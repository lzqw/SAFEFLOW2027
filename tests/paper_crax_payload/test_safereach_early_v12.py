from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts.paper_crax_payload.train_safereach_early_v12 import build, validate, SafeReachTracker


def test_same_seed_all_arms_retain_initial_networks_and_declared_warmup():
    torch.set_num_threads(1)
    cfg=json.loads(Path('configs/paper_crax_payload/simulation_v12_safereach_early.json').read_text())
    validate(cfg); reference=None
    for arm in cfg['arms']:
        torch.manual_seed(2033);learner=build(cfg,arm,2033,'cpu')
        current={name:deepcopy(getattr(learner,name).state_dict()) for name in
                 ('actor','target_actor','q','cost_q','margin_geometry')}
        if reference is None:reference=current
        else:
            assert all(torch.equal(t,current[name][k]) for name,values in reference.items() for k,t in values.items())
        if arm['name']=='E1-critic20k':assert learner.settings['critic_only_updates']==18000
        elif arm['name'].startswith('E2'):
            learner.set_transitions(0);assert learner.current_direction_weight==.25
            learner.set_transitions(arm['ramp_steps']);assert learner.current_direction_weight==1.
        saved=deepcopy(learner.state_dict()); restored=build(cfg,arm,2033,'cpu')
        restored.load_state_dict(saved)
    bad=deepcopy(cfg);bad['steps_per_run']=75000
    with pytest.raises(ValueError,match='registered'):
        validate(bad)


def test_safereach_goal_threshold_stays35mm_not_piper30mm():
    tracker=SafeReachTracker(2);tracker.observe(0,10.,True,dict(distance_to_goal=.033))
    assert tracker.metrics()['ordinary_terminal_task_successes']==1
