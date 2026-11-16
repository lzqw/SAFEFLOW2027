import json
from pathlib import Path

import pytest
import torch

from scripts.paper_crax_payload.component_models import ComponentLearner
from scripts.paper_crax_payload.direction_development import DirectionDevelopmentLearner, outward_loss
from scripts.paper_crax_payload.train_direction_development import validate_config


class FixedGeometry:
    def query(self, actor, obs, midpoint, proposal, step, diagnostics):
        normal = torch.zeros_like(proposal); normal[:, 0] = 1.
        weight = torch.ones_like(proposal[:, 0])
        return normal, weight, {'zero_gradient': torch.zeros_like(weight, dtype=torch.bool)}


def test_outward_loss_unweighted_detached_scale_and_one_sided_gradient():
    velocity = torch.tensor([[[2., 1., 0.]], [[-2., 1., 0.]]], requires_grad=True)
    normal = torch.tensor([[[1., 0., 0.]], [[1., 0., 0.]]])
    weights = torch.tensor([[.1], [1.]])
    loss = outward_loss(velocity, normal, weights, 1.)
    assert loss.item() == pytest.approx(.04, rel=1e-5)
    loss.backward()
    assert velocity.grad[0, 0, 0] > 0
    assert torch.equal(velocity.grad[:, :, 1:], torch.zeros(2, 1, 2))
    assert torch.equal(velocity.grad[1], torch.zeros(1, 3))
    assert outward_loss(velocity.detach(), normal, weights*.1, 1.) == pytest.approx(loss.item()*.1)


def test_shared_shrink_covariance_and_no_rng_or_weight_reinitialization():
    torch.manual_seed(2028); old = ComponentLearner('full')
    torch.manual_seed(2028); new = DirectionDevelopmentLearner(component='full', normal_shrink=True, direction_weight=1.)
    assert all(torch.equal(v, new.actor.state_dict()[k]) for k, v in old.actor.state_dict().items())
    obs = torch.zeros(4, 25); noise = torch.ones(4, 4, 3)*.1
    action, energy, clips, trace = new.actor(obs, geometry=FixedGeometry(), noise_increments=noise, return_reference=True)
    torch.testing.assert_close(trace['vn'], torch.full((4, 4), .01))
    torch.testing.assert_close(trace['vt'], torch.full((4, 4), .04))
    v = trace['velocities']
    expected = (.5*.25*(v[..., 0].square()/.01 + v[..., 1:].square().sum(-1)/.04)).sum(-1)
    torch.testing.assert_close(energy, expected)
    (energy.mean()*.01 + new.actor_regularizer(trace) + action.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in new.actor.parameters())
    with pytest.raises(ValueError, match='reinterpret'):
        DirectionDevelopmentLearner(component='full', normal_shrink=True).load_state_dict(new.state_dict())


def test_old_control_update_identical_and_new_update_roundtrip():
    torch.set_num_threads(1)
    obs = torch.zeros(8, 25)
    batch = dict(obs=obs, next_obs=obs, action=torch.zeros(8, 3), reward=torch.ones(8),
                 done=torch.zeros(8), cost=torch.zeros(8), margin=torch.zeros(8))
    results = []
    for cls in (ComponentLearner, DirectionDevelopmentLearner):
        torch.manual_seed(41); learner = cls(component='full', critic_only_updates=0, actor_interval=1)
        learner.set_transitions(31000); learner.update(batch)
        results.append(learner)
    assert all(torch.equal(v, results[1].actor.state_dict()[k]) for k, v in results[0].actor.state_dict().items())
    new = DirectionDevelopmentLearner(component='full', normal_shrink=True, direction_weight=1.,
                                       critic_only_updates=0, actor_interval=1)
    new.set_transitions(31000); new.update(batch)
    restored = DirectionDevelopmentLearner(component='full', normal_shrink=True, direction_weight=1.,
                                            critic_only_updates=0, actor_interval=1)
    restored.load_state_dict(new.state_dict())
    assert restored.metrics() == new.metrics()
    assert new.actor_updates == 1 and new.reward_updates == 2
    config = json.loads(Path('configs/paper_crax_payload/direction_development_v5.json').read_text())
    validate_config(config)
    config['evaluation']['independent_layout_seed_start'] = 999
    with pytest.raises(ValueError, match='development-only'): validate_config(config)


def test_complete_five_arm_comparison_matches_chronological_successes():
    from scripts.paper_crax_payload.analyze_component_trajectories import compact_summary
    from scripts.paper_crax_payload.analyze_direction_development import combine
    config = json.loads(Path('configs/paper_crax_payload/direction_development_v5.json').read_text())
    summaries = []
    for arm in config['arms']:
        count = 1 if arm['name'] == 'normal-shrink-dir' else 2
        summary = compact_summary(dict(arm=arm['name'], configuration_digest='same', source_commit='fixture',
            initial_network_sha256='paired', training_seed=2028, steps=150000, successes=count,
            analysis_script_sha256='same-analysis', shared_analysis_sha256='same-shared-analysis',
            trajectory_analysis=dict(windows=[[0, 150000]], world_grid_m=.02, arclength_spacing_m=.005),
            violations=1, timeouts=0, completed_receipt=True, explicit_trace_end=True,
            independent_confirmation=False,
            curve=[dict(steps=10, successes=1, violations=0, timeouts=0),
                   dict(steps=150000, successes=count, violations=1, timeouts=0)],
            windows=[dict(start=0, end=150000, all_cells={(0, 0, 0)},
                successful=[dict(cells={(i, 0, 0)}, path_m=float(i+1)) for i in range(count)])]))
        summaries.append(summary)
    result = combine(config, summaries)
    rows = result['matched_success_coverage'][0]['arms']
    assert len(rows) == 5 and all(r['matched_episodes'] == r['unique_world_cells'] == 1 for r in rows)
    assert all(r['mean_path_m'] == 1. for r in rows)
    assert all(r['successes'] == 1 and r['violations'] == 0
               for r in result['equal_violation_budget'][0]['arms'])
    assert not result['independent_confirmation'] and not result['method_selected']
    with pytest.raises(ValueError, match='all five'): combine(config, summaries[:-1])


def test_confirmation_cannot_start_from_development_or_reinterpret_selected_method():
    from copy import deepcopy
    from scripts.paper_crax_payload.train_direction_confirmation import (
        configuration_digest, validate_confirmation)
    development = json.loads(Path('configs/paper_crax_payload/direction_development_v5.json').read_text())
    with pytest.raises(ValueError, match='confirmation freeze'): validate_confirmation(development)
    # Synthetic registration fixture only: no real receipts or test layouts are
    # manufactured, no file is written, no simulator or training is executed.
    config = dict(protocol='direction_confirmation_v5_frozen',
        status='frozen_before_confirmation_training_and_independent_test',
        task_version=development['task_version'], num_envs=2, training_seeds=[2031, 2032, 2033],
        steps_per_run=300000, new_training_budget=2700000,
        arms=[a for a in development['arms'] if a['name'] in ('iso', 'metric-only', 'normal-shrink-dir')],
        shared_learner=deepcopy(development['shared_learner']), geometry=development['geometry'],
        evaluation=deepcopy(development['evaluation']), scientific_early_stopping=False,
        automatic_failed_run_retry=False, physical_robot_interactions=0,
        development_selection=dict(completed_runs=5, actual_training_steps=750000,
            selected_arm='normal-shrink-dir', review_rationale='synthetic fixture, not a scientific decision',
            full_learning_curve_review=True, independent_test_used_for_selection=False,
            run_receipt_sha256={a['name']: '0'*64 for a in development['arms']},
            trajectory_summary_sha256={a['name']: '0'*64 for a in development['arms']}),
        independent_closure=dict(status='not_opened_requires_all_nine_final_policy_hashes',
            checkpoint_rule='fixed300K_all_three_seeds_no_reselection',
            episodes_per_condition_per_policy=100, fresh_layout_seed_commitment_sha256='0'*64))
    config['shared_learner']['replay_capacity'] = 300000
    config['evaluation']['milestones'] = [50000, 100000, 150000, 200000, 250000, 300000]
    config['freeze_sha256'] = configuration_digest(config)
    validate_confirmation(config)
    config['arms'][-1]['direction_weight'] = .5
    config['freeze_sha256'] = configuration_digest(config)
    with pytest.raises(ValueError, match='retune'): validate_confirmation(config)
