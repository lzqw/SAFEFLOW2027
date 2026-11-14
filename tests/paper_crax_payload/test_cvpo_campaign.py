"""Small CPU contracts; no RL or native simulator interactions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.paper_crax_payload.cvpo_campaign import (
    collect_candidates,freeze_selection,sha256,validate_registration)
from scripts.paper_crax_payload.run_queue import prerequisites_ready


CONFIG=Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/cvpo_formal_v4.json'


def test_registered_budget_rounding_and_split_are_explicit():
    config=json.loads(CONFIG.read_text())
    assert validate_registration(config)==[2000128,4000000,6000128,8000000,10000128]
    for field,value in [('steps_per_seed',300000),('development_evaluation_seeds',[71027]),
                        ('scientific_early_stop',True)]:
        changed=deepcopy(config);changed[field]=value
        with pytest.raises(ValueError):validate_registration(changed)


def test_campaign_waits_for_real_scientific_predecessors(tmp_path):
    path=tmp_path/'receipt.json';plan={'prerequisite_receipts':[{'path':str(path),'steps':300000}]}
    assert not prerequisites_ready(plan)
    for data in [dict(status='completed',steps=4,engineering_only=True),
                 dict(status='failed',steps=300000),dict(status='completed',steps=300000,engineering_only=True)]:
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):prerequisites_ready(plan)
    path.write_text(json.dumps(dict(status='completed',steps=300000)))
    assert prerequisites_ready(plan)


def test_all_candidates_required_and_selection_never_rewards_excess_cost(tmp_path):
    config=json.loads(CONFIG.read_text());milestones=validate_registration(config)
    for seed in config['training_seeds']:
        directory=tmp_path/f'cvpo-{seed}-10m';directory.mkdir()
        (directory/'run_config.json').write_text(json.dumps(dict(configuration=config,training_seed=seed)))
        (directory/'receipt.json').write_text(json.dumps(dict(status='completed',steps=milestones[-1],
            training_seed=seed,evaluated_milestones=milestones)))
        for step in milestones:
            policy=directory/f'policy_{step}.pt';policy.write_bytes(f'fixture{seed}:{step}'.encode())
            row=dict(episodes=50,records=[{}]*50,mean_episode_cost=20,
                mean_goals_per_episode=step/1e6,mean_native_return=1,
                mean_capped_first_goal_seconds=1,budget_compliant_fraction=1)
            (directory/f'validation_{step}.json').write_text(json.dumps(dict(steps=step,training_seed=seed,
                split='common_validation',independent_test_used=False,policy_sha256=sha256(policy),
                results=[dict(row,seed=s) for s in config['development_evaluation_seeds']])))
    candidates,receipts=collect_candidates(tmp_path,config)
    assert len(candidates)==15 and len(receipts)==3
    candidates[-1].update(mean_episode_cost=30,mean_goals_per_episode=10000)
    freeze=freeze_selection(candidates,'fixture-source')
    assert freeze['selected']['mean_episode_cost']<=25
    assert freeze['selected']['mean_goals_per_episode']<10000
    assert freeze['independent_test_used'] is False
    with pytest.raises(ValueError):freeze_selection(candidates[:-1],'fixture-source')
    Path(candidates[0]['policy_path']).write_bytes(b'changed bytes')
    with pytest.raises(ValueError,match='bytes changed'):collect_candidates(tmp_path,config)


def test_native_evaluation_counts_only_first_episode_and_preserves_rng(monkeypatch):
    from scripts.paper_crax_payload.train_cvpo import evaluate
    monkeypatch.setitem(sys.modules,'tianshou.data',SimpleNamespace(Batch=lambda **kw:SimpleNamespace(**kw)))
    class Policy:
        device=torch.device('cpu');training=True
        def train(self,mode):self.training=mode
        def __call__(self,batch):
            return SimpleNamespace(act=torch.randn(2,2)*10)
    class Env:
        num_envs=2;native=SimpleNamespace(episode_length=3,dt=.1)
        def reset(self,seed):self.index=0;return torch.zeros(2,5)
        def step(self,action):
            assert action.abs().max()<=1
            self.index+=1
            return torch.zeros(2,5),torch.ones(2),torch.tensor([True,self.index==3]),dict(
                cost=torch.tensor([1.,2.]),goals=torch.ones(2))
    policy=Policy();norm=SimpleNamespace(transform=lambda obs:obs)
    before=torch.get_rng_state().clone()
    result=evaluate(policy,norm,Env(),61027)
    assert torch.equal(before,torch.get_rng_state()) and policy.training
    assert result['native_controls']==6 and result['native_policy_updates']==0
    assert [r['length'] for r in result['records']]==[1,3]
    assert [r['episode_cost'] for r in result['records']]==[1,6]
    assert [r['goals'] for r in result['records']]==[1,3]
