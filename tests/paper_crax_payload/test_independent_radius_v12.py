from copy import deepcopy
import json

import numpy as np
import pytest

from scripts.paper_crax_payload.train_reach_independent_radius_v12 import PLAN,PROJECT,validate_parent
from scripts.paper_crax_payload.train_reach_fresh_parent_v12 import configuration


def fixture(seed):
    plan=json.loads(PLAN.read_text());parent_plan=json.loads((PROJECT/plan['parent_plan']).read_text())
    config=configuration(parent_plan,seed)
    saved=dict(format='paper_native_piper_sim_v1_full',robot_family='Piper',task='Reach',method='EA',
        steps=300000,training_seed=seed,engineering=False,source_commit='22d4966f',configuration=config,
        geometry_adaptation=None,task_transfer=None,replay=dict(data=dict(obs=np.zeros((1,34)))),
        rng=dict(cuda=[None]),violations=10,safe_successes=20,safe_timeouts=30,episodes=60)
    receipt=dict(status='completed',training_steps=300000,training_seed=seed,initial_steps=0,
        newly_collected_transitions=300000,engineering=False,training_source_commit='22d4966f',
        training_violation_count=10,training_safe_successes=20,training_safe_timeouts=30,training_episodes=60)
    digest=next(r['checkpoint_sha256'] for r in plan['parents'] if r['seed']==seed)
    return plan,saved,receipt,digest


def test_only_declared_radius_family_changes_and_both_seeds_are_retained():
    for seed in (41028,41029):
        plan,saved,receipt,digest=fixture(seed)
        config=validate_parent(saved,receipt,plan,seed,digest)
        assert config.pop('training_radius_values')==[.045,.05,.055]
        assert config==saved['configuration']
    assert plan['total_new_training_steps']==2*plan['new_steps_each']==200000
    assert plan['parent_cuda_rng_index']==0
    assert plan['downstream_training_started'] is False


@pytest.mark.parametrize('damage',['sha','seed','budget','config','rng','counts'])
def test_rejects_wrong_parent_identity_and_best_checkpoint_substitution(damage):
    plan,saved,receipt,digest=fixture(41028)
    if damage=='sha':digest='0'*64
    elif damage=='seed':saved['training_seed']=41029
    elif damage=='budget':saved['steps']=240000
    elif damage=='config':saved['configuration']['shared_learner']['actor_lr']=.001
    elif damage=='rng':saved['rng']['cuda']=[None,None]
    else:receipt['training_violation_count']=9
    with pytest.raises(ValueError):
        validate_parent(saved,receipt,plan,41028,digest)
