import numpy as np
import pytest

from types import SimpleNamespace

from scripts.paper_crax_payload.validate_native_sac_common import (
    require_validation_seeds,summarize,unreplicate_saved_policy)


def test_independent_test_seed_family_cannot_enter_development_validator():
    require_validation_seeds([61027])
    require_validation_seeds([61027,61028,61029,61030,61031])
    for seeds in ([71027],[61027,71027],[61027,61027],[]):
        with pytest.raises(ValueError):require_validation_seeds(seeds)


def test_native_goals_cost_budget_and_failure_capped_times_are_preserved():
    raw=np.array([[3.,20.,3.,1000.,100.],[0.,30.,0.,1000.,1000.]])
    result=summarize(raw,25,.08)
    assert result['native_return']==1.5
    assert result['episode_cost']==25
    assert result['goals_per_episode']==1.5
    assert result['at_least_one_goal']==.5
    assert result['budget_compliant_fraction']==.5
    assert result['budget_compliant_return']==3
    assert result['capped_first_goal_seconds']==44
    assert result['conditional_first_goal_seconds']==8
    assert result['records'][1]['first_goal_seconds'] is None


def test_replicated_trainingstate_policy_export_is_unpmapped_once():
    def leaves(value):
        if isinstance(value,dict):return [x for v in value.values() for x in leaves(v)]
        if isinstance(value,tuple):return [x for v in value for x in leaves(v)]
        return [value]
    def tree_map(function,value):
        if isinstance(value,dict):return {k:tree_map(function,v) for k,v in value.items()}
        if isinstance(value,tuple):return tuple(tree_map(function,v) for v in value)
        return function(value)
    backend=SimpleNamespace(tree_util=SimpleNamespace(tree_leaves=leaves,tree_map=tree_map))
    replicated=({'mean':np.zeros((1,62)),'count':np.ones(1)},
                {'kernel':np.zeros((1,62,256)),'bias':np.zeros((1,256))})
    params,axis=unreplicate_saved_policy(replicated,backend)
    assert axis==0 and params[1]['kernel'].shape==(62,256) and params[0]['count'].shape==()
    again,axis=unreplicate_saved_policy(params,backend)
    assert axis is None and again is params
    invalid=({'mean':np.zeros((2,62)),'count':np.ones(2)},
             {'kernel':np.zeros((2,62,256)),'bias':np.zeros((2,256))})
    with pytest.raises(ValueError):unreplicate_saved_policy(invalid,backend)
