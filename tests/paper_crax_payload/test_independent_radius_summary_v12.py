from copy import deepcopy

import pytest

from scripts.paper_crax_payload.summarize_independent_radius_v12 import stage_counts


def fixture():
    parent=dict(status='completed',training_steps=300000,training_seed=41028,engineering=False,
        training_violation_count=558,training_safe_successes=4767,training_safe_timeouts=1482,training_episodes=6807)
    rows=[dict(steps=step,training_violation_count=558+i,training_safe_successes=4767+2*i,
        training_safe_timeouts=1482,episodes=6807+3*i) for i,step in enumerate(range(301000,400001,1000),1)]
    receipt=dict(status='completed',training_steps=400000,initial_steps=300000,newly_collected_transitions=100000,
        training_seed=41028,engineering=False,training_source_commit='10ba914c',training_violation_count=658,
        training_safe_successes=4967,training_safe_timeouts=1482,training_episodes=7107,
        new_training_violations=100,new_training_safe_successes=200)
    return parent,receipt,rows


def test_subtracts_real_parent_counts_not_entire_lineage():
    parent,receipt,rows=fixture();result=stage_counts(parent,receipt,rows,41028)
    assert result['new_training_steps']==100000
    assert result['new_training_violation_count']==100
    assert result['new_training_safe_successes']==200
    assert result['new_training_safe_timeouts']==0
    assert result['cumulative_training_violation_count']==658


@pytest.mark.parametrize('damage',['partial','seed','prefix','terminal','reset','new_count'])
def test_rejects_incomplete_seed_relabeling_or_cumulative_count_inflation(damage):
    parent,receipt,rows=deepcopy(fixture())
    if damage=='partial':rows.pop()
    elif damage=='seed':receipt['training_seed']=41029
    elif damage=='prefix':receipt['initial_steps']=0
    elif damage=='terminal':rows[-1]['episodes']+=1
    elif damage=='reset':rows[-1]['training_violation_count']=0
    else:receipt['new_training_violations']=658
    with pytest.raises(ValueError):
        stage_counts(parent,receipt,rows,41028)
