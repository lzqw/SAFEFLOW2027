import numpy as np
import pytest

from scripts.paper_crax_payload.diagnose_safereach_normals_v12 import alignment,select_pool


def test_keeps_every_unsafe_transition_and_fixed_unique_safe_subset():
    pools={}
    for label,bad in [('P5',13),('Iso',11),('Metric',14)]:
        margin=np.r_[np.full(bad,.001),np.full(100,-.002),np.full(200,-.02)]
        pools[label]=dict(margin=margin,obs=np.zeros((len(margin),25)),action=np.zeros((len(margin),3)))
    selected=select_pool(pools)
    assert selected==select_pool(pools) and len(selected)==128
    assert sum(r['original_margin_m']>0 for r in selected)==38
    for label,bad in [('P5',13),('Iso',11),('Metric',14)]:
        assert {r['original_transition'] for r in selected if r['behavior_pool']==label and r['original_margin_m']>0}==set(range(240001,240001+bad))
    pools['P5']['margin'][0]=-.002
    with pytest.raises(ValueError,match='all38'):
        select_pool(pools)


def test_flat_truth_is_not_silently_counted_as_correct_or_opposite():
    assert alignment([1,0,0],[0,0,0])['normal_signed_cosine'] is None
    result=alignment([1,0,0],[-2,0,0])
    assert result['normal_signed_cosine']==-1 and result['normal_axis_cosine_squared']==1
    assert alignment([1,0,0],[0,2,0])['normal_signed_cosine']==0
