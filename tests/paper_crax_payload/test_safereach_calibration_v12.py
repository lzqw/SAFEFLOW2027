import numpy as np
import pytest

from scripts.paper_crax_payload.diagnose_safereach_calibration_v12 import replay_window, statistics


def test_circular_window_is_chronological_and_keeps_unsafe_labels():
    replay = dict(capacity=6,count=6,cursor=2,data=dict(obs=np.zeros((6,25)),
        action=np.zeros((6,3)),margin=np.array([0.,.001,-.003,-.002,-.001,0.]),
        cost=np.array([0,1,0,0,0,0])))
    selected = replay_window(replay,size=4)
    assert selected['margin'].tolist()==[-.001,0.,0.,.001]
    assert selected['cost'].tolist()==[0,0,0,1]
    replay['data']['cost'][1] = 0
    with pytest.raises(ValueError,match='labels'):
        replay_window(replay,size=4)


def test_false_safe_counts_use_both_head_and_mean_without_probability_claim():
    actual = np.array([.001,.002,-.003])
    heads = np.array([[-.001,.003,-.002],[-.002,-.001,-.004]])
    result = statistics(heads,actual,np.ones(3,dtype=bool))
    assert result['actual_unsafe']==2
    assert result['false_safe_mean_count']==1
    assert result['unsafe_both_heads_safe_count']==1
    assert statistics(heads,actual,np.zeros(3,dtype=bool))==dict(count=0)
