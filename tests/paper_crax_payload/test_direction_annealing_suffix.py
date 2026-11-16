import pytest

from scripts.paper_crax_payload.analyze_direction_annealing import terminal_chronology


def test_inherited_terminal_and_censored_suffix_counts_without_recounting_parent():
    rows=[dict(record_type='fork',transitions=100),
          dict(record_type='initial',env=0,episode_id=7,transitions=100,inherited_partial_episode_length=5),
          dict(record_type='initial',env=1,episode_id=8,transitions=100,inherited_partial_episode_length=1),
          dict(record_type='step',env=0,episode_id=7,transitions=101,done=True,safe_success=True,violation=False),
          dict(record_type='initial',env=0,episode_id=8,transitions=101),
          dict(record_type='step',env=1,episode_id=8,transitions=102,done=False,safe_success=False,violation=False),
          dict(record_type='resume',transitions=102),
          dict(record_type='step',env=0,episode_id=8,transitions=103,done=True,safe_success=False,violation=True),
          dict(record_type='initial',env=0,episode_id=9,transitions=103),
          dict(record_type='step',env=1,episode_id=8,transitions=104,done=False,safe_success=False,violation=False),
          dict(record_type='end',transitions=104,new_transitions=4)]
    result=terminal_chronology(rows,100,104,[102,104])
    assert result['successes'] == result['violations'] == 1 and result['timeouts'] == 0
    assert result['new_training_transitions'] == 4
    assert result['inherited_episode_terminals'][0]['inherited_length'] == 5
    assert len(result['censored_episodes']) == 1 and result['censored_episodes'][0]['suffix_length'] == 2
    assert [w['transitions'] for w in result['windows']] == [2,2]
    with pytest.raises(ValueError,match='entire closed suffix'):
        terminal_chronology(rows[:-1],100,104,[102,104])
    with pytest.raises(ValueError,match='Missing or duplicate'):
        terminal_chronology(rows[:5]+rows[7:],100,104,[102,104])
