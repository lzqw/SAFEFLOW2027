"""Bounded semantic tests, no physical robot or simulator steps."""
from collections import Counter
import json
import math
from pathlib import Path

import pytest

from deployment.piper.bundle import validate_provenance
from deployment.piper.offline import (OfflineController, OfflineLimits,
    PhysicalTransportDisabled, Recorder, connect_physical_robot, replay_offline,
    si_pose_to_sdk_units)
from deployment.piper.protocol import fixed_policy_schedule


def test_si_conversion_and_hardware_transport_disabled():
    assert si_pose_to_sdk_units([.1,-.2,.3],[math.pi,0.,-math.pi/2]) == (
        100000,-200000,300000,180000,0,-90000)
    with pytest.raises(PhysicalTransportDisabled):
        connect_physical_robot()


def test_stale_or_late_action_is_not_executed(tmp_path):
    controller=OfflineController()
    valid=controller.propose([1.,1.,0.],[.3,0.,.3],10.,now_s=10.,inference_latency_s=.003)
    assert valid['accepted_offline_proposal']
    assert math.sqrt(sum(x*x for x in valid['command_delta_m'])) <= .15*.04+1e-12
    assert valid['executed_action'] is None and valid['physical_commands_sent']==0
    stale=controller.propose([0.,0.,0.],[.3,0.,.3],10.04,now_s=10.3)
    late=controller.propose([0.,0.,0.],[.3,0.,.3],10.4,now_s=10.4,inference_latency_s=.05)
    assert not stale['accepted_offline_proposal'] and stale['command_delta_m'] is None
    assert not late['accepted_offline_proposal'] and late['deadline_miss']
    recorder=Recorder(tmp_path/'record.jsonl',{'fixture':True})
    try:
        recorder.append(valid)
        recorder.append(stale)
        recorder.append(late)
        with pytest.raises(ValueError):
            recorder.append(stale,measured_execution={'action':[0.,0.,0.]})
    finally:
        recorder.close()
    assert replay_offline(tmp_path/'record.jsonl')==dict(
        records=3,accepted=1,deadline_misses=1,rejected=2,
        physical_commands_sent=0,real_interactions=0)


def test_panda_or_untrained_state_cannot_be_exported_as_piper():
    for metadata in [{'robot_family':'Panda'}, {'robot_family':'Piper','task':'Reach',
                      'method':'EA','observation_dim':34,'action_dim':3,
                      'training_robot_family':'Panda','training_steps':1000000}]:
        with pytest.raises(ValueError):
            validate_provenance(metadata)


def test_interleaved_schedule_exactly_balanced_and_no_motion():
    rows=fixed_policy_schedule()
    counts=Counter((r['task'],r['condition'],r['method']) for r in rows)
    assert len(rows)==360 and len(counts)==18 and set(counts.values())=={20}
    assert all(r['status']=='planned_not_executed' for r in rows)
    for offset in range(0,len(rows),3):
        assert {r['method'] for r in rows[offset:offset+3]}=={'Iso','Iso-Low','EA'}
    assert fixed_policy_schedule()==rows
