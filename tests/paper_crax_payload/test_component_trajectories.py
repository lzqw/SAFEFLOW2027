import gzip
import json

from scripts.paper_crax_payload.analyze_component_trajectories import compare, resample


def test_complete_trace_counts_failures_censoring_and_common_success_samples(tmp_path):
    paths=[]
    for arm in ('iso','noise-only','metric-only','full'):
        path=tmp_path/arm;path.mkdir();paths.append(path)
        config=dict(arm=dict(name=arm),source_commit='fixture',configuration=dict(trajectory_analysis=dict(
            windows=[[0,4]],world_grid_m=.02,arclength_spacing_m=.005)))
        (path/'run_config.json').write_text(json.dumps(config))
        initial=lambda env,ep,step:dict(record_type='initial',env=env,episode_id=ep,transitions=step,
            tcp_xyz=[0,-.02,0],obstacle=[0,0,0])
        step=lambda env,ep,n,done,success,violation:dict(record_type='step',env=env,episode_id=ep,
            transitions=n,tcp_xyz=[.02,.02,0],done=done,safe_success=success,violation=violation)
        records=[initial(0,0,0),initial(1,0,0),step(0,0,1,True,True,False),
                 initial(0,1,1),step(1,0,2,True,False,True),initial(1,1,2),
                 step(0,1,3,False,False,False),step(1,1,4,True,False,False),
                 initial(1,2,4),dict(record_type='end')]
        with gzip.open(path/'trajectories.jsonl.gz','wt') as f:
            for row in records:f.write(json.dumps(row)+'\n')
        (path/'receipt.json').write_text(json.dumps(dict(steps=4,safe_successes=1,training_violations=1)))
    result=compare(paths)
    assert all(r['successes']==r['violations']==r['timeouts']==r['censored_episodes']==1 for r in result['results'])
    assert result['matched_success_coverage'][0]['arms'][0]['matched_episodes']==1
    assert result['equal_violation_budget'][0]['arms'][0]['violations']==0
    assert result['equal_violation_budget'][0]['arms'][0]['successes']==1
    assert not result['legal_route_or_homotopy_proof']
    assert len(resample([[0,0,0],[0,0,0]],.005))==1
