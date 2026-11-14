import gzip
import json

from copy import deepcopy
import pytest

from scripts.paper_crax_payload.analyze_component_trajectories import (
    compare, resample, compact_summary, compare_summaries)


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


def test_remote_prefix_summaries_preserve_first_n_matching_exactly():
    results=[]
    for i,arm in enumerate(('iso','noise-only','metric-only','full')):
        episodes=[dict(cells={(0,0,0),(1,0,0)},path_m=1.),
                  dict(cells={(1,0,0),(2,0,0)},path_m=3.)]
        # Full has only its first success: the other arms must not use second
        # successes or extra coverage when matching to this common N=1.
        selected=episodes[:1] if arm=='full' else episodes
        result=dict(arm=arm,source_commit='fixture',configuration_digest='fixture-config',
            steps=4,successes=len(selected),violations=1,timeouts=0,
            curve=[dict(steps=1,successes=1,violations=0,timeouts=0),
                   dict(steps=4,successes=len(selected),violations=1,timeouts=0)],
            windows=[dict(start=0,end=4,all_cells={(0,0,0),(1,0,0),(2,0,0)},
                successful=selected)])
        results.append(json.loads(json.dumps(compact_summary(result))))
    value=compare_summaries(results)
    assert all(row['matched_episodes']==1 and row['unique_world_cells']==2 and row['mean_path_m']==1
               for row in value['matched_success_coverage'][0]['arms'])
    assert value['equal_violation_budget'][0]['maximum_training_violations']==0
    assert value['equal_violation_budget'][0]['arms'][0]['steps']==1
    changed=deepcopy(results);changed[0]['configuration_digest']='another-protocol'
    with pytest.raises(ValueError):compare_summaries(changed)


def test_zero_success_window_does_not_invent_coverage_or_mean_path():
    results=[]
    for arm in ('iso','noise-only','metric-only','full'):
        results.append(compact_summary(dict(arm=arm,configuration_digest='fixture',source_commit='fixture',
            steps=4,successes=0,violations=0,timeouts=1,curve=[],
            windows=[dict(start=0,end=4,all_cells={(0,0,0)},successful=[])])))
    value=compare_summaries(results)
    assert all(row['matched_episodes']==0 and row['unique_world_cells']==0 and row['mean_path_m'] is None
               for row in value['matched_success_coverage'][0]['arms'])


def test_exploration_report_renders_all_arms_and_zero_success_windows(tmp_path):
    from scripts.paper_crax_payload.report_component_exploration import report
    summaries=[]
    for arm in ('iso','noise-only','metric-only','full','full-rho010','full-rho050'):
        summaries.append(compact_summary(dict(arm=arm,configuration_digest='fixture',source_commit='fixture',
            steps=300000,successes=0,violations=0,timeouts=3000,censored_episodes=2,
            explicit_trace_end=True,completed_receipt=True,
            curve=[dict(steps=300000,successes=0,violations=0,timeouts=3000)],
            windows=[dict(start=lo,end=lo+100000,all_cells={(0,0,0)},successful=[],
                first_success_examples=[]) for lo in (0,100000,200000)])))
    output=tmp_path/'fixture_report'
    receipt=report(compare_summaries(summaries),output)
    assert receipt['new_training_transitions']==receipt['new_simulation_controls']==0
    assert (output/'task_discovery_and_risk.png').stat().st_size>0
    assert (output/'first_chronological_training_trajectories.pdf').stat().st_size>0
    assert (output/'training_totals.csv').read_text().count('\n')==7
    assert (output/'matched_success_coverage.csv').read_text().count('\n')==13
