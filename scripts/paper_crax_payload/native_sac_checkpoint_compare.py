"""Read-only comparison of the one authorized native SAC resume experiment.

No environment creation, learner update, or simulator step.  Whole-pickle bytes
are NOT expected to match: timing and artifact paths differ by construction.
All typed algorithm state is compared, not just the policy parameters.
"""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np


def compare_tree(jax,a,b):
    lhs,left_tree=jax.tree_util.tree_flatten_with_path(a)
    rhs,right_tree=jax.tree_util.tree_flatten_with_path(b)
    result=dict(tree_structure_equal=left_tree==right_tree,leaves_left=len(lhs),leaves_right=len(rhs),
                all_leaves_bitwise_equal=True,all_leaves_close=True,maximum_absolute_error=0.,mismatches=[])
    if left_tree!=right_tree or len(lhs)!=len(rhs):
        result.update(all_leaves_bitwise_equal=False,all_leaves_close=False)
        return result
    for (path,x),(_,y) in zip(lhs,rhs):
        x,y=np.asarray(x),np.asarray(y)
        same_meta=x.shape==y.shape and x.dtype==y.dtype
        if not same_meta:
            equal=close=False;error=None
        elif x.dtype.kind in 'fc':
            equal=x.tobytes()==y.tobytes()
            close=np.allclose(x,y,rtol=1e-6,atol=1e-7,equal_nan=True)
            finite=np.isfinite(x)&np.isfinite(y)
            error=float(np.abs(x[finite].astype(np.float64)-y[finite].astype(np.float64)).max()) if finite.any() else 0.
        else:
            equal=x.tobytes()==y.tobytes()
            close=np.array_equal(x,y);error=0. if equal else None
        result['all_leaves_bitwise_equal'] &= bool(equal)
        result['all_leaves_close'] &= bool(close)
        if error is not None:result['maximum_absolute_error']=max(result['maximum_absolute_error'],error)
        if not equal and len(result['mismatches'])<20:
            result['mismatches'].append(dict(path=jax.tree_util.keystr(path),shape_left=list(x.shape),
                shape_right=list(y.shape),dtype_left=str(x.dtype),dtype_right=str(y.dtype),
                absolute_error=error,within_declared_tolerance=bool(close)))
    return result


def host_rng_equal(a,b):
    return bool(a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Do not overwrite an earlier comparison')
    import jax
    snapshots={}
    receipts={}
    for name in ('uninterrupted','prefix','resumed'):
        with (args.root/name/'full_latest.pkl').open('rb') as stream:snapshots[name]=pickle.load(stream)
        receipts[name]=json.loads((args.root/name/'receipt.json').read_text())
    full,prefix,resumed=(snapshots[name] for name in ('uninterrupted','prefix','resumed'))
    keys=('training_state','env_state','buffer_state','local_key','evaluator_key')
    comparisons={key:compare_tree(jax,full[key],resumed[key]) for key in keys}
    candidate_schedule=lambda p:sorted((c['completed_epoch'],c['steps']) for c in p['candidate_checkpoints']
                                      if c['completed_epoch'] in p['epoch_plan']['candidate_epochs'])
    counters_equal=all(full[k]==resumed[k] for k in ('current_step','completed_epochs','evaluation_calls','epoch_plan','settings'))
    unused_host=dict(python_full_vs_resumed_equal=full['python_rng']==resumed['python_rng'],
        numpy_full_vs_resumed_equal=host_rng_equal(full['numpy_rng'],resumed['numpy_rng']),
        python_resumed_vs_own_prefix_equal=prefix['python_rng']==resumed['python_rng'],
        numpy_resumed_vs_own_prefix_equal=host_rng_equal(prefix['numpy_rng'],resumed['numpy_rng']),
        explanation='dc197f3 does not initially seed unused Python/NumPy global RNGs;full-vs-fresh-prefix host states need not equal. Official algorithm uses explicit JAX keys. Exact restoration is checked against the actual prefix state;no whole-pickle/all-host-state bitwise claim.')
    result=dict(status='completed_read_only_comparison',source_commit='dc197f333dab788c915075f05a4f6c2f81ce0ec5',
        comparisons=comparisons,algorithm_counters_and_plan_equal=counters_equal,
        candidate_steps_equal=candidate_schedule(full)==candidate_schedule(resumed),
        candidate_steps=candidate_schedule(full),unused_host_rng=unused_host,
        actual_total_new_training_transitions=sum(r['new_training_transitions'] for r in receipts.values()),
        native_evaluation_controls=sum(r['native_evaluation_controls_this_invocation'] for r in receipts.values()),
        common_validation_controls=sum(r['common_validation_controls'] for r in receipts.values()),
        comparison_new_training_transitions=0,comparison_simulator_queries=0,comparison_real_interactions=0,
        segment_receipts={name:{k:r[k] for k in ('status','steps','initial_steps','new_training_transitions','elapsed_seconds')}
                          for name,r in receipts.items()})
    result['all_typed_algorithm_state_bitwise_equal']=all(r['all_leaves_bitwise_equal'] for r in comparisons.values())
    result['all_typed_algorithm_state_within_declared_tolerance']=all(r['all_leaves_close'] for r in comparisons.values())
    result['learner_replay_rng_bitwise_equal']=all(comparisons[k]['all_leaves_bitwise_equal']
        for k in ('training_state','buffer_state','local_key','evaluator_key'))
    documented_auxiliary={'.pipeline_state.subtree_com','.pipeline_state._impl.subtree_angmom'}
    result['only_documented_auxiliary_float_cache_differs']=bool(comparisons['env_state']['all_leaves_close']
        and all(r['path'] in documented_auxiliary for r in comparisons['env_state']['mismatches']))
    result['canonical_resume_contract_passed']=bool(result['learner_replay_rng_bitwise_equal']
        and result['only_documented_auxiliary_float_cache_differs'] and counters_equal
        and result['candidate_steps_equal'] and unused_host['python_resumed_vs_own_prefix_equal']
        and unused_host['numpy_resumed_vs_own_prefix_equal'])
    result['numerical_resume_contract_passed']=bool(result['all_typed_algorithm_state_within_declared_tolerance']
        and counters_equal and result['candidate_steps_equal'] and unused_host['python_resumed_vs_own_prefix_equal']
        and unused_host['numpy_resumed_vs_own_prefix_equal'])
    result['exact_resume_contract_passed']=bool(result['all_typed_algorithm_state_bitwise_equal'] and counters_equal
        and result['candidate_steps_equal'] and unused_host['python_resumed_vs_own_prefix_equal']
        and unused_host['numpy_resumed_vs_own_prefix_equal'])
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    if not result['exact_resume_contract_passed']:raise SystemExit(1)


if __name__=='__main__':main()
