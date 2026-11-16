"""CPU batch1 latency of the exact selected native PointGoal SAC policy.

Uses the same genuine observation timing fixture as the Flow measurement.
No simulator, training state, second policy, or robot transport is instantiated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import time

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--observation-pool',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve prior timing; no repeat')
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
    import jax
    from jax import numpy as jp
    from crax.training.acme import running_statistics
    from crax.training.agents.sac_lag import networks
    from scripts.paper_crax_payload.validate_native_sac_common import unreplicate_saved_policy
    if any(d.platform!='cpu' for d in jax.local_devices()):
        raise ValueError('This isolated timing must not take a training GPU')
    selection=json.loads(args.selection.read_text())
    if selection.get('independent_test_used') is not False:
        raise ValueError('Use the untouched common-validation freeze')
    candidate=selection['methods']['sac_lag']['selected']
    run=args.run_root/candidate['run']
    path=run/f"policy_normalizer_{candidate['checkpoint_steps']}.pkl"
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=candidate['policy_sha256']:
        raise ValueError('Not the exact selected native SAC model')
    settings=json.loads((run/'configuration.json').read_text())['settings']
    # Only the trusted internally exported selected policy is unpickled.
    with path.open('rb') as stream: saved=pickle.load(stream)
    if saved['steps']!=candidate['checkpoint_steps'] or saved.get('normalizer_saved') is not True:
        raise ValueError('Wrong native policy/normalizer identity')
    params,replica=unreplicate_saved_policy(saved['params'],jax)
    pool=json.loads(args.observation_pool.read_text())
    observations=np.asarray([r['observation'] for r in pool['records']],dtype=np.float32)
    if observations.ndim!=2 or observations.shape[1]!=settings['obs_dim'] or not np.isfinite(observations).all():
        raise ValueError('Require the genuine matched-dimension observation fixture')
    normalize=running_statistics.normalize if settings['normalize_observations'] else lambda obs,params:obs
    network=networks.make_sac_lag_networks(observation_size=settings['obs_dim'],
        action_size=settings['action_dim'],preprocess_observations_fn=normalize)
    make_policy=networks.make_inference_fn(network)
    @jax.jit
    def action(parameters,observation,key):
        return make_policy(parameters,deterministic=False)(observation,key)[0]
    params=jax.device_put(params)
    key=jax.random.PRNGKey(695000)
    started=time.perf_counter()
    compiled=action.lower(params,jp.asarray(observations[:1]),key).compile()
    compile_seconds=time.perf_counter()-started
    timings=[]
    for index in range(220):
        start=time.perf_counter()
        key,noise=jax.random.split(key)
        result=np.asarray(compiled(params,jp.asarray(observations[index%len(observations)][None]),noise))
        if result.shape!=(1,settings['action_dim']) or not np.isfinite(result).all():
            raise ValueError('Nonfinite native action')
        if index>=20: timings.append(time.perf_counter()-start)
    output=dict(method='sac_lag',selected_run=candidate['run'],selected_steps=candidate['checkpoint_steps'],
        policy_sha256=digest,selection_freeze_digest=selection['freeze_digest'],
        host=platform.node(),device='cpu',jax_version=jax.__version__,batch_size=1,
        thread_configuration='OMP1; XLA CPU eigen multithreading disabled; runtime worker count not asserted',
        warmup=20,measured_actions=200,p50_s=float(np.quantile(timings,.5)),
        p95_s=float(np.quantile(timings,.95)),mean_s=float(np.mean(timings)),compile_seconds=compile_seconds,
        compilation_excluded_from_action_timing=True,stored_policy_replica_axis_removed=replica,
        includes='observation conversion/normalization, native stochastic Actor, RNG split and host return; finite output check',
        excludes='model load, JIT compile, observation acquisition, physical transport/actuation',
        fixture_path=str(args.observation_pool),fixture_sha256=hashlib.sha256(args.observation_pool.read_bytes()).hexdigest(),
        fixture_role='same existing recorded4M observations as Flow CPU timing; not a mechanism analysis',
        base_flow_integration_steps=0,margin_forwards=0,input_VJPs=0,
        native_policy_network_forwards=1,training_transitions=0,simulator_controls=0,physical_robot_interactions=0,
        measurement_host_is_not_deployment_machine=True,
        cross_framework_timing_not_exact_runtime_equivalence=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
