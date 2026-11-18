"""Batch-one CPU inference of fixed v10 policies on common training replay.

No environment construction, new trajectory, optimizer update or independent
test access. Measures complete target Actor/geometry/action conversion latency,
not episode completion time. Shared-host timings are descriptive, not a gate.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from scripts.paper_crax_payload.direction_confirmation_v10 import FORMAT, build_confirmation
from scripts.paper_crax_payload.direction_annealing import sha256
from scripts.paper_crax_payload.evaluate_direction_polyak import target_deployment, actor_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    freeze = json.loads(args.freeze.read_text())
    if (freeze['protocol'] != 'v10_all_six_fixed250K_target_policy_identity_freeze'
            or len(freeze['policies']) != 6):
        raise ValueError('Use the exact existing six-policy v10 endpoint freeze')
    args.output.mkdir(parents=True, exist_ok=False)
    pool_case = next(c for c in freeze['policies'] if c['arm']['name']=='iso' and c['training_seed']==2031)
    if sha256(pool_case['checkpoint']) != pool_case['checkpoint_sha256']:
        raise ValueError('Common training-replay checkpoint changed')
    saved = torch.load(pool_case['checkpoint'], map_location='cpu', weights_only=False)
    replay = saved['replay']; count = min(replay['count'], replay['capacity'])
    indices = np.random.default_rng(18112026).choice(count, size=16, replace=False)
    observations = replay['data']['obs'][indices].copy()
    del saved, replay
    if observations.shape != (16,25):
        raise ValueError('Use native SafeReach25D training observations')
    registration = dict(host=platform.node(), processor=platform.processor(), torch=str(torch.__version__),
        device='cpu', torch_threads=1, batch_size=1, freeze_sha256=sha256(args.freeze),
        common_pool='Iso2031 final250K native training replay; no independent observations',
        common_pool_checkpoint_sha256=pool_case['checkpoint_sha256'], replay_indices=indices.tolist(),
        observation_bytes_sha256=hashlib.sha256(observations.tobytes()).hexdigest(),
        measured_rounds=5, observations_each_round=16, warmup_calls_each_policy=8,
        timed_scope='learner.action: numpy-to-tensor, full target flow/geometry, tensor-to-numpy',
        new_training_steps=0, new_simulation_controls=0, independent_test_used=False,
        execution_order='random permutation of six methods per round, fixed order seed18112027')
    (args.output/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    learners = []
    for c in freeze['policies']:
        if sha256(c['checkpoint']) != c['checkpoint_sha256']:
            raise ValueError('Frozen endpoint identity changed')
        saved = torch.load(c['checkpoint'],map_location='cpu',weights_only=False)
        if saved['format'] != FORMAT or saved['steps'] != 250000 or saved['seed'] != c['training_seed']:
            raise ValueError('Preserve fixed native model, age and seed')
        learner = build_confirmation(saved['configuration'],c['arm'],c['training_seed'],'cpu')
        learner.load_state_dict(saved['learner']); del saved
        identity = actor_digest(learner.target_actor)
        with target_deployment(learner):
            for i in range(8):
                torch.manual_seed(18113000+i)
                learner.action(observations[i:i+1])
        learners.append((c,learner,identity))
    rows = []
    order_rng = np.random.default_rng(18112027)
    for repeat in range(5):
        for choice in order_rng.permutation(len(learners)):
            c,learner,_ = learners[int(choice)]
            with target_deployment(learner):
                for i,observation in enumerate(observations):
                    torch.manual_seed(18114000+repeat*100+i)
                    start = time.perf_counter_ns()
                    result = learner.action(observation[None])
                    duration = time.perf_counter_ns()-start
                    if result.shape != (1,3) or not np.isfinite(result).all():
                        raise FloatingPointError('Invalid policy action')
                    rows.append(dict(arm=c['arm']['name'],training_seed=c['training_seed'],
                        repeat=repeat,observation_index=i,elapsed_ns=duration))
    summaries=[]
    for c,learner,identity in learners:
        if actor_digest(learner.target_actor) != identity:
            raise ValueError('Benchmark altered frozen target parameters')
        measured = [r['elapsed_ns']/1e6 for r in rows
                    if r['arm']==c['arm']['name'] and r['training_seed']==c['training_seed']]
        summaries.append(dict(arm=c['arm']['name'],training_seed=c['training_seed'],calls=len(measured),
            mean_ms=float(np.mean(measured)),p50_ms=float(np.percentile(measured,50)),
            p95_ms=float(np.percentile(measured,95)),p99_ms=float(np.percentile(measured,99))))
    for filename,values in [('raw_timings.csv',rows),('summary.csv',summaries)]:
        with (args.output/filename).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    receipt=dict(status='completed',measured_calls=len(rows),cases=len(summaries),
        new_training_steps=0,new_simulation_controls=0,independent_test_used=False,
        inference_is_distinct_from_episode_time=True,
        limitation='Shared CPU host, one short session; no deployment certification or candidate selection',summary=summaries)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
