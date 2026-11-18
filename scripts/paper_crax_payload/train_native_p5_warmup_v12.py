"""Registered native Piper P5/ramp comparison, data-disk artifacts, no hardware."""
import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.paper_crax_payload.direction_warmup_v12 import DirectionWarmup
from scripts.paper_crax_payload.native_p5_adaptation import (
    attach_adaptation, PiperAdaptationLearner)
from scripts.paper_crax_payload.piper_train import make_piper_env
from scripts.paper_crax_payload.train_native_p5_adaptation import train
from scripts.star.arm_safe_entropy.stability_runtime import isolated_rng

FORMAT = 'native_piper_p5_direction_warmup_v12_full'
PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_piper_round1.json'


def validate(plan, args):
    if plan != json.loads(PLAN.read_text()) or args.line != 'piper_reach' or args.arm not in plan['arms']:
        raise ValueError('Use the complete registered native Piper round1, no implicit revisions')
    if plan['additional_steps_per_run'] != 100000 or args.resume:
        raise ValueError('This first v12 execution admits fresh exact100K forks only')


class PiperWarmupLearner(PiperAdaptationLearner):
    def actor_regularizer(self, trace):
        age = self.transitions-self.adaptation['parent_steps']
        return self.warmup.weight(age)*super().actor_regularizer(trace)

    def metrics(self):
        return super().metrics() | dict(direction_weight_effective=self.warmup.weight(
            self.transitions-self.adaptation['parent_steps']))

    def state_dict(self):
        return super().state_dict() | dict(native_direction_warmup_v12=deepcopy(self.warmup_spec))

    def load_state_dict(self, state):
        if state.get('native_direction_warmup_v12') != self.warmup_spec:
            raise ValueError('Strict native v12 arm/ramp identity required')
        super().load_state_dict(state)


def attach(learner, plan, line_name, arm):
    base = deepcopy(plan); base['arms'] = ['p5']
    attach_adaptation(learner, base, line_name, 'p5')
    learner.__class__ = PiperWarmupLearner
    learner.warmup = DirectionWarmup(**plan['candidates'][arm])
    learner.warmup_spec = dict(arm=arm, warmup=asdict(learner.warmup),
        time_basis='new_adaptation_transitions_from_same_native400K_parent')
    return learner


class TrainingTracker:
    def __init__(self, n, success_distance=.03):
        self.success_distance = success_distance
        self.distances = [deque(maxlen=11) for _ in range(n)]
        self.stalled = [False]*n
        self.stall_episodes = self.task_successes = self.stall_windows = self.episodes = 0
        self.total_return = 0.

    def observe(self, i, reward, terminal, info):
        self.total_return += float(reward)
        self.distances[i].append(float(info['distance_to_goal']))
        stall = len(self.distances[i])==11 and self.distances[i][0]-self.distances[i][-1] < .001
        self.stall_windows += int(stall); self.stalled[i] |= stall
        if terminal:
            self.episodes += 1; self.stall_episodes += int(self.stalled[i])
            self.task_successes += int(info['distance_to_goal'] < self.success_distance)
            self.stalled[i] = False; self.distances[i].clear()

    def metrics(self):
        return dict(ordinary_terminal_task_successes=self.task_successes,
            training_stall_episodes=self.stall_episodes, training_stall_windows=self.stall_windows,
            training_cumulative_return=self.total_return,
            training_mean_completed_plus_partial_return=self.total_return/max(1,self.episodes))

    def state_dict(self):
        return dict(distances=[list(v) for v in self.distances],stalled=self.stalled.copy(),
            **{k:deepcopy(getattr(self,k)) for k in
               ('stall_episodes','task_successes','stall_windows','episodes','total_return')})

    def load_state_dict(self, value):
        self.distances = [deque(v,maxlen=11) for v in value['distances']]
        for k,v in value.items():
            if k!='distances':setattr(self,k,deepcopy(v))


def evaluate(learner, config, assets, task, episodes, seed_start):
    groups=[]; count_before=(learner.transitions,learner.updates,learner.actor_updates)
    with isolated_rng():
        for ci, condition in enumerate(config['evaluation']['conditions']):
            env=make_piper_env(config,assets,task,condition.get('simulation_overrides'))
            rows=[]
            try:
                for episode in range(episodes):
                    torch.manual_seed(config['evaluation']['action_sampling_seed']+ci*1000+episode)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(config['evaluation']['action_sampling_seed']+ci*1000+episode)
                    obs,initial=env.reset(seed=seed_start+episode)
                    minimum=initial['minimum_distance']; total=0.; distances=deque(maxlen=11)
                    distances.append(float(np.linalg.norm(env.task_position()-env.goal)))
                    trajectory=[env.task_position().tolist()]; stalled=False; durations=[]
                    for t in range(env.config.horizon):
                        if learner.device.type=='cuda':torch.cuda.synchronize(learner.device)
                        started=time.perf_counter_ns(); action=learner.action(obs[None])[0]
                        if learner.device.type=='cuda':torch.cuda.synchronize(learner.device)
                        durations.append(time.perf_counter_ns()-started)
                        obs,reward,done,truncated,info=env.step(action)
                        trajectory.append(env.task_position().tolist());total+=reward
                        minimum=min(minimum,info['minimum_distance'])
                        distances.append(float(info['distance_to_goal']))
                        stalled |= len(distances)==11 and distances[0]-distances[-1] < .001
                        if done or truncated:break
                    rows.append(dict(seed=seed_start+episode,episode_return=total,
                        safe_success=bool(info['safe_success']),violation=bool(info['violation']),
                        safe_timeout=not info['safe_success'] and not info['violation'],stall=stalled,
                        ordinary_terminal_task_success=bool(info['distance_to_goal'] < env.config.success_distance_m),
                        length=t+1,minimum_distance_m=minimum,collision=bool(info['collision']),
                        time_to_goal_capped_seconds=((t+1) if info['safe_success'] else env.config.horizon)*env.control_dt,
                        trajectory=trajectory,inference_p50_ms=float(np.percentile(durations,50))/1e6,
                        inference_p95_ms=float(np.percentile(durations,95))/1e6))
                groups.append(dict(condition=condition['name'],episodes=episodes,
                    safe_success_count=sum(r['safe_success'] for r in rows),
                    violation_count=sum(r['violation'] for r in rows),
                    safe_timeout_count=sum(r['safe_timeout'] for r in rows),
                    stall_count=sum(r['stall'] for r in rows),
                    **{k:float(np.mean([r[k] for r in rows])) for k in
                       ('safe_success','violation','episode_return','minimum_distance_m','time_to_goal_capped_seconds')},
                    records=rows))
            finally:env.close()
    if count_before!=(learner.transitions,learner.updates,learner.actor_updates):
        raise ValueError('Evaluation modified frozen learner')
    return dict(conditions=groups,physical_robot_interactions=0,new_training_transitions=0,
        evaluation_split='development_only_new_layouts',independent_test_used=False,
        ordinary_success_scope='terminal goal distance regardless of simultaneous violation; no postfailure actions',
        stall_definition='any10-control window with net goal progress less than1mm; diagnostic only',
        inference_timing_scope='sharedGPU full batch1 action call; no simulator time; no hardware claims')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','parent','assets-receipt','output','data-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--line',default='piper_reach');p.add_argument('--arm',required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--cuda-memory-fraction',type=float,default=.05)
    args=p.parse_args();args.resume=None
    train(args,protocol_validator=validate,adaptation_attacher=attach,
        format_name=FORMAT,piper_evaluator=evaluate,tracker_factory=TrainingTracker)


if __name__=='__main__':main()
