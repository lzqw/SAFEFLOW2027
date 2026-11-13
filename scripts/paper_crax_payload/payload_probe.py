"""One bounded constructive-path feasibility check; not a learned policy result."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.paper_crax_payload.payload_environment import PayloadConfig, SafePayloadTransport
from scripts.star.arm_safe_entropy.train import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Do not overwrite feasibility evidence')
    config = json.loads(args.config.read_text())
    records = []
    for condition in config['evaluation']['conditions']:
        task = config['task'] | condition.get('task_overrides', {})
        if 'obstacle_offset' in condition:
            task['obstacle_offset'] = condition['obstacle_offset']
        env = SafePayloadTransport(PayloadConfig(**task))
        try:
            _, initial = env.reset(seed=config['evaluation']['development_layout_seed_start'])
            minimum = initial['minimum_distance']
            waypoints = [env.start + [0., -.07, 0.], env.start + [.03, -.07, .20],
                         env.goal + [.03, 0., .20], env.goal]
            target = 0
            for step in range(env.config.horizon):
                position = env.ee_position()
                if np.linalg.norm(position - waypoints[target]) < .025 and target < 3:
                    target += 1
                action = np.clip((waypoints[target] - position) / env.config.displacement_scale, -1., 1.)
                action = np.clip(action * condition.get('action_scale', 1.), -1., 1.)
                _, _, done, truncated, info = env.step(action)
                minimum = min(minimum, info['minimum_distance'])
                if done or truncated:
                    break
            records.append(dict(condition=condition['name'], safe_success=info['safe_success'],
                violation=info['violation'], collision=info['collision'],
                physical_control_intervals=env.physical_control_intervals,
                minimum_distance_m=minimum, arm_minimum_final_interval_m=info['arm_minimum_distance'],
                payload_minimum_final_interval_m=info['payload_minimum_distance']))
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(training_transitions=0, learner_updates=0,
        control='same fixed high detour for all five conditions;not a learned-policy score',
        physical_control_intervals=sum(r['physical_control_intervals'] for r in records), records=records)
    atomic_json(args.output, result)
    print(json.dumps(result))
    if not all(record['safe_success'] for record in records):
        raise RuntimeError('Common constructive route does not establish feasibility for every condition')


if __name__ == '__main__':
    main()
