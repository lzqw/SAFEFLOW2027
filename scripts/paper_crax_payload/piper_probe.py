"""Two constructive official-Piper task probes BEFORE any real Piper training."""
import argparse
import json
from pathlib import Path

import numpy as np

from deployment.piper.simulation import PiperSimConfig
from scripts.paper_crax_payload.piper_helpers import PiperTrainingEnvironment
from scripts.star.arm_safe_entropy.train import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets-receipt', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Keep every previous route probe')
    config = json.loads(args.config.read_text())
    assets = json.loads(args.assets_receipt.read_text())
    records = []
    for task in config['tasks']:
        env = PiperTrainingEnvironment(assets, PiperSimConfig(task=task, **config['simulation']))
        try:
            _, initial = env.reset(seed=62027, options={'fixed_layout': True})
            start = env.tcp_pose()[0]
            # Same predeclared high-side detour for both tasks, not Actor or
            # oracle-selected behavior. Refinement failures must remain saved.
            waypoints = [start + [0., -.03, 0.], start + [0., -.03, .14],
                         env.goal + [0., 0., .14], env.goal.copy()]
            index = 0
            trajectory = []
            minimum = initial['minimum_distance']
            for step in range(env.config.horizon):
                tcp = env.tcp_pose()[0]
                if np.linalg.norm(tcp-waypoints[index]) < .018 and index < len(waypoints)-1:
                    index += 1
                action = np.clip((waypoints[index]-tcp)/env.config.displacement_scale_m, -1., 1.)
                _, reward, done, _, info = env.step(action)
                trajectory.append(dict(tcp=env.tcp_pose()[0].tolist(), task_position=env.task_position().tolist(),
                    minimum_distance=info['minimum_distance'], reward=reward))
                minimum = min(minimum, info['minimum_distance'])
                if done:
                    break
            records.append(dict(task=task, safe_success=info['safe_success'], violation=info['violation'],
                minimum_distance=minimum, control_intervals=env.total_control_intervals,
                physics_substeps=env.total_physics_substeps, distance_queries=env.total_distance_queries,
                goal_admissibility_queries=env.goal_admissibility_queries,
                trajectory=trajectory, definition=env.definition()))
        finally:
            env.close()
    result = dict(status='completed_constructive_feasibility_probe', records=records,
        new_training_transitions=0, physical_robot_interactions=0,
        new_simulation_control_intervals=sum(r['control_intervals'] for r in records),
        new_physics_substeps=sum(r['physics_substeps'] for r in records),
        static_and_interval_distance_queries=sum(r['distance_queries'] for r in records),
        learned_policy_result=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))
    print(json.dumps([{k: v for k, v in row.items() if k not in ('trajectory', 'definition')} for row in records]))
    if not all(r['safe_success'] for r in records):
        raise RuntimeError('At least one genuine Piper task lacks a proven common constructive route')


if __name__ == '__main__':
    main()
