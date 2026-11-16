"""Fixed ten-layout Piper Payload motion diagnostic; no training or real hardware."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.paper_crax_payload.piper_train import load_frozen, make_piper_env
from scripts.star.arm_safe_entropy.train import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the fixed development diagnostic')
    config = json.loads(args.config.read_text())
    assets = json.loads(args.assets_receipt.read_text())
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    summary, records = [], []
    for label in ['zero', 'detour', 'Iso', 'Iso-Low', 'EA']:
        learner = None
        if label not in ('zero', 'detour'):
            suffix = {'Iso': 'iso', 'Iso-Low': 'iso-low', 'EA': 'ea'}[label]
            args.checkpoint = args.project_root/'runs'/f'piper-payload-{suffix}-41027-v1/full_300000.pt'
            _, _, saved, learner = load_frozen(args)
            if saved['task'] != 'Payload' or saved['steps'] != 300000:
                raise ValueError('Expected completed300K Payload diagnostic parent')
        env = make_piper_env(config, assets, 'Payload')
        try:
            for index in range(10):
                seed = 650000+index
                torch.manual_seed(690000+index)
                obs, initial = env.reset(seed=seed)
                start = env.tcp_pose()[0]
                initial_distance = float(np.linalg.norm(env.task_position()-env.goal))
                waypoints = [start+[0., -.03, 0.], start+[0., -.03, .14],
                             env.goal+[0., 0., .14], env.goal.copy()]
                waypoint = 0
                episode = []
                for step in range(env.config.horizon):
                    before_tcp = env.tcp_pose()[0]
                    before_task = env.task_position().copy()
                    before_distance = float(np.linalg.norm(before_task-env.goal))
                    if label == 'zero':
                        action = np.zeros(3)
                    elif label == 'detour':
                        if np.linalg.norm(before_tcp-waypoints[waypoint]) < .018 and waypoint < 3:
                            waypoint += 1
                        action = np.clip((waypoints[waypoint]-before_tcp)/env.config.displacement_scale_m, -1., 1.)
                    else:
                        action = learner.action(obs[None])[0]
                    desired = np.clip(before_tcp+action*env.config.displacement_scale_m,
                                      env.config.workspace_min_m, env.config.workspace_max_m)
                    obs, reward, done, truncated, info = env.step(action)
                    after_tcp = env.tcp_pose()[0]
                    row = dict(policy=label, layout_seed=seed, step=step+1,
                        action_norm=float(np.linalg.norm(action)),
                        tcp_displacement_m=float(np.linalg.norm(after_tcp-before_tcp)),
                        payload_displacement_m=float(np.linalg.norm(env.task_position()-before_task)),
                        control_interval_tracking_error_m=float(np.linalg.norm(after_tcp-desired)),
                        goal_distance_m=info['distance_to_goal'],
                        progress_reward=5.*(before_distance-info['distance_to_goal']),
                        time_reward=-.01, success_reward=10.*info['safe_success'],
                        violation_reward=-10.*info['violation'], reward=reward,
                        minimum_distance_m=info['minimum_distance'], margin_label=info['local_margin'],
                        safe_success=info['safe_success'], violation=info['violation'])
                    if abs(row['progress_reward']+row['time_reward']+row['success_reward']+
                           row['violation_reward']-reward) > 1e-6:
                        raise ValueError('Reward decomposition does not match simulator')
                    records.append(row); episode.append(row)
                    if done or truncated:
                        break
                summary.append(dict(policy=label, layout_seed=seed, steps=len(episode),
                    initial_goal_distance_m=initial_distance, final_goal_distance_m=episode[-1]['goal_distance_m'],
                    goal_distance_reduction_m=initial_distance-episode[-1]['goal_distance_m'],
                    safe_success=episode[-1]['safe_success'], violation=episode[-1]['violation'],
                    episode_return=sum(r['reward'] for r in episode),
                    mean_action_norm=float(np.mean([r['action_norm'] for r in episode])),
                    total_tcp_path_m=sum(r['tcp_displacement_m'] for r in episode),
                    mean_control_interval_tracking_error_m=float(np.mean([r['control_interval_tracking_error_m'] for r in episode])),
                    minimum_distance_m=min(r['minimum_distance_m'] for r in episode)))
        finally:
            env.close()
    for name, rows in [('motion_steps.csv', records), ('motion_episodes.csv', summary)]:
        with (args.output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(args.output/'summary.json', dict(training_transitions=0, physical_robot_interactions=0,
        fixed_development_layouts=list(range(650000, 650010)),
        control_interval_tracking_error_is_not_static_IK_residual=True,
        policy_summary=[dict(policy=label, episodes=10,
            safe_success=sum(r['safe_success'] for r in summary if r['policy']==label)/10,
            mean_goal_distance_reduction_m=float(np.mean([r['goal_distance_reduction_m'] for r in summary if r['policy']==label])),
            mean_action_norm=float(np.mean([r['mean_action_norm'] for r in summary if r['policy']==label])))
            for label in ['zero', 'detour', 'Iso', 'Iso-Low', 'EA']]))


if __name__ == '__main__':
    main()
