"""One fixed-layout qualitative rollout; never replaces native evaluation."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from scripts.star.arm_safe_entropy.environment import SafeReachEnv, TaskConfig
from scripts.star.arm_safe_entropy.models import ArmFlowActor, GaussianActor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--checkpoint-steps', type=int, required=True)
    parser.add_argument('--layout-seed', type=int, default=52049)
    parser.add_argument('--sampling-seed', type=int, default=902027)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--slowdown', type=int, default=3)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['method'] not in ['sac_lag', 'flow_iso', 'flow_learned']:
        parser.error('This qualitative renderer only covers the primary three methods')
    if args.slowdown < 1 or args.output.exists():
        parser.error('Use positive slowdown and a new output directory')
    torch.set_num_threads(1)
    saved = torch.load(args.policy, map_location=args.device, weights_only=False)
    state = saved.get('learner', saved)
    if state['updates'] != max(0, args.checkpoint_steps - 2000):
        parser.error('Policy update count does not match the explicit milestone')
    flow = manifest['method'].startswith('flow')
    actor = (ArmFlowActor(25, variance=manifest['reference_variance']) if flow
             else GaussianActor(25)).to(args.device)
    actor.load_state_dict(state['actor'])
    geometry = None
    if manifest['method'] == 'flow_learned':
        from scripts.star.arm_safe_entropy.geometry import LocalMarginEnsemble
        margin = LocalMarginEnsemble().to(args.device)
        margin.load_state_dict(state['margin'])
        geometry = margin.geometry
    # Reset sampling AFTER network construction; neither method's constructor
    # consumes another method's qualitative-rollout random stream.
    torch.manual_seed(args.sampling_seed)
    np.random.seed(args.sampling_seed)
    args.output.mkdir(parents=True)
    env = SafeReachEnv(TaskConfig(**manifest['task_config']))
    records, frames = [], []
    try:
        obs, _ = env.reset(seed=args.layout_seed)
        frames.extend([env.render()] * args.slowdown)
        for index in range(env.config.horizon):
            with torch.no_grad():
                batch = torch.as_tensor(obs[None], dtype=torch.float32, device=args.device)
                result = actor(batch, geometry=geometry) if flow else actor(batch)
                action = result[0][0].cpu().numpy()
            obs, reward, done, truncated, info = env.step(action)
            records.append(dict(step=index + 1, action=action.tolist(), reward=reward, **info))
            frames.extend([env.render()] * args.slowdown)
            if done or truncated:
                break
        imageio.mimsave(args.output / 'rollout.mp4', frames, fps=25, macro_block_size=1)
        payload = dict(kind='qualitative_new_stochastic_rollout_not_primary_evaluation',
            method=manifest['method'], training_seed=manifest['training_seed'],
            policy_training_steps=args.checkpoint_steps, policy_source_commit=manifest['source_commit'],
            render_source_commit=args.source_commit, policy_file=str(args.policy),
            layout_seed=args.layout_seed, sampling_seed=args.sampling_seed,
            policy_sampling='same single stochastic behavior actor; no filtering, retries or reranking',
            layout_selection='fixed52049, known clearance-failure layout in native Learned2028 200k evaluation; new sampling is not a replay of that episode',
            device=args.device, torch=str(torch.__version__), cuda=torch.version.cuda,
            control_dt=.04, video_fps=25, playback_slowdown=args.slowdown,
            new_training_transitions=0, qualitative_rollout_transitions=len(records),
            episode_return=sum(r['reward'] for r in records),
            episode_cost=sum(r['cost'] for r in records), terminal=records[-1], records=records)
        (args.output / 'rollout.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        print(json.dumps({key:payload[key] for key in ['method', 'training_seed', 'policy_training_steps',
            'layout_seed', 'new_training_transitions', 'qualitative_rollout_transitions', 'episode_cost']}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
