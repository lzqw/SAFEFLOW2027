"""Evaluate preserved v1 ability policies on the corrected v2 API, no learning."""
import argparse
import json
from pathlib import Path

import torch

from scripts.star.arm_safe_entropy.environment import TaskConfig
from scripts.star.arm_safe_entropy.models import ArmLearner
from scripts.star.arm_safe_entropy.train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    records = []
    for method in ['sac', 'flow']:
        torch.manual_seed(2027)
        root = Path('/home/lzqw/SAFEFLOW2027_ARM/runs/ability') / (method + '-seed2027-v1')
        config = json.loads((root / 'run_config.json').read_text())
        model = ArmLearner(25, method, 'cuda:0')
        model.load_state_dict(torch.load(root / 'policy_100000.pt', map_location='cuda:0', weights_only=False))
        result = evaluate(model, TaskConfig(obstacle=False), 30)
        records.append(dict(method=method, training_task_version=config['task_config']['version'],
            checkpoint_training_steps=100000, checkpoint_source_commit=config['source_commit'],
            evaluated_task_version=TaskConfig().version, eval_source_commit=args.source_commit,
            checkpoint=str(root / 'policy_100000.pt'), evaluation=result,
            evaluation_transitions=sum(r['length'] for r in result['records'])))
    payload = dict(kind='zero_shot_api_fix_compatibility_not_v2_training', new_training_transitions=0,
                   records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(dict(new_training_transitions=0, results=[dict(method=r['method'],
        success=r['evaluation']['success'], episodes=30) for r in records])))


if __name__ == '__main__':
    main()
