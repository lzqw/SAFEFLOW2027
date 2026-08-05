# Copyright 2026 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Small end-to-end integration gate for the ordered STAR P3 loop."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import torch

import omnisafe
from omnisafe.algorithms.off_policy.star import UPDATE_ORDER


def test_star_micro_loop_has_fixed_utd_order_and_artifacts(tmp_path: Path) -> None:
    custom = {
        'seed': 2127,
        'train_cfgs': {
            'device': 'cpu',
            'torch_threads': 1,
            'total_steps': 24,
            'eval_episodes': 0,
        },
        'algo_cfgs': {
            'steps_per_epoch': 12,
            'size': 64,
            'batch_size': 8,
            'random_steps': 4,
            'terminal_utility_steps': 8,
            'geometry_min_steps': 12,
            'w3_td_error_threshold': 100.0,
            'w3_calibration_threshold': 1.0,
            'w3_valid_gradient_threshold': 1.0,
            'w3_disagreement_threshold': 1.0,
            'w3_boundary_samples_threshold': 0,
            'w3_positive_labels_threshold': 999,
            'eval_milestones': [0, 24],
            'counterfactual_batch_size': 8,
            'milestone_checkpoints': True,
        },
        'model_cfgs': {
            'actor': {'hidden_sizes': [16, 16]},
            'critic': {'hidden_sizes': [16, 16], 'num_quantiles': 5},
        },
        'logger_cfgs': {
            'log_dir': str(tmp_path),
            'use_tensorboard': False,
            'use_wandb': False,
        },
    }
    wrapper = omnisafe.Agent('STAR', 'SafetyPointGoal1-v0', custom_cfgs=custom)
    wrapper.learn()
    algorithm = wrapper.agent
    log_dir = Path(algorithm.logger.log_dir)

    assert algorithm.env_steps == 24
    assert algorithm.update_count == 20
    assert algorithm.update_count / (algorithm.env_steps - 4) == 1.0
    assert algorithm.last_update_order == UPDATE_ORDER
    assert algorithm._warmup.stage == 'W2'  # pylint: disable=protected-access
    assert algorithm._buf.device.type == 'cpu'  # pylint: disable=protected-access
    assert (log_dir / 'checkpoints/milestone_0.pt').is_file()
    assert (log_dir / 'checkpoints/milestone_24.pt').is_file()
    assert (log_dir / 'artifacts/counterfactual_step_24.json').is_file()
    evaluation = json.loads(
        (log_dir / 'artifacts/evaluation_step_24.json').read_text(encoding='utf-8'),
    )
    assert evaluation['spearman'] is None

    summary = json.loads((log_dir / 'artifacts/training_summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'completed'
    assert summary['last_update_order'] == list(UPDATE_ORDER)
    assert summary['updates'] == 20
    assert summary['run_env_steps'] == 24
    assert summary['global_safety_constraint_enabled'] is True
    assert summary['local_safety_geometry_enabled'] is True
    manifest = json.loads((log_dir / 'artifacts/run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['p4_2_backbone_ablation'] == {
        'reward_flow_enabled': True,
        'global_safety_constraint_enabled': True,
        'local_safety_geometry_enabled': True,
    }
    with (log_dir / 'artifacts/diagnostics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(float(row['TotalEnvSteps']) in {12.0, 24.0} for row in rows)
    assert all('Critic/RiskP50' in row for row in rows)


def test_star_full_checkpoint_resume_preserves_counters_and_log_interval(tmp_path: Path) -> None:
    source_config = {
        'seed': 2227,
        'train_cfgs': {
            'device': 'cpu',
            'torch_threads': 1,
            'total_steps': 16,
            'eval_episodes': 0,
        },
        'algo_cfgs': {
            'steps_per_epoch': 8,
            'size': 64,
            'batch_size': 8,
            'random_steps': 4,
            'terminal_utility_steps': 8,
            'geometry_min_steps': 12,
            'eval_milestones': [0, 16],
            'milestone_checkpoints': False,
            'checkpoint_replay': True,
        },
        'model_cfgs': {
            'actor': {'hidden_sizes': [16, 16]},
            'critic': {'hidden_sizes': [16, 16], 'num_quantiles': 5},
        },
        'logger_cfgs': {
            'log_dir': str(tmp_path / 'source'),
            'use_tensorboard': False,
            'use_wandb': False,
        },
    }
    source = omnisafe.Agent('STAR', 'SafetyPointGoal1-v0', custom_cfgs=source_config)
    source.learn()
    checkpoint = Path(source.agent.logger.log_dir) / 'checkpoints/final_full.pt'
    assert checkpoint.is_file()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['extra']['resolved_config']['model_cfgs']['actor']['hidden_sizes'] == [16, 16]

    resumed_config = deepcopy(source_config)
    resumed_config['train_cfgs']['total_steps'] = 24
    resumed_config['algo_cfgs']['eval_milestones'] = [0, 16, 24]
    resumed_config['algo_cfgs']['resume_from'] = str(checkpoint)
    resumed_config['logger_cfgs']['log_dir'] = str(tmp_path / 'resumed')
    resumed = omnisafe.Agent('STAR', 'SafetyPointGoal1-v0', custom_cfgs=resumed_config)
    resumed.learn()

    algorithm = resumed.agent
    assert algorithm.env_steps == 24
    assert algorithm.run_env_steps == 8
    assert algorithm.update_count == 20
    assert algorithm._buf.size == 24  # pylint: disable=protected-access
    diagnostics = Path(algorithm.logger.log_dir) / 'artifacts/diagnostics.csv'
    with diagnostics.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert float(rows[0]['TotalEnvSteps']) == 24.0
    assert float(rows[0]['Time/FPS']) > 0.0
    summary = json.loads(
        (Path(algorithm.logger.log_dir) / 'artifacts/training_summary.json').read_text(
            encoding='utf-8',
        ),
    )
    assert summary['run_env_steps'] == 8
    assert summary['end_to_end_fps'] == algorithm.run_env_steps / algorithm.training_seconds
