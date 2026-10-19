#!/usr/bin/env python3
"""Publish only the completed, verified host report in an isolated Git worktree."""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = Path('/home/lzqw/PycharmProject/ICLR/ICLR-SAFE')
WORKTREE = Path('/home/lzqw/SAFEFLOW2027_delivery/recovered-final-20261013')
FINAL = Path('/home/lzqw/SAFEFLOW2027_artifacts/recovered_a100_4080/final')
SOURCE = '62ca7399357c10645a955dcbead7ed2153a82c69'
BASE_COMMIT = '427079c7674ba5e9f5eae417700d6bec2a553aa4'
BRANCH = 'codex/shapo-p47-recovered-final-20261013'
REMOTE = 'git@github.com:lzqw/SAFEFLOW2027.git'
REPORT_PATH = 'reports/star/shapo_p47_four_env_artifacts/final'
STATE = BASE / 'github_final_publication.json'
MESSAGE = 'Publish complete recovered SHAPO P4.7 four-environment results'
RECOVERY_RECEIPT = BASE / 'point_button_2029_gpu_cpu_failure_recovery.json'
GOAL_RECOVERY_RECEIPT = BASE / 'point_goal_2029_gpu_cpu_failure_recovery.json'
ANT_RECOVERY_RECEIPT = BASE / 'ant_2027_disk_full_recovery.json'
GOAL_2028_HANDOFF_RECEIPT = BASE / 'point_goal_2028_cpu_gpu_handoff_receipt.json'
BUTTON_2028_RECOVERY_RECEIPT = BASE / 'point_button_2028_gpu_cpu_failure_recovery.json'
BUTTON_2030_HANDOFF_RECEIPT = BASE / 'point_button_2030_4080_handoff_receipt.json'
BUTTON_2030_HANDOFF_STATE = BASE / 'point_button_2030_4080_handoff_state.json'
BUTTON_2031_HANDOFF_RECEIPT = BASE / 'point_button_2031_4080_handoff_receipt.json'
BUTTON_2031_HANDOFF_STATE = BASE / 'point_button_2031_4080_handoff_state.json'
ANT_2031_CPU_PLAN = BASE / 'ant_2031_cpu_handoff_plan.json'
ANT_2031_CPU_RECEIPT = BASE / 'ant_2031_gpu_cpu_handoff_receipt.json'
ANT_2031_CPU_STATE = BASE / 'ant_2031_gpu_cpu_handoff_state.json'
ANT_2030_CPU_PLAN = BASE / 'ant_2030_cpu_handoff_plan.json'
ANT_2030_CPU_RECEIPT = BASE / 'ant_2030_gpu_cpu_handoff_receipt.json'
ANT_2030_CPU_STATE = BASE / 'ant_2030_gpu_cpu_handoff_state.json'
WALKER_2030_CPU_PLAN = BASE / 'walker2d_2030_cpu_handoff_plan.json'
WALKER_2030_CPU_RECEIPT = BASE / 'walker2d_2030_gpu_cpu_handoff_receipt.json'
WALKER_2030_CPU_STATE = BASE / 'walker2d_2030_gpu_cpu_handoff_state.json'
CUDA_RECOVERY_MANIFEST = BASE / 'cuda_runtime_failure_recovery_20261016.json'
REBOOT_4080_MANIFEST = BASE / 'reboot_4080_recovery_20261019.json'
REBOOT_A100_MANIFEST = BASE / 'reboot_a100_recovery_20261019.json'


def run(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}, timeout=1800)


def git(*args, repo=WORKTREE):
    return run('git', '-C', str(repo), *args).stdout.strip()


def readiness():
    if not (FINAL / 'finalization_complete').is_file():
        return False
    metadata = json.loads((FINAL / 'finalization_metadata.json').read_text())
    summary = json.loads((FINAL / 'shapo_p47_final_summary.json').read_text())
    if metadata.get('status') != 'passed' or metadata.get('formal_runs') != 20:
        raise ValueError('finalization did not prove all twenty runs')
    if metadata.get('formal_source_commit') != SOURCE or summary.get('source_commit') != SOURCE:
        raise ValueError('final report uses a different scientific source')
    expected = {'statistical_unit': 'training_seed', 'n_training_seeds_per_environment': 5,
                'evaluation_artifact_count': 60, 'provenance_artifact_count': 80,
                'trajectory_artifact_count': 60, 'checkpoint_artifact_count': 120}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError('complete frozen result matrix is not present')
    if not (FINAL / 'SHAPO_P47_FOUR_ENV_REPORT.md').is_file():
        raise ValueError('final scientific report is missing')
    return True


def remote_head():
    output = git('ls-remote', '--heads', 'origin', f'refs/heads/{BRANCH}')
    return output.split()[0] if output else None


def annotate_recovery(target):
    """Disclose known discarded exploration without changing audited raw metrics."""
    receipt = json.loads(RECOVERY_RECEIPT.read_text())
    expected = {
        'seed': 2029, 'total_training_budget': 10000000,
        'resume_steps': 600000, 'failed_at_steps': 637479,
        'repeated_steps_due_to_recovery': 37479,
        'resume_cumulative_training_cost': 18001,
        'failed_summary_cumulative_training_cost': 18540,
        'training_cost_in_discarded_tail': 539,
        'checkpoint_sha256': '1f3d4a83fb5af9af2203ad6285050cec6e9464c2051b45c449dd3a7a46dd1cb1',
        'scientific_parameters_changed': False,
        'failure_records_preserved': True,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('known recovery accounting changed; reconcile before publication')
    goal_receipt = json.loads(GOAL_RECOVERY_RECEIPT.read_text())
    goal_expected = {
        'seed': 2029, 'total_training_budget': 10000000,
        'resume_steps': 700000, 'failed_at_steps': 703705,
        'repeated_steps_due_to_recovery': 3705,
        'resume_cumulative_training_cost': 15628,
        'failed_summary_cumulative_training_cost': 15628,
        'training_cost_in_discarded_tail': 0,
        'checkpoint_sha256': '633f9c5a415188ba4812915dde5da0024e8c4e859dcb1513c2e1afc0b26a18aa',
        'scientific_parameters_changed': False,
        'failure_records_preserved': True,
    }
    if any(goal_receipt.get(key) != value for key, value in goal_expected.items()):
        raise ValueError('PointGoal recovery accounting changed; reconcile before publication')
    ant_receipt = json.loads(ANT_RECOVERY_RECEIPT.read_text())
    ant_expected = {
        'job': 'ant-2027', 'environment': 'Ant-v4', 'seed': 2027,
        'source_commit': SOURCE, 'total_training_budget': 5000000,
        'last_observed_steps': 4780000, 'resume_steps': 4750000,
        'repeated_steps_due_to_recovery_lower_bound': 30000,
        'resume_cumulative_training_cost': 893,
        'last_observed_cumulative_training_cost': 895,
        'known_training_cost_in_discarded_tail_lower_bound': 2,
        'checkpoint_sha256': 'f9ec2440369101878e9c23c7e90bb308ae285645731b925d3cd4e1a2a01f1b94',
        'scientific_parameters_changed': False,
        'original_log_and_checkpoint_preserved': True,
    }
    if any(ant_receipt.get(key) != value for key, value in ant_expected.items()):
        raise ValueError('Ant disk-full recovery accounting changed; reconcile before publication')
    handoff_receipt = json.loads(GOAL_2028_HANDOFF_RECEIPT.read_text())
    handoff_expected = {
        'job': 'point-goal-2028', 'source_commit': SOURCE,
        'total_training_budget': 10000000, 'resume_steps': 950000,
        'resume_cumulative_training_cost': 14277,
        'checkpoint_sha256': 'da65c3d07b66dd80cf0baad331d4df3fbc2f7d124019126361a7326a89554ed9',
        'unlogged_source_tail_may_exist': True,
        'exact_final_in_memory_source_counters_available': False,
        'scientific_parameters_changed': False,
    }
    if any(handoff_receipt.get(key) != value for key, value in handoff_expected.items()):
        raise ValueError('PointGoal-2028 handoff accounting changed; reconcile before publication')
    button_2028_receipt = json.loads(BUTTON_2028_RECOVERY_RECEIPT.read_text())
    button_2028_expected = {
        'job': 'point-button-2028', 'source_commit': SOURCE,
        'total_training_budget': 10000000, 'failed_at_steps': 855881,
        'resume_steps': 850000, 'repeated_steps_due_to_recovery': 5881,
        'resume_cumulative_training_cost': 24206,
        'failed_summary_cumulative_training_cost': 24206,
        'training_cost_in_discarded_tail': 0,
        'checkpoint_sha256': '092c20289d8f99d175d566eaab47eb4dc1afa1a994ee20ae7ad7b43d11a05882',
        'scientific_parameters_changed': False, 'failure_records_preserved': True,
    }
    if any(button_2028_receipt.get(key) != value for key, value in button_2028_expected.items()):
        raise ValueError('PointButton-2028 recovery accounting changed; reconcile before publication')
    button_2030_receipt = json.loads(BUTTON_2030_HANDOFF_RECEIPT.read_text())
    button_2030_state = json.loads(BUTTON_2030_HANDOFF_STATE.read_text())
    button_2030_expected = {
        'job': 'point-button-2030', 'source_commit': SOURCE,
        'total_training_budget': 10000000, 'resume_steps': 3100000,
        'resume_cumulative_training_cost': 74462,
        'checkpoint_sha256': '56624ff05c648e4ce2f263d71e42ceb4a50ab8e534df08863cfdc307ecbecfbc',
        'unlogged_source_tail_may_exist': True,
        'physical_repeated_steps_and_cost_unknown_not_zero': True,
        'scientific_parameters_changed': False,
    }
    if any(button_2030_receipt.get(key) != value for key, value in button_2030_expected.items()):
        raise ValueError('PointButton-2030 handoff accounting changed; reconcile before publication')
    if (button_2030_state.get('phase') != 'handoff_completed_first_target_epoch_verified'
            or button_2030_state.get('source_retired') is not True):
        raise ValueError('PointButton-2030 handoff was not reconciled')
    button_2031_receipt = json.loads(BUTTON_2031_HANDOFF_RECEIPT.read_text())
    button_2031_state = json.loads(BUTTON_2031_HANDOFF_STATE.read_text())
    button_2031_expected = {
        'job': 'point-button-2031', 'source_commit': SOURCE,
        'total_training_budget': 10000000, 'resume_steps': 3400000,
        'resume_cumulative_training_cost': 91584,
        'last_logged_source_steps': 3405000,
        'last_logged_source_cumulative_training_cost': 91659,
        'known_repeated_environment_steps_lower_bound': 5000,
        'known_additional_training_cost_lower_bound': 75,
        'checkpoint_sha256': 'c2fbe4c5cd5520ce15b2c0978f8f2ac22e52da600ea941f833033964cb9e0e24',
        'physical_totals_are_lower_bounds_unlogged_tail_unknown': True,
        'scientific_parameters_changed': False,
    }
    if any(button_2031_receipt.get(key) != value for key, value in button_2031_expected.items()):
        raise ValueError('PointButton-2031 handoff accounting changed; reconcile before publication')
    if (button_2031_state.get('phase') != 'handoff_completed_first_target_epoch_verified'
            or button_2031_state.get('source_retired') is not True
            or button_2031_state.get('checkpoint_sha256') != button_2031_expected['checkpoint_sha256']):
        raise ValueError('PointButton-2031 handoff was not reconciled')
    cuda_manifest = None
    if CUDA_RECOVERY_MANIFEST.exists():
        cuda_manifest = json.loads(CUDA_RECOVERY_MANIFEST.read_text())
        entries = cuda_manifest['entries']
        if (cuda_manifest.get('source_commit') != SOURCE
                or {e['job'] for e in entries} != {'ant-2031', 'walker2d-2031', 'point-button-2027'}
                or len(entries) != 3
                or sum(e['known_repeated_steps'] for e in entries) != 111548
                or sum(e['known_additional_cost'] for e in entries) != 753):
            raise ValueError('CUDA interruption accounting manifest changed; reconcile before publication')
    ant_cpu_receipt = None
    ant_cpu_state = None
    if ANT_2031_CPU_PLAN.exists() and cuda_manifest is None:
        ant_cpu_receipt = json.loads(ANT_2031_CPU_RECEIPT.read_text())
        ant_cpu_state = json.loads(ANT_2031_CPU_STATE.read_text())
        expected_cpu_handoff = {
            'job': 'ant-2031', 'source_commit': SOURCE,
            'source_host': 'tjucta-a100', 'target_host': 'tjucta-a100',
            'source_device': 'cuda:0', 'target_device': 'cpu',
            'total_training_budget': 5000000,
            'scientific_parameters_changed': False, 'additional_evaluation': False,
            'known_logged_epoch_rollback_steps': 0,
            'physical_repeated_steps_and_cost_unknown_not_zero': True,
            'original_gpu_logs_preserved': True,
        }
        if any(ant_cpu_receipt.get(k) != v for k, v in expected_cpu_handoff.items()):
            raise ValueError('Ant-2031 CPU handoff needs accounting reconciliation')
        resume_steps = int(ant_cpu_receipt['resume_steps'])
        digest = ant_cpu_receipt['checkpoint_sha256']
        if (resume_steps < 3650000 or resume_steps % 50000
                or ant_cpu_receipt.get('last_logged_source_steps') != resume_steps
                or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or ant_cpu_state.get('checkpoint_sha256') != digest
                or ant_cpu_state.get('source_retired') is not True
                or ant_cpu_state.get('phase') != 'handoff_completed_first_cpu_epoch_verified'
                or ant_cpu_state.get('first_verified_target_steps', -1) < resume_steps + 5000):
            raise ValueError('Ant-2031 CPU handoff completion is unproven')
    extra_cpu_handoffs = []
    for cpu_job, minimum_steps, plan_path, receipt_path, state_path in (
            ('ant-2030', 3600000, ANT_2030_CPU_PLAN,
             ANT_2030_CPU_RECEIPT, ANT_2030_CPU_STATE),
            ('walker2d-2030', 3650000, WALKER_2030_CPU_PLAN,
             WALKER_2030_CPU_RECEIPT, WALKER_2030_CPU_STATE)):
        if not plan_path.exists():
            continue
        cpu_receipt = json.loads(receipt_path.read_text())
        cpu_state = json.loads(state_path.read_text())
        expected_cpu = {
            'job': cpu_job, 'source_commit': SOURCE,
            'source_host': 'tjucta-a100', 'target_host': 'tjucta-a100',
            'source_device': 'cuda:0', 'target_device': 'cpu',
            'total_training_budget': 5000000,
            'scientific_parameters_changed': False, 'additional_evaluation': False,
            'known_logged_epoch_rollback_steps': 0,
            'physical_repeated_steps_and_cost_unknown_not_zero': True,
            'original_gpu_logs_preserved': True,
        }
        steps = int(cpu_receipt['resume_steps'])
        digest = cpu_receipt['checkpoint_sha256']
        if (any(cpu_receipt.get(k) != v for k, v in expected_cpu.items())
                or steps < minimum_steps or steps % 50000
                or cpu_receipt.get('last_logged_source_steps') != steps
                or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or cpu_state.get('checkpoint_sha256') != digest
                or cpu_state.get('source_retired') is not True
                or cpu_state.get('phase') != 'handoff_completed_first_cpu_epoch_verified'
                or cpu_state.get('first_verified_target_steps', -1) < steps + 5000):
            raise ValueError(f'{cpu_job} CPU handoff completion/accounting unproven')
        extra_cpu_handoffs.append({'job': cpu_job, 'receipt': cpu_receipt,
            'completed_handoff_state': cpu_state,
            'runtime_placement': 'A100 GPU to spare CPU on the same host',
            'additional_physical_steps': None,
            'additional_physical_training_cost': None, 'unknown_is_not_zero': True})
    cuda_recoveries = []
    if cuda_manifest is not None:
        for expected_recovery in cuda_manifest['entries']:
            job = expected_recovery['job']
            receipt_file = BASE / (job + '_cuda_failure_recovery_20261016_receipt.json')
            proof_file = BASE / (job + '_cuda_failure_recovery_20261016_state.json')
            recovered = json.loads(receipt_file.read_text())
            proof = json.loads(proof_file.read_text())
            if (any(recovered.get(k) != v for k, v in expected_recovery.items())
                    or recovered.get('failure_records_preserved') is not True
                    or recovered.get('scientific_parameters_changed') is not False
                    or recovered.get('source_commit') != SOURCE
                    or proof.get('source_commit') != SOURCE
                    or proof.get('job') != job
                    or proof.get('phase') != 'recovery_started_first_epoch_verified'
                    or proof.get('target_device') != recovered['target_device']
                    or proof.get('checkpoint_sha256') != recovered['checkpoint_sha256']
                    or proof.get('first_verified_steps', -1) < recovered['resume_steps'] + 5000
                    or proof.get('source_failed_with_exit_code') != 1):
                raise ValueError('CUDA recovery not reconciled: ' + job)
            cuda_recoveries.append({'job': job, 'receipt': recovered,
                'first_new_epoch_proof': proof,
                'known_additional_physical_steps': recovered['known_repeated_steps'],
                'known_additional_physical_training_cost': recovered['known_additional_cost']})
    reboot_recoveries = []
    a100_jobs = {'point-goal-2027', 'point-goal-2029', 'point-goal-2030',
                 'point-goal-2031', 'point-button-2027', 'point-button-2029'}
    a100_jobs.update(f'{env}-{seed}' for env in ('ant', 'walker2d') for seed in (2028, 2029, 2030, 2031))
    for manifest_path, host, jobs, repeated_steps, extra_cost, suffix in (
            (REBOOT_4080_MANIFEST, '4080',
             {'point-goal-2028', 'point-button-2028', 'point-button-2030', 'point-button-2031'},
             85000, 1614, '_reboot_recovery_20261019'),
            (REBOOT_A100_MANIFEST, 'tjucta-a100', a100_jobs,
             355000, 2770, '_a100_reboot_recovery_20261019'),
            (BASE / 'reboot_a100_recovery_20261019_1512.json', 'tjucta-a100',
             {'point-goal-2027', 'point-goal-2030', 'point-goal-2031', 'point-button-2027',
              'ant-2030', 'ant-2031', 'walker2d-2030', 'walker2d-2031'},
             150000, 575, '_a100_reboot_recovery_20261019_1512')):
        if not manifest_path.exists():
            continue
        reboot_manifest = json.loads(manifest_path.read_text())
        entries = reboot_manifest['entries']
        if (reboot_manifest.get('source_commit') != SOURCE
                or reboot_manifest.get('host') != host
                or reboot_manifest.get('source_absence_confirmed_after_reboot') is not True
                or reboot_manifest.get('science_changed') is not False
                or len(entries) != len(jobs)
                or {e['job'] for e in entries} != jobs
                or sum(e['known_repeated_steps'] for e in entries) != repeated_steps
                or sum(e['known_additional_cost'] for e in entries) != extra_cost):
            raise ValueError(host + ' reboot recovery manifest needs reconciliation')
        for entry in entries:
            job = entry['job']
            recovered = json.loads((BASE / (job + suffix + '_receipt.json')).read_text())
            proof = json.loads((BASE / (job + suffix + '_state.json')).read_text())
            if (any(recovered.get(k) != v for k, v in entry.items())
                    or recovered.get('source_commit') != SOURCE
                    or recovered.get('original_logs_and_receipts_preserved') is not True
                    or recovered.get('reboot_and_worker_absence_observed') is not True
                    or recovered.get('unknown_unlogged_tail_not_zero') is not True
                    or recovered.get('scientific_parameters_changed') is not False
                    or proof.get('source_commit') != SOURCE
                    or proof.get('job') != job
                    or proof.get('boot_id') != reboot_manifest['expected_boot_id']
                    or proof.get('target_device') != entry['target_device']
                    or ('cuda_visible_devices' in entry and proof.get('cuda_visible_devices') != entry['cuda_visible_devices'])
                    or proof.get('phase') != 'recovery_started_first_epoch_verified'
                    or proof.get('checkpoint_sha256') != entry['checkpoint_sha256']
                    or proof.get('first_verified_steps', -1) < entry['resume_steps'] + 5000):
                raise ValueError(host + ' reboot recovery completion unproven: ' + job)
            reboot_recoveries.append({'job': job, 'host': host, 'receipt': recovered,
                'recovery_manifest': manifest_path.name,
                'first_new_epoch_proof': proof,
                'known_additional_physical_steps': entry['known_repeated_steps'],
                'known_additional_physical_training_cost': entry['known_additional_cost'],
                'unknown_unlogged_tail_not_zero': True})
    gpu_5080_handoffs = []
    gpu_5080_jobs = ['point-goal-2029', 'point-button-2029']
    if (BASE / 'point-goal-2031_5080_handoff_state.json').exists():
        gpu_5080_jobs.append('point-goal-2031')
    if any((BASE / (job + '_5080_handoff_state.json')).exists()
           for job in gpu_5080_jobs):
        for job in gpu_5080_jobs:
            moved = json.loads((BASE / (job + '_5080_handoff_receipt.json')).read_text())
            proof = json.loads((BASE / (job + '_5080_handoff_state.json')).read_text())
            if (moved.get('job') != job or moved.get('source_commit') != SOURCE
                    or moved.get('source_host') != 'tjucta-a100'
                    or moved.get('target_host') != '5080'
                    or moved.get('total_training_budget') != 10000000
                    or moved.get('scientific_parameters_changed') is not False
                    or moved.get('additional_evaluation') is not False
                    or moved.get('full_replay_and_rng_present') is not True
                    or moved.get('original_evidence_preserved') is not True
                    or moved.get('physical_repeated_steps_and_cost_unknown_not_zero') is not True
                    or moved.get('runtime_change', {}).get('to_torch') != '2.7.1+cu128'
                    or moved.get('runtime_change', {}).get('bitwise_equivalence_claimed') is not False
                    or proof.get('checkpoint_sha256') != moved.get('checkpoint_sha256')
                    or proof.get('phase') != 'handoff_completed_first_target_epoch_verified'
                    or proof.get('source_retired') is not True
                    or proof.get('source_canonical_path_vacated_for_final_collection') is not True
                    or proof.get('first_verified_target_steps', -1) < moved['resume_steps'] + 5000
                    or moved['known_repeated_environment_steps_lower_bound'] != moved['last_logged_source_steps'] - moved['resume_steps']
                    or moved['known_additional_training_cost_lower_bound'] != moved['last_logged_source_cumulative_training_cost'] - moved['resume_cumulative_training_cost']
                    or min(moved['known_repeated_environment_steps_lower_bound'], moved['known_additional_training_cost_lower_bound']) < 0):
                raise ValueError('5080 handoff completion/accounting unproven: ' + job)
            gpu_5080_handoffs.append({'job': job, 'receipt': moved,
                'completed_handoff_state': proof,
                'known_additional_physical_steps': moved['known_repeated_environment_steps_lower_bound'],
                'known_additional_physical_training_cost': moved['known_additional_training_cost_lower_bound'],
                'unknown_unlogged_tail_not_zero': True})
    cpu_5080_recoveries = []
    cpu_manifest_path = BASE / 'reboot_cpu_to_5080_20261019_1512.json'
    if cpu_manifest_path.exists():
        cpu_manifest = json.loads(cpu_manifest_path.read_text())
        entries = cpu_manifest['entries']
        if (cpu_manifest.get('source_commit') != SOURCE
                or cpu_manifest.get('expected_boot_id') != '6928f333-1036-473c-92d5-4c482d5fef59'
                or cpu_manifest.get('source_absence_confirmed_after_reboot') is not True
                or {e['job'] for e in entries} != {'ant-2028','ant-2029','walker2d-2028','walker2d-2029'}
                or len(entries) != 4 or sum(e['known_repeated_steps'] for e in entries) != 100000
                or sum(e['known_additional_cost'] for e in entries) != 19):
            raise ValueError('5080 CPU recovery manifest needs reconciliation')
        for entry in entries:
            job = entry['job']
            prefix = job + '_reboot_cpu_5080_20261019_1512'
            moved = json.loads((BASE / (prefix + '_receipt.json')).read_text())
            proof = json.loads((BASE / (prefix + '_state.json')).read_text())
            if (any(moved.get(k) != v for k,v in entry.items())
                    or moved.get('source_commit') != SOURCE
                    or moved.get('source_boot_id') != cpu_manifest['expected_boot_id']
                    or moved.get('source_host') != 'tjucta-a100' or moved.get('target_host') != '5080'
                    or moved.get('scientific_parameters_changed') is not False
                    or moved.get('additional_evaluation') is not False
                    or moved.get('reboot_and_worker_absence_observed') is not True
                    or moved.get('full_replay_and_rng_present') is not True
                    or moved.get('original_logs_and_receipts_preserved') is not True
                    or moved.get('unknown_unlogged_tail_not_zero') is not True
                    or moved.get('runtime_change',{}).get('to_torch') != '2.7.1+cu128'
                    or moved.get('runtime_change',{}).get('bitwise_equivalence_claimed') is not False
                    or proof.get('source_commit') != SOURCE or proof.get('job') != job
                    or proof.get('source_boot_id') != cpu_manifest['expected_boot_id']
                    or proof.get('target_host') != '5080' or proof.get('target_device') != 'cpu'
                    or proof.get('checkpoint_sha256') != entry['checkpoint_sha256']
                    or proof.get('phase') != 'recovery_completed_first_target_epoch_verified'
                    or proof.get('source_absent_after_reboot_confirmed') is not True
                    or proof.get('source_canonical_path_vacated_for_final_collection') is not True
                    or proof.get('first_verified_steps',-1) < entry['resume_steps']+5000):
                raise ValueError('5080 CPU recovery completion/accounting unproven: '+job)
            cpu_5080_recoveries.append({'job':job,'receipt':moved,'first_new_epoch_proof':proof,
                'known_additional_physical_steps':entry['known_repeated_steps'],
                'known_additional_physical_training_cost':entry['known_additional_cost'],
                'unknown_unlogged_tail_not_zero':True})
    near_cpu_rebalances = []
    near_plan_path = BASE / 'a100_gpu_cpu_rebalance_20261019_1550.json'
    if near_plan_path.exists():
        near_plan = json.loads(near_plan_path.read_text())
        entries = near_plan['entries']
        boot_entries = {e['job']:e for e in json.loads((BASE / 'reboot_a100_recovery_20261019_1512.json').read_text())['entries']}
        if (near_plan.get('source_commit') != SOURCE or near_plan.get('science_changed') is not False
                or near_plan.get('scientific_early_stop') is not False
                or near_plan.get('additional_evaluation') is not False
                or len(entries) != 4 or {e['job'] for e in entries} != {'ant-2030','ant-2031','walker2d-2030','walker2d-2031'}):
            raise ValueError('Same-host near-end CPU rebalance plan needs reconciliation')
        for entry in entries:
            job = entry['job']; prefix = job + '_a100_gpu_cpu_rebalance_20261019_1550'
            moved = json.loads((BASE / (prefix + '_receipt.json')).read_text())
            proof = json.loads((BASE / (prefix + '_state.json')).read_text())
            if (any(moved.get(k) != v for k,v in entry.items())
                    or any(entry.get(k) != boot_entries[job].get(k) for k in ('checkpoint','checkpoint_sha256','resume_steps','resume_cumulative_training_cost'))
                    or moved.get('source_commit') != SOURCE
                    or moved.get('boot_id') != near_plan['expected_boot_id']
                    or moved.get('source_device') != 'cuda:0' or moved.get('target_device') != 'cpu'
                    or moved.get('total_training_budget') != 5000000
                    or moved.get('scientific_parameters_changed') is not False
                    or moved.get('additional_evaluation') is not False
                    or moved.get('original_gpu_logs_preserved') is not True
                    or moved.get('unknown_unlogged_tail_not_zero') is not True
                    or moved.get('physical_repeated_steps_and_cost_unknown_not_zero') is not True
                    or moved.get('checkpoint_verification_inherited_from') != 'reboot_a100_recovery_20261019_1512.json'
                    or moved['known_repeated_steps'] != moved['last_logged_source_steps'] - moved['resume_steps']
                    or not 0 <= moved['known_repeated_steps'] < 50000
                    or moved['known_additional_cost'] != moved['last_logged_source_cumulative_training_cost'] - moved['resume_cumulative_training_cost']
                    or moved['known_additional_cost'] < 0
                    or proof.get('job') != job or proof.get('source_commit') != SOURCE
                    or proof.get('boot_id') != near_plan['expected_boot_id']
                    or proof.get('checkpoint_sha256') != entry['checkpoint_sha256']
                    or proof.get('phase') != 'handoff_completed_first_cpu_epoch_verified'
                    or proof.get('source_retired') is not True
                    or proof.get('first_verified_target_steps',-1) < entry['resume_steps']+5000):
                raise ValueError('Same-host CPU rebalance completion/accounting unproven: '+job)
            near_cpu_rebalances.append({'job':job,'receipt':moved,'first_new_epoch_proof':proof,
                'known_additional_physical_steps':moved['known_repeated_steps'],
                'known_additional_physical_training_cost':moved['known_additional_cost'],
                'unknown_unlogged_tail_not_zero':True})
    same_host_gpu_rebalances = []
    gpu_side = 'a100_gpu_rebalance_20261019_1951'
    gpu_plan_path = BASE / (gpu_side + '.json')
    if gpu_plan_path.exists():
        gpu_plan = json.loads(gpu_plan_path.read_text())
        job = 'point-goal-2030'
        prefix = job + '_' + gpu_side
        moved = json.loads((BASE / (prefix + '_receipt.json')).read_text())
        proof = json.loads((BASE / (prefix + '_state.json')).read_text())
        q = gpu_plan['capacity_qualification']
        capacity = moved['target_capacity_at_launch']
        if (gpu_plan.get('source_commit') != SOURCE or gpu_plan.get('job') != job
                or moved.get('job') != job or proof.get('job') != job
                or gpu_plan.get('total_training_budget') != 10000000
                or gpu_plan.get('source_gpu_index') != 0 or gpu_plan.get('target_gpu_index') != 1
                or q.get('status') != 'available_for_rebalance'
                or q['observed_at_unix'] - q['candidate_since_unix'] < 1800
                or moved.get('source_commit') != SOURCE or proof.get('source_commit') != SOURCE
                or moved.get('controller_sha256') != gpu_plan.get('controller_sha256')
                or proof.get('controller_sha256') != gpu_plan.get('controller_sha256')
                or len(gpu_plan.get('controller_sha256', '')) != 64
                or moved.get('boot_id') != gpu_plan['expected_boot_id']
                or proof.get('boot_id') != gpu_plan['expected_boot_id']
                or moved.get('source_host') != 'tjucta-a100' or moved.get('target_host') != 'tjucta-a100'
                or proof.get('source_host') != 'tjucta-a100' or proof.get('target_host') != 'tjucta-a100'
                or proof.get('source_identity') != moved.get('source_identity')
                or moved.get('source_device') != 'cuda:0' or moved.get('target_device') != 'cuda:0'
                or proof.get('target_device') != 'cuda:0'
                or moved.get('source_identity', {}).get('cuda_visible_devices') != '0'
                or moved.get('source_identity', {}).get('pid') != gpu_plan['expected_source_pid']
                or moved.get('target_cuda_visible_devices') != '1'
                or moved.get('total_training_budget') != 10000000
                or moved.get('scientific_parameters_changed') is not False
                or moved.get('additional_evaluation') is not False
                or moved.get('original_gpu_logs_preserved') is not True
                or moved.get('full_replay_and_rng_present') is not True
                or moved.get('unknown_unlogged_tail_not_zero') is not True
                or moved.get('physical_repeated_steps_and_cost_unknown_not_zero') is not True
                or capacity.get('physical_gpu_index') != 1 or capacity.get('foreign_gpu_processes') != 0
                or capacity.get('free_mib', 0) < 16384 or capacity.get('utilization_percent', 100) > 85
                or moved['known_repeated_steps'] != moved['last_logged_source_steps'] - moved['resume_steps']
                or not 0 <= moved['known_repeated_steps'] < 50000
                or moved['known_additional_cost'] != moved['last_logged_source_cumulative_training_cost'] - moved['resume_cumulative_training_cost']
                or moved['known_additional_cost'] < 0
                or proof.get('checkpoint_sha256') != moved['checkpoint_sha256']
                or proof.get('resume_steps') != moved['resume_steps']
                or proof.get('resume_cumulative_training_cost') != moved['resume_cumulative_training_cost']
                or proof.get('known_repeated_steps') != moved['known_repeated_steps']
                or proof.get('known_additional_cost') != moved['known_additional_cost']
                or proof.get('phase') != 'handoff_completed_first_target_epoch_verified'
                or proof.get('source_retired') is not True or proof.get('source_paused') is not False
                or proof.get('canonical_container_unchanged') is not True
                or proof.get('target_identity', {}).get('cuda_visible_devices') != '1'
                or proof.get('target_identity', {}).get('pid') == gpu_plan['expected_source_pid']
                or proof.get('target_identity', {}).get('pid', 0) <= 0
                or proof.get('target_gpu_uuid_verified') != capacity['uuid']
                or proof.get('first_verified_target_steps', -1) < moved['resume_steps'] + 5000
                or proof.get('first_verified_target_steps', 10000001) > 10000000
                or proof.get('first_verified_target_csv') in moved['prior_csvs']):
            raise ValueError('A100 card-to-card handoff completion/accounting unproven: ' + job)
        same_host_gpu_rebalances.append({'job': job, 'plan': gpu_plan, 'receipt': moved,
            'first_new_epoch_proof': proof,
            'known_additional_physical_steps': moved['known_repeated_steps'],
            'known_additional_physical_training_cost': moved['known_additional_cost'],
            'unknown_unlogged_tail_not_zero': True})
    shared_cpu_rebalances = []
    cpu_side = '4080_gpu_cpu_rebalance_20261019_2330'
    cpu_plan_path = BASE / (cpu_side + '.json')
    if cpu_plan_path.exists():
        plan = json.loads(cpu_plan_path.read_text())
        job = 'point-goal-2028'
        prefix = job + '_' + cpu_side
        moved = json.loads((BASE / (prefix + '_receipt.json')).read_text())
        proof = json.loads((BASE / (prefix + '_state.json')).read_text())
        capacity = moved['target_capacity_at_launch']
        controller = BASE / 'rebalance_4080_point_goal_2028_cpu_20261019.py'
        if (plan.get('job') != job or plan.get('source_commit') != SOURCE
                or plan.get('total_training_budget') != 10000000
                or plan.get('source_gpu_index') != 0 or plan.get('target_device') != 'cpu'
                or hashlib.sha256(controller.read_bytes()).hexdigest() != plan.get('controller_sha256')
                or moved.get('controller_sha256') != plan.get('controller_sha256')
                or proof.get('controller_sha256') != plan.get('controller_sha256')
                or any(r.get('job') != job or r.get('source_commit') != SOURCE
                       or r.get('source_host') != '4080' or r.get('target_host') != '4080'
                       or r.get('boot_id') != plan['expected_boot_id']
                       or r.get('source_device') != 'cuda:0' or r.get('target_device') != 'cpu'
                       or r.get('target_cuda_visible_devices') != '-1'
                       or r.get('total_training_budget') != 10000000
                       or r.get('scientific_parameters_changed') is not False
                       or r.get('additional_evaluation') is not False
                       or r.get('original_gpu_logs_preserved') is not True
                       or r.get('full_replay_and_rng_present') is not True
                       or r.get('unknown_unlogged_tail_not_zero') is not True
                       or r.get('physical_repeated_steps_and_cost_unknown_not_zero') is not True
                       for r in (moved, proof))
                or moved['source_identity']['pid'] != plan['expected_source_pid']
                or moved['source_identity']['start_ticks'] != plan['expected_source_start_ticks']
                or moved['source_identity']['cuda_visible_devices'] != '0'
                or proof.get('source_identity') != moved['source_identity']
                or capacity.get('cpu_affinity_count', 0) < 24
                or capacity.get('load_1m', float('inf')) >= capacity['cpu_affinity_count'] * .7
                or capacity.get('memory_available_bytes', 0) < 8 * 1024**3
                or capacity.get('project_free_bytes', 0) < 3 * 1024**3
                or moved['resume_steps'] < plan['minimum_resume_steps']
                or moved['known_repeated_steps'] != moved['last_logged_source_steps'] - moved['resume_steps']
                or not 0 <= moved['known_repeated_steps'] < 50000
                or moved['known_additional_cost'] != moved['last_logged_source_cumulative_training_cost'] - moved['resume_cumulative_training_cost']
                or moved['known_additional_cost'] < 0
                or any(proof.get(key) != moved[key] for key in
                       ('checkpoint_sha256', 'resume_steps', 'resume_cumulative_training_cost',
                        'known_repeated_steps', 'known_additional_cost'))
                or proof.get('phase') != 'handoff_completed_first_target_epoch_verified'
                or proof.get('source_retired') is not True or proof.get('source_paused') is not False
                or proof.get('canonical_container_unchanged') is not True
                or proof.get('target_cpu_placement_verified') is not True
                or proof.get('target_identity', {}).get('device') != 'cpu'
                or proof.get('target_identity', {}).get('cuda_visible_devices') != '-1'
                or proof.get('target_identity', {}).get('pid', 0) <= 0
                or proof.get('target_identity', {}).get('pid') == plan['expected_source_pid']
                or proof.get('first_verified_target_steps', -1) < moved['resume_steps'] + 5000
                or proof.get('first_verified_target_steps', 10000001) > 10000000
                or proof.get('first_verified_target_csv') in moved['prior_csvs']):
            raise ValueError('4080 GPU-to-CPU handoff/accounting unproven: ' + job)
        shared_cpu_rebalances.append({'job': job, 'plan': plan, 'receipt': moved,
            'first_new_epoch_proof': proof,
            'known_additional_physical_steps': moved['known_repeated_steps'],
            'known_additional_physical_training_cost': moved['known_additional_cost'],
            'unknown_unlogged_tail_not_zero': True})
    report = target / 'SHAPO_P47_FOUR_ENV_REPORT.md'
    original = report.read_bytes()
    marker = b'<!-- known-recovery-accounting-v1 -->'
    if marker in original:
        raise ValueError('expected the unmodified audited report before annotation')
    ledger = {
        'schema': 'shapo-known-recovery-accounting-v1',
        'scientific_source_commit': SOURCE,
        'audited_report_before_annotation_sha256': hashlib.sha256(original).hexdigest(),
        'raw_metrics_modified': False,
        'accounting_scope': 'known discarded tail only; not an exhaustive ledger of historical attempts',
        'known_discarded_training_cost': 616 + (753 if cuda_manifest is not None else 0) + sum(r['known_additional_physical_training_cost'] for r in reboot_recoveries + gpu_5080_handoffs + cpu_5080_recoveries + near_cpu_rebalances + same_host_gpu_rebalances + shared_cpu_rebalances),
        'known_discarded_training_cost_is_lower_bound': True,
        'known_repeated_environment_steps': 82065 + (111548 if cuda_manifest is not None else 0) + sum(r['known_additional_physical_steps'] for r in reboot_recoveries + gpu_5080_handoffs + cpu_5080_recoveries + near_cpu_rebalances + same_host_gpu_rebalances + shared_cpu_rebalances),
        'known_repeated_environment_steps_is_lower_bound': True,
        'reported_training_cost_basis': 'accepted checkpoint lineage',
        'interpretation': 'Physically incurred training cost is at least accepted-lineage cost plus 539 for point-button-2029, plus 2 for ant-2027, and plus 75 for point-button-2031. Other historical discarded tails are not reconstructed here.',
        'recoveries': [{'job': 'point-button-2029', 'receipt': receipt},
                       {'job': 'point-goal-2029', 'receipt': goal_receipt},
                       {'job': 'ant-2027', 'receipt': ant_receipt},
                       {'job': 'point-button-2028', 'receipt': button_2028_receipt},
                       {'job': 'point-button-2031', 'receipt': button_2031_receipt,
                        'completed_handoff_state': button_2031_state,
                        'known_additional_physical_steps_lower_bound': 5000,
                        'known_additional_physical_training_cost_lower_bound': 75,
                        'unlogged_source_tail_steps_and_cost': None,
                        'unknown_is_not_zero': True},
                       {'job': 'point-button-2030', 'receipt': button_2030_receipt,
                        'completed_handoff_state': button_2030_state,
                        'additional_physical_steps': None,
                        'additional_physical_training_cost': None,
                        'unknown_is_not_zero': True},
                       {'job': 'point-goal-2028', 'receipt': handoff_receipt,
                        'startup_interruption': 'The first GPU target disappeared after its controller exited, before logging an epoch or terminal counters. Session lifetime was suspected, not proven. Restart used independent tmux from the same verified checkpoint.',
                        'additional_physical_steps': None,
                        'additional_physical_training_cost': None,
                        'unknown_is_not_zero': True}],
    }
    if ant_cpu_receipt is not None:
        ledger['recoveries'].append({
            'job': 'ant-2031', 'receipt': ant_cpu_receipt,
            'completed_handoff_state': ant_cpu_state,
            'runtime_placement': 'A100 GPU to spare CPU on the same host',
            'additional_physical_steps': None,
            'additional_physical_training_cost': None,
            'unknown_is_not_zero': True,
        })
    ledger['recoveries'].extend(extra_cpu_handoffs)
    ledger['recoveries'].extend(cuda_recoveries)
    ledger['recoveries'].extend(reboot_recoveries)
    ledger['recoveries'].extend(gpu_5080_handoffs)
    ledger['recoveries'].extend(cpu_5080_recoveries)
    ledger['recoveries'].extend(near_cpu_rebalances)
    ledger['recoveries'].extend(same_host_gpu_rebalances)
    ledger['recoveries'].extend(shared_cpu_rebalances)
    if cuda_recoveries:
        ledger['interpretation'] += (' The observed CUDA-interruption tails add '
            '14 cost units for ant-2031, 17 for walker2d-2031, and 722 for '
            'point-button-2027, each on top of its accepted-lineage cost.')
    if reboot_recoveries:
        ledger['interpretation'] += (' Reboot-tail adjustments in the per-run receipts '
            'are cumulative with all earlier known tails, including multiple '
            'interruptions of the same training seed; unknown tails are not zero.')
    ledger_path = target / 'recovery_accounting.json'
    ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
    disclosure = (
        '\n\n<!-- known-recovery-accounting-v1 -->\n'
        '## Recovery accounting disclosure\n\n'
        'PointButton seed 2029 stopped with a non-finite actor gradient norm at '
        '637,479 steps on CPU and resumed on A100 from the verified 600,000-step '
        'full checkpoint. The discarded tail contains 37,479 executed steps and '
        '539 observed training-cost units (18,540 minus 18,001).\n\n'
        'PointGoal seed 2029 stopped with the same engineering guard at '
        '703,705 CPU steps and resumed from its verified 700,000-step full '
        'checkpoint on A100. Its discarded tail contains 3,705 executed steps '
        'and zero observed additional training-cost units (15,628 minus 15,628).\n\n'
        'Ant seed 2027 stopped when the 4080 filesystem filled during a backup '
        'transfer. Its last complete CSV/log row records 4,780,000 steps and '
        '895 training-cost units; the verified restore checkpoint records '
        '4,750,000 steps and 893 cost units. The known discarded tail is '
        'therefore at least 30,000 executed steps and 2 training-cost units. '
        'Exact final in-memory counters were not recorded because the disk-full '
        'exception also prevented a training summary. No failed-run final '
        'counters have been invented.\n\n'
        'PointGoal seed 2028 moved from CPU to the 4080 using its verified '
        '950,000-step full checkpoint (14,277 accepted-lineage training-cost '
        'units). The first GPU target disappeared before logging an epoch or '
        'terminal counters and was restarted in independent tmux from the '
        'same checkpoint. Short unlogged CPU and first-target tails may have '
        'executed; their physical steps and costs are unknown, not zero. '
        'Original logs and the checkpoint identity were preserved.\n\n'
        'PointButton seed 2028 stopped at 855,881 CPU steps with a non-finite '
        'actor gradient norm and resumed on the 4080 GPU from its verified '
        '850,000-step full checkpoint. The discarded tail contains 5,881 '
        'executed steps and zero observed additional training-cost units '
        '(24,206 minus 24,206). This is a runtime recovery, not evidence that '
        'numerical stability is permanently fixed.\n\n'
        'PointButton seed 2030 moved from A100 to the 4080 at a verified '
        '3,100,000-step full checkpoint (74,462 accepted-lineage training-cost '
        'units). The source was retired only after the target logged '
        '3,105,000 steps; its original container was preserved in an archive. '
        'No logged epoch rollback occurred, but a short unlogged source tail '
        'may have been replayed. Extra physical steps and costs are unknown, '
        'not assumed zero.\n\n'
        'PointButton seed 2031 moved from A100 to the 4080 from its verified '
        '3,400,000-step full checkpoint, which records 91,584 training-cost '
        'units. The last complete source row records 3,405,000 steps and '
        '91,659 cost units. This handoff therefore replays at least 5,000 '
        'executed steps and 75 training-cost units. Any additional unlogged '
        'source tail remains unknown, not zero. An earlier capacity-guard '
        'cancellation started no target worker and resumed the original '
        'source; its operational record was preserved.\n\n'
        'The raw results above retain the accepted-checkpoint-lineage counters. '
        'For PointButton-2029, physically incurred training cost is therefore **at least '
        'the reported training cost plus 539**. This is a known lower-bound '
        'adjustment, not a reconstruction of every historical recovery attempt. '
        'For Ant-2027, the corresponding lower bound is its reported training '
        'cost plus 2. For PointButton-2031 it is the reported cost plus 75. '
        'Budgets, scientific parameters, raw receipts, and audit gates were not '
        'changed. See [the recovery ledger](recovery_accounting.json) for the '
        'checkpoint identity and preserved operational receipt. This section '
        'was appended during host publication; the ledger records the original '
        'audited report hash.\n'
    )
    if ant_cpu_receipt is not None:
        disclosure += (
            '\nAnt seed 2031 changed runtime placement from A100 GPU to spare '
            'CPU on the same host after a verified full checkpoint at '
            f'{ant_cpu_receipt["resume_steps"]:,} steps. The paused GPU source '
            'was retired only after the CPU target logged a new epoch. '
            'The five-million-step budget and scientific settings were unchanged; '
            'original GPU logs and assignment manifests were retained. No logged '
            'epoch rollback occurred, but a short unlogged source tail may have '
            'been replayed. Its additional physical steps and costs are unknown, '
            'not assumed zero. Checkpoint identity and the actual completed '
            'handoff state are included in the recovery ledger.\n'
        )
    for handoff in extra_cpu_handoffs:
        disclosure += (
            f'\n{handoff["job"]} changed runtime placement from A100 GPU '
            'to spare CPU on the same host after a verified full checkpoint at '
            f'{handoff["receipt"]["resume_steps"]:,} steps. The paused GPU source '
            'was retired only after a new CPU epoch. Scientific settings and the '
            'five-million-step budget were unchanged, and original GPU logs were '
            'preserved. No logged epoch rollback occurred; additional unlogged '
            'physical replay steps and costs remain unknown, not zero. The '
            'actual checkpoint identity and completed state are in the ledger.\n'
        )
    if cuda_recoveries:
        disclosure += (
            '\nThree A100 GPU1 runs were interrupted by `CUDA error: unspecified '
            'launch failure`. The underlying GPU/driver cause was not established. '
            'The shared GPU was not reset and other projects were not paused. '
            'Ant-2031 and Walker-2031 resumed on spare same-host CPU; '
            'PointButton-2027 used a fresh GPU1 process. Each resumed from a '
            'verified full replay/RNG checkpoint with its original full budget. '
            'These are runtime recoveries, not proof of a permanent stability fix.\n'
        )
        for item in cuda_recoveries:
            r = item['receipt']
            disclosure += (
                f'\n- {r["job"]}: interrupted at {r["failed_at_steps"]:,} steps '
                f'and {r["failed_cumulative_training_cost"]:,} cost units; '
                f'resumed at {r["resume_steps"]:,} steps and '
                f'{r["resume_cumulative_training_cost"]:,} accepted-lineage cost. '
                f'The recorded discarded tail contains {r["known_repeated_steps"]:,} '
                f'executed steps and {r["known_additional_cost"]:,} cost units.\n'
            )
    if reboot_recoveries:
        disclosure += (
            '\nOn 19 October the 4080 and A100 were observed after reboot, '
            'with their former workers absent. The 4080 SSH identity matched '
            'the original machine at a new IP address. Each job resumed from '
            'its latest verified full replay/RNG checkpoint with its original '
            'five- or ten-million-step budget unchanged. On A100, six Point '
            'workers were balanced three per GPU and eight Ant/Walker workers '
            'retained CPU placement; PointGoal-2027 moved from physical GPU1 '
            'to GPU0. Scientific settings were not changed. '
            'The exact unlogged pre-reboot tail and exit status are unknown, '
            'not assumed zero. Earlier logs, receipts and checkpoint bytes were preserved.\n'
        )
        for item in reboot_recoveries:
            r = item['receipt']
            disclosure += (
                f'\n- {item["host"]}/{r["job"]} (boot {r["boot_id"][:8]}): last recorded {r["last_logged_source_steps"]:,} '
                f'steps; resumed at {r["resume_steps"]:,}. The known discarded '
                f'tail contains {r["known_repeated_steps"]:,} executed steps and '
                f'{r["known_additional_cost"]:,.0f} training-cost units.\n'
            )
    if (BASE / 'reboot_a100_recovery_20261019_1512.json').exists():
        disclosure += (
            '\nA100 was observed after a second reboot at 15:22 on 19 October '
            '(boot 6928f333), with old workers absent and both GPUs idle. Eight '
            'existing cells resumed four per GPU, including Ant/Walker 2030–2031 '
            'which moved from CPU to GPU. Four longer Ant/Walker 2028–2029 cells '
            'were recovered on idle RTX-5080-host CPU cores. No foreign project '
            'was paused. PointGoal/Button 2029 remained on their already-live '
            '5080 GPU targets and were not restarted on A100. PointButton-2029 '
            'source cleanup was reconciled through observed absence after this '
            'reboot, not a claimed successful SIGTERM during the SSH outage.\n'
        )
    for item in cpu_5080_recoveries:
        r = item['receipt']
        disclosure += (
            f'\n{r["job"]} resumed on 5080-host CPU at {r["resume_steps"]:,} '
            'steps after confirmed A100 reboot/worker absence. PyTorch build '
            'changed from 2.7.1+cu118 to 2.7.1+cu128, with CUDA hidden for '
            'these CPU workers; bitwise equivalence is not claimed. The original '
            'five-million-step budget, frozen algorithm and source evidence were '
            f'preserved. The known reboot tail is at least {r["known_repeated_steps"]:,} '
            f'steps and {r["known_additional_cost"]:,.0f} training-cost units; '
            'unlogged work remains unknown, not zero.\n'
        )
    for item in near_cpu_rebalances:
        r = item['receipt']
        disclosure += (
            f'\n{r["job"]} moved back from A100 GPU to spare same-host CPU '
            'during a resource-only rebalance on 19 October. This freed GPU '
            'compute for the four longer Point cells; no reward/safety metric '
            'was used for selection and no foreign project was paused. '
            f'The verified full checkpoint at {r["resume_steps"]:,} steps '
            'was reused, the five-million-step budget was preserved, and '
            'the source retired only after a new CPU epoch. The known replayed '
            f'tail is at least {r["known_repeated_steps"]:,} steps and '
            f'{r["known_additional_cost"]:,.0f} training-cost units; unlogged '
            'work remains unknown, not zero. Original GPU logs are preserved.\n'
        )
    for item in same_host_gpu_rebalances:
        r = item['receipt']
        disclosure += (
            f'\n{r["job"]} moved between physical A100 cards 0 and 1 after '
            'stable resource-only capacity observations. Logical CUDA device '
            'remained cuda:0; the physical target UUID was verified. The exact '
            f'full replay/RNG checkpoint at {r["resume_steps"]:,} steps and '
            'the ten-million-step budget were retained. The source retired '
            'only after a new target epoch; original evidence was preserved '
            'and no foreign project was stopped. The known replayed tail is '
            f'at least {r["known_repeated_steps"]:,} steps and '
            f'{r["known_additional_cost"]:,.0f} training-cost units; unlogged '
            'work remains unknown, not zero. No scientific parameter changed.\n'
        )
    for item in shared_cpu_rebalances:
        r = item['receipt']
        disclosure += (
            f'\n{r["job"]} moved from heavily shared4080 GPU to spare CPU '
            'on the same host using prior same-configuration native CPU '
            'throughput and current resource headroom, not reward/safety '
            'results. No extra benchmark/evaluation rollouts were added. '
            f'The full replay/RNG checkpoint at {r["resume_steps"]:,} steps '
            'and ten-million-step budget were retained. Source retirement '
            'required a new CPU training epoch. Original logs and input '
            'full checkpoints remain preserved; bitwise equivalence is not '
            'claimed across CPU/GPU. The known replayed tail is at least '
            f'{r["known_repeated_steps"]:,} steps and '
            f'{r["known_additional_cost"]:,.0f} training-cost units; unlogged '
            'work remains unknown, not zero. No other project was paused.\n'
        )
    for item in gpu_5080_handoffs:
        r = item['receipt']
        disclosure += (
            f'\n{r["job"]} moved from shared A100 to RTX 5080 on 19 October, '
            f'resuming the exact full replay/RNG checkpoint at {r["resume_steps"]:,} steps. '
            'The source was paused for transfer and retired only after a new target epoch. '
            'Frozen algorithm settings and the ten-million-step budget were unchanged. '
            'Runtime changed from PyTorch 2.7.1+cu118 to 2.7.1+cu128 for Blackwell support; '
            'bitwise equivalence is not claimed. Original evidence and relocation history '
            f'were retained. The known repeated tail is at least {item["known_additional_physical_steps"]:,} '
            f'steps and {item["known_additional_physical_training_cost"]:,.0f} training-cost units; '
            'the unlogged tail remains unknown, not zero.\n'
        )
    disclosure += (
        '\nThe combined known recovery ledger is a lower bound of '
        f'{ledger["known_repeated_environment_steps"]:,} repeated environment '
        f'steps and {ledger["known_discarded_training_cost"]:,} additional '
        'training-cost units. Other explicitly unknown historical or unlogged '
        'tails remain unknown, not zero. Audited raw counters and the '
        'twenty-run/sixty-evaluation final gates are unchanged.\n'
    )
    report.write_bytes(original + disclosure.encode('utf-8'))
    # Replace only these two hashes; retain every audited evidence checksum.
    checksums = target / 'checksums.sha256'
    rows = checksums.read_text().splitlines()
    names = {'SHAPO_P47_FOUR_ENV_REPORT.md', 'recovery_accounting.json'}
    kept = []
    for row in rows:
        filename = row[66:]
        if filename.removeprefix('./') not in names:
            kept.append(row)
    for name in sorted(names):
        digest = hashlib.sha256((target / name).read_bytes()).hexdigest()
        kept.append(f'{digest}  ./{name}')
    checksums.write_text('\n'.join(kept) + '\n')


def publish():
    if not readiness():
        raise RuntimeError('final evidence is not ready; refusing publication')
    receipt = json.loads(STATE.read_text()) if STATE.exists() else {}
    if not WORKTREE.exists():
        # Never checkout or stage the user's working tree and never force-push.
        if receipt or git('ls-remote', '--heads', 'origin', f'refs/heads/{BRANCH}', repo=REPO):
            raise RuntimeError('existing publication state needs reconciliation')
        WORKTREE.parent.mkdir(parents=True, exist_ok=True)
        git('worktree', 'add', '-b', BRANCH, str(WORKTREE), BASE_COMMIT, repo=REPO)
    if git('symbolic-ref', '--short', 'HEAD') != BRANCH:
        raise RuntimeError('publication worktree branch changed')
    if git('remote', 'get-url', '--push', 'origin') != REMOTE:
        raise RuntimeError('publication remote changed')
    head = git('rev-parse', 'HEAD')
    if head == BASE_COMMIT:
        if remote_head() is not None:
            raise RuntimeError('publication branch appeared remotely; refusing overwrite')
        changes = git('status', '--porcelain=v1', '--untracked-files=all')
        if any(not line[3:].startswith(REPORT_PATH + '/') for line in changes.splitlines()):
            raise RuntimeError('unrelated changes in the isolated publication worktree')
        files = list(FINAL.rglob('*'))
        if any(path.is_symlink() for path in files):
            raise RuntimeError('symlink in public evidence')
        if any(path.is_file() and path.stat().st_size > 95 * 1024**2 for path in files):
            raise RuntimeError('public evidence exceeds GitHub file-size limit')
        total = sum(path.stat().st_size for path in files if path.is_file())
        if shutil.disk_usage(WORKTREE).free < 2 * total + 1024**3:
            raise RuntimeError('not enough host disk space for report and Git objects')
        target = WORKTREE / REPORT_PATH
        target.mkdir(parents=True, exist_ok=True)
        run('rsync', '-a', str(FINAL) + '/', str(target) + '/')
        run('sha256sum', '-c', 'checksums.sha256', cwd=target)
        annotate_recovery(target)
        run('sha256sum', '-c', 'checksums.sha256', cwd=target)
        git('add', '--', REPORT_PATH)
        staged = git('diff', '--cached', '--name-only').splitlines()
        if not staged or any(not path.startswith(REPORT_PATH + '/') for path in staged):
            raise RuntimeError('publication index is empty or contains unrelated paths')
        git('diff', '--cached', '--check')
        git('commit', '-m', MESSAGE, '--', REPORT_PATH)
        head = git('rev-parse', 'HEAD')
    elif git('rev-parse', 'HEAD^') != BASE_COMMIT or git('log', '-1', '--format=%s') != MESSAGE:
        raise RuntimeError('unexpected commit in publication worktree')
    paths = git('show', '--format=', '--name-only', head).splitlines()
    if not paths or any(not path.startswith(REPORT_PATH + '/') for path in paths):
        raise RuntimeError('publication commit contains unrelated changes')
    if git('status', '--porcelain'):
        raise RuntimeError('uncommitted changes remain before GitHub upload')
    remote = remote_head()
    if remote not in (None, head):
        raise RuntimeError('remote publication branch changed; refusing overwrite')
    if remote is None:
        git('push', '--porcelain', 'origin', f'{head}:refs/heads/{BRANCH}')
    if remote_head() != head:
        raise RuntimeError('GitHub does not identify the publication commit')
    payload = {'status': 'published', 'commit': head, 'branch': BRANCH,
               'scientific_source_commit': SOURCE,
               'published_at': datetime.datetime.now().astimezone().isoformat(),
               'report_url': f'https://github.com/lzqw/SAFEFLOW2027/blob/{head}/{REPORT_PATH}/SHAPO_P47_FOUR_ENV_REPORT.md'}
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, STATE)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-readiness', action='store_true')
    args = parser.parse_args()
    print(json.dumps({'ready': readiness()} if args.check_readiness else publish()))
