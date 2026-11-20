"""Bind the completed two-parent development campaign to its two data disks.

Writes execution descriptions only. Does not launch training or select a seed.
The scientific recipe remains the previously registered three-arm template.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import validate_resolved


SOURCE = '77a517d5'
ROOTS = {
    'CTAlab': '/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12',
    '447': '/home/carla/4T/SAFEFLOW2027_SIMULATION_V12',
}
PYTHONS = {
    'CTAlab': '/home/lzqw/SAFEFLOW2027_ARM/env/bin/python',
    '447': '/home/carla/4T/SAFEFLOW2027_ARM/env/bin/python',
}
ASSETS = {
    'CTAlab': '/home/lzqw/SAFEFLOW2027_PAPER/analysis/piper/assets-ac41fcb.json',
    '447': '/home/carla/4T/SAFEFLOW2027_PAPER/analysis/piper/assets-ac41fcb.json',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    plans = {}
    for seed in (41028, 41029):
        path = directory / f'reach-three-arm150K-{seed}.json'
        plans[seed] = json.loads(path.read_text())
        validate_resolved(plans[seed])
    jobs = []
    for seed, host in ((41028, 'CTAlab'), (41029, '447')):
        root = ROOTS[host]
        for index, arm in enumerate(plans[seed]['arms']):
            jobs.append(dict(seed=seed, arm=arm, host=host,
                output=f'{root}/runs-crossseed900K-20261120/Reach-{seed}-{arm}',
                visible_devices=str(index % 2) if host == 'CTAlab' else '0'))
    admission = dict(status='admitted_fixed_crossseed900K_development',
        template_sha256=plans[41028]['template_sha256'],
        runner_source_commit=SOURCE,
        runner_sha256=digest(Path(__file__).with_name('train_reach_crossseed_v12.py')),
        new_training_steps=900000, independent_test_used=False,
        scientific_early_stopping=False, automatic_retry=False,
        storage_policy='40GiB_floor_plus_unwritten_registered_output_v1',
        projected_output_gib_per_local_arm=2, data_roots_by_host=ROOTS,
        plan_sha256_by_seed={str(seed): digest(directory / f'reach-three-arm150K-{seed}.json')
                            for seed in plans}, jobs=jobs,
        same_seed_methods_remain_on_one_host=True,
        shared_gpu_foreign_processes_untouched=True,
        physical_robot_interactions=0,
        completed_parent_analysis='independent_radius_complete_native/receipt.json')
    outputs = {'execution_registration.json': admission}
    for host, seed in (('CTAlab', 41028), ('447', 41029)):
        root = ROOTS[host]
        operation = f'{root}/crossseed900K-20261120'
        local_jobs = []
        for job in jobs:
            if job['host'] != host:
                continue
            local_jobs.append(dict(name=f"Reach-{seed}-{job['arm']}",
                module='scripts.paper_crax_payload.train_reach_crossseed_v12',
                arguments=['--plan', f'{operation}/reach-three-arm150K-{seed}.json',
                    '--parent', plans[seed]['lines']['piper_reach']['parent_path'],
                    '--arm', job['arm'], '--line', 'piper_reach',
                    '--assets-receipt', ASSETS[host], '--data-root', root,
                    '--execution-registration', f'{operation}/execution_registration.json',
                    '--device', 'cuda:0', '--cuda-memory-fraction', '.025'],
                environment={'CUDA_VISIBLE_DEVICES': job['visible_devices']},
                output=job['output'], log=f"{root}/logs/crossseed-{seed}-{job['arm']}.log",
                require_completed_receipt=True, required_steps=550000))
        outputs[f'queue-{host}.json'] = dict(
            project=f'{root}/source-crossseed900K-20261120', python=PYTHONS[host],
            slots=2, state=f'{root}/queue_crossseed900K_20261120.json',
            poll_interval_seconds=30, minimum_free_disk_bytes=46*1024**3,
            projected_additional_output_gib=6,
            gpu_admission={'index': 0, 'minimum_free_mib': 2500},
            jobs=local_jobs, scientific_early_stopping=False,
            automatic_retry=False, foreign_processes_changed=False)
    for name in outputs:
        if (directory / name).exists():
            raise FileExistsError('Preserve existing execution registrations')
    for name, value in outputs.items():
        (directory / name).write_text(json.dumps(value, indent=2)+'\n')


if __name__ == '__main__':
    main()
