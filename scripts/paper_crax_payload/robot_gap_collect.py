"""Compact receipt-based Robot-Gap snapshots; never launch or modify training."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    old_piper, pointgoal, final_rows = [], [], []
    for path in sorted((args.original_root/'runs').glob('piper-*/receipt.json')):
        receipt = json.loads(path.read_text())
        old_piper.append(dict(run=path.parent.name, **receipt))
        eval_path = path.parent/'evaluation_300000.json'
        if eval_path.exists():
            for condition in json.loads(eval_path.read_text())['conditions']:
                final_rows.append(dict(run=path.parent.name, task=receipt['task'], method=receipt['method'],
                    training_steps=receipt['training_steps'], training_seed=receipt['training_seed'],
                    split='development', **{key: condition[key] for key in (
                        'condition', 'episodes', 'safe_success', 'safe_success_count', 'violation',
                        'violation_count', 'safe_timeout_count', 'episode_return')},
                    cumulative_elapsed_seconds=receipt['cumulative_elapsed_seconds'],
                    complete_training_sps=receipt['training_steps']/receipt['cumulative_elapsed_seconds']))
    for path in sorted((args.original_root/'runs').glob('crax-formal-*/receipt.json')):
        pointgoal.append(dict(run=path.parent.name, **json.loads(path.read_text())))
    campaign_path = args.campaign_root/'campaign_status.json'
    campaign = json.loads(campaign_path.read_text()) if campaign_path.exists() else {}
    followup_path = args.campaign_root/'followup_status.json'
    followup = json.loads(followup_path.read_text()) if followup_path.exists() else {}
    new_jobs = []
    for job in campaign.get('jobs', [])+followup.get('jobs', []):
        row = {key: job[key] for key in ('name', 'status', 'pid', 'target_steps', 'engineering')}
        run = Path(job['output'])
        receipt_path = run/'receipt.json'
        progress_path = run/'status.json'
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            row.update(actual_steps=receipt['newly_collected_transitions'],
                complete_training_sps=receipt['complete_training_sps'],
                learner_updates=receipt['learner_updates'], actor_updates=receipt['actor_updates'])
        elif progress_path.exists():
            progress = json.loads(progress_path.read_text())
            row.update(actual_steps=max(0,progress.get('steps', 0)-job.get('initial_steps',0)),
                complete_training_sps=progress.get('complete_training_sps'),
                learner_updates=progress.get('learner_updates'), actor_updates=progress.get('actor_updates'))
        else:
            row['actual_steps'] = 0
        new_jobs.append(row)
    followup_reserved = followup.get('reserved_new_training',0)
    budget = dict(as_of=now, source='actual existing receipts and current new-run status; no historical audit rerun',
        new_round_cap=2200000, initial_phase_reserved=230000,
        initial_phase_reservation_breakdown=dict(three_profiles=30000, paired_transfer=200000),
        reach_geometry_followup_reserved=followup_reserved,
        total_new_round_reserved=230000+followup_reserved,
        actually_launched_target=sum(r['target_steps'] for r in new_jobs),
        observed_new_training_including_partial_prefixes=sum(r['actual_steps'] for r in new_jobs),
        unreserved_later_allowance=1970000-followup_reserved, pointgoal_new_training=0,
        old_piper_completed_candidates=len(old_piper),
        old_piper_actual_candidate_transitions=sum(r['newly_collected_transitions'] for r in old_piper),
        old_pointgoal_formal_completed=len(pointgoal),
        old_pointgoal_actual_transitions=sum(r.get('steps', r.get('training_steps', 0)) for r in pointgoal),
        old_remaining_is_not_reused_as_new_budget=True,
        physical_robot_interactions=0, other_projects_modified=False)
    release_path=args.campaign_root/'reach-release/status.json'
    release=json.loads(release_path.read_text()) if release_path.exists() else {}
    snapshot = dict(as_of=now, campaign=campaign, reach_followups=followup, reach_release=release, budget=budget,
        live_process_proof='Snapshot alone is not proof of liveness; confirm its recorded PID on host',
        all_results_development_not_independent_test=True)
    (args.output/'TASK_STATUS.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    write_csv(args.output/'piper_existing_finals.csv', final_rows)
    write_csv(args.output/'new_training_progress.csv', new_jobs)
    diagnostic = args.campaign_root/'diagnostic'
    if (diagnostic/'summary.json').exists():
        for name in ('summary.json', 'motion_episodes.csv'):
            (args.output/('diagnostic_'+name)).write_bytes((diagnostic/name).read_bytes())
    print(json.dumps(budget, indent=2))


if __name__ == '__main__':
    main()
