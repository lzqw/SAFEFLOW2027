"""One-shot E01 final simulation, only after a recorded validation-only review."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time

from scripts.paper_crax_payload.payload_repair_validation import save, sha256


LABELS = ('EA-control', 'EA-repair', 'Iso-repair', 'SAC-repair')
CONDITIONS = ['nominal', 'radius_070', 'payload_wide', 'execution_scale_085']


def admit_review(review, selection, receipt, plan, hashes):
    """Pure metadata gate; no Torch import, simulation, training or selection."""
    if (receipt.get('status') != 'stage1_validation_completed_pending_review' or
            receipt.get('validation_candidates') != 20 or receipt.get('validation_episodes') != 4000 or
            receipt.get('stage1_training_receipts_each') != 100000 or
            receipt.get('total_E01_training_including_failure') != 400325 or
            receipt.get('conditional_second_stage_launched') is not False or
            receipt.get('independent_test_executed') is not False):
        raise ValueError('Require the genuine complete stage1 validation, not a prefix or continuation')
    if (review.get('decision') != 'finalize_after_stage1' or review.get('review_completed') is not True or
            review.get('conditional_continuation') is not False or
            review.get('independent_test_labels_seen') is not False or
            not isinstance(review.get('reason'), str) or not review['reason'].strip()):
        raise ValueError('A substantive validation-only review must freeze the no-continuation decision first')
    if any(review.get(k) != v for k, v in hashes.items()):
        raise ValueError('Review does not bind the exact validation plan, receipt and selection')
    if (receipt['plan_sha256'] != hashes['validation_plan_sha256'] or
            receipt['selection_sha256'] != hashes['validation_selection_sha256']):
        raise ValueError('Validation source hash mismatch')
    if (plan['conditions'] != CONDITIONS or plan['validation_layout_seed_start'] != 81027 or
            plan['validation_episodes_per_condition'] != 50 or plan['training_seed'] != 41028 or
            plan['independent_test_reserved_layout_seed_start'] != 91027 or
            plan['new_candidate_steps'] != [20000, 40000, 60000, 80000, 100000]):
        raise ValueError('Do not change the preregistered candidates or layout streams')
    if selection.get('independent_test_used') is not False or set(selection['selected']) != set(LABELS):
        raise ValueError('Require all four arms and no test-driven selection')
    if set(review.get('selected_checkpoint_sha256', {})) != set(LABELS):
        raise ValueError('Review must freeze all four identities, not just a positive subset')
    for label in LABELS:
        chosen = selection['selected'][label]
        candidates = [c for c in selection['candidates'] if c['label'] == label]
        if (len(candidates) != 5 or sorted(c['steps'] for c in candidates) != plan['new_candidate_steps'] or
                chosen['candidate_count'] != 5 or chosen['candidate'] not in candidates):
            raise ValueError('Require the same five recorded opportunities per arm')
        if review['selected_checkpoint_sha256'][label] != chosen['candidate']['full_checkpoint_sha256']:
            raise ValueError('Selected checkpoint identity changed after review')


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--validation-dir', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validation = args.validation_dir.resolve()
    selection_path, receipt_path = validation/'selection.json', validation/'receipt.json'
    selection, receipt = [json.loads(p.read_text()) for p in (selection_path, receipt_path)]
    review = json.loads(args.review.read_text())
    hashes = dict(validation_plan_sha256=sha256(args.plan), validation_receipt_sha256=sha256(receipt_path),
                  validation_selection_sha256=sha256(selection_path))
    admit_review(review, selection, receipt, plan, hashes)
    root = Path(plan['root']).resolve()
    if (args.output.exists() or root not in args.output.resolve().parents or
            root not in validation.parents or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise ValueError('Use a new output on the registered4T disk, with CUDA hidden')
    if os.stat(root).st_dev == os.stat('/home/carla').st_dev or shutil.disk_usage(root).free < 40*1024**3:
        raise RuntimeError('Require the separate data filesystem and40GiB headroom')
    frozen = []
    for job in plan['jobs']:
        run = root/'runs'/job['run']
        training = json.loads((run/'receipt.json').read_text())
        chosen = selection['selected'][job['label']]
        candidate = chosen['candidate']
        checkpoint = run/f"full_{job['initial_steps']+candidate['steps']}.pt"
        if (training['status'] != 'completed' or training['newly_collected_transitions'] != 100000 or
                training['initial_steps'] != job['initial_steps'] or training['method'] != job['method'] or
                training['training_seed'] != 41028 or str(checkpoint) != candidate['checkpoint'] or
                sha256(checkpoint) != candidate['full_checkpoint_sha256']):
            raise ValueError('Require completed equal NEW training and the frozen native identity')
        frozen.append(dict(**job, candidate=candidate, validation_status=chosen['status'],
                           training_receipt_sha256=sha256(run/'receipt.json')))
    # Atomic one-shot consumption prevents a new output name silently retaking the test.
    stamp = validation/'independent_test_admission.json'
    admission = dict(native_pid=os.getpid(), source_commit=args.source_commit,
        review_sha256=sha256(args.review), output=str(args.output), **hashes,
        new_training_transitions=0, automatic_restart=False)
    with stamp.open('x') as stream:
        json.dump(admission, stream, indent=2)
        stream.write('\n')
    args.output.mkdir()
    save(args.output/'freeze.json', admission|dict(frozen_arms=frozen, review=review,
        layouts=list(range(91027, 91127)), episodes_per_condition=100, conditions=CONDITIONS,
        action_rng='unchanged native condition-level stream; not episode-local noise pairing',
        physical_motion_enabled=False))
    started = time.monotonic()
    state = dict(state='independent_test', native_pid=os.getpid(), source_commit=args.source_commit,
        completed_arms=0, planned_episodes=1600, new_training_transitions=0, automatic_restart=False)
    save(args.output/'status.json', state)
    try:
        # No simulator import occurs until review, identities and one-shot admission pass.
        import torch
        from scripts.paper_crax_payload.payload_train import FORMAT, evaluate_payload
        from scripts.paper_crax_payload.payload_models import build_payload_learner
        torch.set_num_threads(1)
        summaries, episodes, controls = [], [], 0
        for arm in frozen:
            candidate = arm['candidate']
            manifest = json.loads((root/'runs'/arm['run']/'run_config.json').read_text())
            config = manifest['configuration']
            saved = torch.load(candidate['checkpoint'], map_location='cpu', weights_only=False)
            if (saved.get('format') != FORMAT or saved['configuration'] != config or
                    saved['method'] != arm['method'] or saved['training_seed'] != 41028 or
                    saved['steps'] != candidate['full_checkpoint_steps']):
                raise ValueError('Saved native learner identity/configuration mismatch')
            conditions = [c for c in config['evaluation']['conditions'] if c['name'] in CONDITIONS]
            if [c['name'] for c in conditions] != CONDITIONS:
                raise ValueError('Use exactly the original four conditions, no route replacement')
            learner = build_payload_learner(config, arm['method'], 'cpu')
            learner.load_state_dict(saved['learner'])
            before = (learner.updates, learner.actor_updates, learner.reward_updates, learner.safety_multiplier)
            state.update(active_arm=arm['label'], elapsed_seconds=time.monotonic()-started)
            save(args.output/'status.json', state)
            result = evaluate_payload(learner, config, 100, seed_start=91027, conditions=conditions)
            if before != (learner.updates, learner.actor_updates, learner.reward_updates, learner.safety_multiplier):
                raise ValueError('Frozen evaluation changed training/dual counters')
            result.update(label=arm['label'], candidate=candidate, split='independent_test',
                validation_status=arm['validation_status'], source_commit=args.source_commit)
            save(args.output/f"independent-{arm['label']}.json", result)
            for condition in result['conditions']:
                base = dict(label=arm['label'], method=arm['method'], new_checkpoint_steps=candidate['steps'],
                            full_checkpoint_steps=candidate['full_checkpoint_steps'], condition=condition['condition'])
                summaries.append(base|{k:v for k,v in condition.items() if k != 'records'})
                episodes.extend(base|{k:v for k,v in r.items() if k != 'trajectory_payload_xyz'}
                                for r in condition['records'])
            controls += result['physical_evaluation_transitions']
            state['completed_arms'] += 1
            save(args.output/'status.json', state)
            del learner, saved
        if len(episodes) != 1600:
            raise ValueError('Require all planned independent episodes, not a positive subset')
        write_csv(args.output/'independent_summary.csv', summaries)
        write_csv(args.output/'independent_episodes.csv', episodes)
        save(args.output/'receipt.json', dict(status='completed_independent_test', source_commit=args.source_commit,
            **hashes, review_sha256=admission['review_sha256'], frozen_manifest_sha256=sha256(args.output/'freeze.json'),
            independent_episodes=len(episodes), independent_arms=4, native_evaluation_controls=controls,
            new_training_transitions=0, total_E01_training_including_failure=400325,
            conditional_second_stage_launched=False, final_test_reselection=False,
            physical_robot_interactions=0, elapsed_seconds=time.monotonic()-started))
        state.update(state='completed_independent_test', elapsed_seconds=time.monotonic()-started)
        save(args.output/'status.json', state)
    except BaseException as error:
        state.update(state='needs_inspection_no_automatic_retry', error_type=type(error).__name__, error=str(error))
        save(args.output/'status.json', state)
        raise


if __name__ == '__main__':
    main()
