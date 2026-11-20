"""Archive all available methods at one fixed DEV node; never mark a partial final."""
import argparse
import json
from pathlib import Path
import platform

from scripts.paper_crax_payload.close_reach_crossseed_20261120 import summarize_groups
from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import sha256, validate_resolved
from scripts.paper_crax_payload.summarize_simulation_v12 import output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(41028, 41029), required=True)
    parser.add_argument('--new-steps', type=int, choices=(20000, 40000, 60000, 80000, 100000, 150000), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve(strict=True)
    if not args.output.resolve().is_relative_to(root):
        raise ValueError('Snapshot must stay on the admitted data disk')
    registration = root/'crossseed900K-20261120/execution_registration.json'
    admission = json.loads(registration.read_text())
    jobs = [j for j in admission['jobs'] if j['seed'] == args.seed]
    if len(jobs) != 3 or any(j['host'] != platform.node() for j in jobs):
        raise ValueError('Only this host registered seed may be archived')
    nodes, episodes, training, provenance, missing = [], [], [], [], []
    for job in jobs:
        directory = Path(job['output'])
        path = directory/f'development_{args.new_steps}.json'
        if not path.exists():
            missing.append(job['arm'])
            continue
        manifest = json.loads((directory/'run_config.json').read_text())
        plan = manifest['plan']
        line = validate_resolved(plan)
        if (manifest['host'] != platform.node() or manifest['arm'] != job['arm']
                or manifest['source_commit'] != admission['runner_source_commit']
                or line['seed'] != args.seed or manifest['parent_sha256'] != line['parent_sha256']):
            raise ValueError('Keep exact original native method, seed and parent')
        ns, es = summarize_groups(json.loads(path.read_text()), plan, job['arm'], args.new_steps)
        nodes.extend(ns); episodes.extend(es)
        raw = (directory/'progress.jsonl').read_text()
        complete_lines = raw.splitlines() if raw.endswith('\n') else raw.splitlines()[:-1]
        rows = [json.loads(s) for s in complete_lines]
        rows = [r for r in rows if r['new_steps'] <= args.new_steps]
        if [r['new_steps'] for r in rows] != list(range(1000, args.new_steps+1, 1000)):
            raise ValueError('Keep the whole observed prefix, not only a favorable interval')
        training.extend(dict(training_seed=args.seed, arm=job['arm'], **r) for r in rows)
        provenance.append(dict(arm=job['arm'], native_evaluation_sha256=sha256(path),
            native_manifest_sha256=sha256(directory/'run_config.json'), parent_sha256=line['parent_sha256']))
    if not provenance:
        raise ValueError('No registered node has completed; no snapshot created')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, records in [('development_nodes.csv', nodes), ('development_episodes.csv', episodes),
                          ('training.csv', training)]:
        output(args.output/name, records)
    (args.output/'receipt.json').write_text(json.dumps(dict(
        status='partial_crossseed_development_snapshot', training_seed=args.seed,
        fixed_new_steps=args.new_steps, available_arms=[r['arm'] for r in provenance],
        missing_arms=missing, provenance=provenance, development_episodes=len(episodes),
        execution_registration_sha256=sha256(registration),
        new_training_steps=0, new_simulation_controls=0, independent_test_used=False,
        winner_confirmation=False, complete_crossseed_campaign=False,
        physical_robot_interactions=0), indent=2)+'\n')


if __name__ == '__main__':
    main()
