"""Compact v6 collection only; pending and negative nodes remain visible."""
import argparse
import csv
import json
from pathlib import Path

from scripts.paper_crax_payload.direction_annealing import sha256, validate_config


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def csv_output(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def summarize(config, root, output, plot=False):
    validate_config(config); inventory = []; training = []; evaluations = []; gradients = []; probes = set(); sources = set()
    for arm in config['arms']:
        directory = root/arm['name']; manifest = load(directory/'run_config.json')
        status = load(directory/'status.json') or dict(state='not_started', steps=100000)
        receipt = load(directory/'receipt.json')
        if manifest:
            if (manifest['configuration'] != config or manifest['arm'] != arm or manifest['engineering_only']
                    or manifest['parent_checkpoint_sha256'] != config['parent']['checkpoint_sha256']
                    or manifest['new_target_transitions'] != 150000):
                raise ValueError('Registered v6 parent/config/budget mismatch')
            probes.add(manifest['probe_sha256']); sources.add(manifest['source_commit'])
        progress_path = directory/'progress.jsonl'
        progress = [json.loads(line) for line in progress_path.read_text().splitlines()] if progress_path.exists() else []
        if any(a['steps'] >= b['steps'] for a,b in zip(progress, progress[1:])):
            raise ValueError('Repeated/backwards suffix chronology')
        for row in progress:
            if row['steps']-100000 != row['new_training_transitions']:
                raise ValueError('Do not count common100K prefix as new training')
            training.append(dict(arm=arm['name'], **row))
        if receipt and (receipt['status'] != 'completed' or receipt['steps'] != 250000
                        or receipt['new_training_transitions'] != 150000 or receipt['engineering_only']
                        or receipt['independent_endpoint_test_executed']
                        or receipt['evaluations'] != config['evaluation']['milestones']
                        or receipt['diagnosis_steps'] != list(range(100000,250001,10000))):
            raise ValueError('Require complete registered scientific suffix, evaluations and diagnosis')
        last = receipt or (progress[-1] if progress else {})
        inventory.append(dict(arm=arm['name'], state=status['state'], completed=bool(receipt),
            actual_absolute_steps=receipt['steps'] if receipt else status['steps'],
            actual_new_steps=(receipt['steps'] if receipt else status['steps'])-100000,
            new_target_steps=150000, new_safe_successes=last.get('new_safe_successes'),
            new_training_violations=last.get('new_training_violations'), new_safe_timeouts=last.get('new_safe_timeouts'),
            safe_successes=last.get('safe_successes'), training_violations=last.get('training_violations'),
            parent_checkpoint_sha256=manifest['parent_checkpoint_sha256'] if manifest else None))
        gradient_path = directory/'gradient_diagnostics.jsonl'
        if gradient_path.exists():
            for line in gradient_path.read_text().splitlines():
                row = json.loads(line)
                if manifest is None or row['probe_sha256'] != manifest['probe_sha256']:
                    raise ValueError('Gradient probe identity mismatch')
                gradients.append(dict(arm=arm['name'], **{k:v for k,v in row.items() if k not in ('norms','cosines','loss_values')},
                    **{'norm_'+k:v for k,v in row['norms'].items()},
                    **{'cos_'+k:v for k,v in row['cosines'].items()},
                    **{'loss_'+k:v for k,v in row['loss_values'].items()}))
        for step in config['evaluation']['milestones']:
            result = load(directory/f'evaluation_{step}.json')
            if not result: continue
            if (result['split'] != 'paired_fork_development_only' or result['independent_confirmation']
                    or result['milestone'] != step):
                raise ValueError('Development-only fixed registered nodes required')
            for condition in result['conditions']:
                records = condition['records']
                if len(records) != 30: raise ValueError('Keep all30 outcomes in each condition')
                evaluations.append(dict(arm=arm['name'], steps=step, condition=condition['condition'], episodes=30,
                    safe_successes=sum(r['safe_success'] for r in records), violations=sum(r['violation'] for r in records),
                    safe_timeouts=sum(r['safe_timeout'] for r in records),
                    mean_return=sum(r['episode_return'] for r in records)/30,
                    mean_capped_seconds=sum(r['capped_seconds'] for r in records)/30,
                    split='paired_fork_development_only'))
    if len(probes) > 1 or len(sources) > 1: raise ValueError('All three share one probe and scientific source')
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('training_inventory',inventory),('training_curve',training),
                           ('development_evaluations',evaluations),('gradient_diagnostics',gradients)]:
        csv_output(output/(filename+'.csv'), rows)
    if plot:
        plot_curves(config, inventory, training, evaluations, output)
        plot_gradient_curves(config, gradients, output)
    result = dict(protocol=config['protocol'], completed_runs=sum(r['completed'] for r in inventory), expected_runs=3,
        actual_new_training_steps=sum(r['actual_new_steps'] for r in inventory), finite_new_training_budget=450000,
        common_observed_parent_prefix_steps=100000, training_seed_count=1, independent_confirmation=False,
        method_selected=False, condition_rows=len(evaluations), gradient_probe_rows=len(gradients),
        new_training_transitions_from_analysis=0, source_commits=sorted(sources),
        inputs=[dict(path=str(p.relative_to(root)), sha256=sha256(p)) for p in sorted(root.rglob('*')) if p.is_file()])
    (output/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'inputs'}))
    return result


def plot_curves(config, inventory, training, evaluations, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(13,7))
    for arm in config['arms']:
        name = arm['name']; rows = [r for r in training if r['arm'] == name]
        for ax,key in zip(axes[0],('new_safe_successes','new_training_violations','new_safe_timeouts')):
            if rows: ax.plot([100,*[r['steps']/1000 for r in rows]],[0,*[r[key] for r in rows]],label=name)
            ax.set(xlabel='Absolute transitions (K), shared parent100K', ylabel=key.replace('_',' '))
            ax.grid(alpha=.2)
        for ax,condition in zip(axes[1],config['evaluation']['conditions']):
            nodes = [r for r in evaluations if r['arm'] == name and r['condition'] == condition['name']]
            if nodes:
                ax.plot([r['steps']/1000 for r in nodes],[100*r['safe_successes']/30 for r in nodes],'o-',label=name)
            ax.set(title=condition['name']+' development', xlabel='Absolute transitions (K)',
                   ylabel='Safe successes (%)',ylim=(-2,102)); ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    if handles: fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.96),ncol=3)
    fig.suptitle(f"v6 paired observed100K fork: {sum(r['completed'] for r in inventory)}/3 complete; one training seed")
    fig.tight_layout(rect=(0,.02,1,.9)); fig.savefig(output/'paired_fork_learning.png',dpi=140); plt.close(fig)


def plot_gradient_curves(config, gradients, output):
    """All fixed-probe nodes, including zero gradients; descriptive, not causal."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    columns = len(config['arms'])
    fig, axes = plt.subplots(2, columns, figsize=(13*columns/3, 7),
                             sharex=True, sharey='row', squeeze=False)
    for column, arm in enumerate(config['arms']):
        rows = [r for r in gradients if r['arm'] == arm['name']]
        steps = [r['absolute_steps']/1000 for r in rows]
        top, bottom = axes[:, column]
        for term in ('reward', 'cost', 'energy', 'direction'):
            top.plot(steps, [r['norm_'+term] for r in rows], 'o-', markersize=3, label=term)
        top.set_yscale('symlog', linthresh=1e-5)
        top.set(title=arm['name'], ylabel='Actor gradient norm (symlog; zero retained)')
        for pair in ('reward__energy', 'reward__direction', 'energy__direction'):
            bottom.plot(steps, [float('nan') if r['cos_'+pair] is None else r['cos_'+pair]
                               for r in rows], 'o-', markersize=3, label=pair.replace('__', ' / '))
        bottom.set(ylim=(-1.05, 1.05), xlabel='Absolute transitions (K)', ylabel='Gradient cosine')
        bottom.axhline(0, color='gray', linewidth=.5)
        for axis in (top, bottom):
            axis.grid(alpha=.2)
            axis.axvline(150, color='gray', linestyle=':', linewidth=.8)
        top.legend(fontsize=8)
        bottom.legend(fontsize=8)
    fig.suptitle('Same64 parent-replay states + fixed noise; one training seed; descriptive DEV only')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output/'fixed_probe_gradient_learning.png', dpi=140)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--plot',action='store_true')
    args=p.parse_args(); summarize(json.loads(args.config.read_text()), args.runs, args.output,args.plot)


if __name__ == '__main__': main()
