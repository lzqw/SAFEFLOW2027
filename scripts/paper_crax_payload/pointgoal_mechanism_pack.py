"""Render the completed final64x32 mechanism; no model loading or simulation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


METHODS = ('stable_iso', 'ea')
SAMPLERS = ('iso_reference', 'local', 'endpoint_direct', 'endpoint_pullback')


def prepare(plan, receipt, results, pool, selection):
    """Reject historical/partial studies; preserve every admitted intervention."""
    if not (receipt['status'] == 'completed' and receipt['physical_controls'] == 16704
            and receipt['states'] == 64 and receipt['draws_per_state'] == 32
            and receipt['training_transitions'] == receipt['optimizer_steps'] == 0
            and receipt['no_pool_reselection_after_cost_labels'] is True
            and receipt['analysis_sha256'] == plan['analysis_sha256']
            and plan['planned_total_controls'] == 16704
            and plan['frozen_final_selection_digest'] == selection['freeze_digest']
            and selection['independent_test_used'] is False):
        raise ValueError('Require the completed final selected-model64x32 receipt')
    records = pool['records']
    if len(records) != 64 or len({r['physical_state_sha256'] for r in records}) != 64:
        raise ValueError('Require the original64 distinct physical states')
    expected = {(m, s) for m in METHODS for s in SAMPLERS}
    rows = results['summaries']
    state_groups = results['per_state']
    if (len(rows) != 8 or {(r['method'], r['sampler']) for r in rows} != expected
            or len(state_groups) != 8
            or {(r['method'], r['sampler']) for r in state_groups} != expected):
        raise ValueError('Retain both Actors and all four sampler interventions')
    by_state = {}
    for group in state_groups:
        states = group['states']
        if len(states) != 64 or {r['state_index'] for r in states} != set(range(64)):
            raise ValueError('Missing or duplicate per-state labels')
        by_state[group['method'], group['sampler']] = {r['state_index']: r for r in states}
    packed = []
    for method in METHODS:
        expected_step = selection['methods'][method]['selected']['checkpoint_steps']
        for sampler in SAMPLERS:
            row = dict(next(r for r in rows if (r['method'], r['sampler']) == (method, sampler)))
            if row['states'] != 64 or row['draws'] != 32 or row['checkpoint_steps'] != expected_step:
                raise ValueError('Wrong selected Actor identity or draw count')
            vn, vt = row['endpoint_vn'], row['endpoint_vt']
            row['endpoint_vn_over_vt'] = vn / vt if vn is not None and vt is not None and vt > 0 else None
            packed.append(row)
    diagnostic = {}
    for method in METHODS:
        rates = [[by_state[method, s][i]['native_cost_positive_fraction'] for s in SAMPLERS]
                 for i in range(64)]
        diagnostic[method] = dict(
            states_with_different_cost_event_rates=sum(max(v) - min(v) > 1e-12 for v in rates),
            states_all_zero_every_sampler=sum(max(v) == 0 for v in rates),
            states_all_positive_every_sampler=sum(min(v) == 1 for v in rates),
            note='Within-Actor descriptive comparison; not a significance test or episode safety claim')
    return packed, diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = {name: args.input / f'{name}.json' for name in ('plan', 'receipt', 'results', 'pool')}
    data = {name: json.loads(path.read_text()) for name, path in files.items()}
    selection = json.loads(args.selection.read_text())
    rows, diagnostic = prepare(**data, selection=selection)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'mechanism_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.8), squeeze=False)
    labels = ('Iso ref.', 'Local', 'Direct', 'Pullback')
    metrics = (('endpoint_vn_over_vt', 'Common-axis action variance VN/VT', 1),
               ('mean_endpoint_proxy_margin_m', 'Endpoint-clearance proxy risk (mm)', 1000),
               ('native_mean_cost', 'Native one-control mean cost', 1))
    for index, method in enumerate(METHODS):
        selected = [r for r in rows if r['method'] == method]
        for column, (key, title, scale) in enumerate(metrics):
            ax = axes[index, column]
            values = [float('nan') if r[key] is None else r[key] * scale for r in selected]
            ax.bar(range(4), values, color=('#9daebd', '#b2cfb0', '#e1b27f', '#397ca8'))
            ax.set_xticks(range(4), labels, rotation=18)
            ax.set_title(f'{method} — {title}', fontsize=10)
            ax.grid(axis='y', alpha=.2)
            if key == 'endpoint_vn_over_vt':
                ax.axhline(1, color='#555555', linestyle='--', linewidth=.8)
    figure.suptitle('Frozen selected PointGoal Actors: same64 states,32 draws, shared noise')
    figure.text(.5, .012,
                'Post-hoc native labels; no action/pool reselection. Proxy normal is not a cost-Q or interval-safety certificate.',
                ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .04, 1, .95))
    for suffix in ('png', 'pdf'):
        figure.savefig(args.output / f'mechanism.{suffix}', dpi=170)
    plt.close(figure)
    audit = dict(status='completed_postprocessing_only', input_sha256={
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
        frozen_selection_digest=selection['freeze_digest'], summary_rows=8,
        native_controls_already_in_source_receipt=data['receipt']['physical_controls'],
        new_native_controls=0, training_transitions=0, physical_robot_interactions=0,
        event_rate_diagnostics=diagnostic, normal_scope=data['results']['normal_scope'])
    (args.output / 'figure_receipt.json').write_text(json.dumps(audit, indent=2) + '\n')
    notes = ['# Final selected-model PointGoal mechanism', '',
             'Actual64×32 study, two frozen Actors and all four sampler interventions. '
             'These are common-state training-distribution diagnostics, not independent episode tests. '
             'VN/VT is the ratio of the mean within-state variances on valid common axes; '
             'it is not an average of per-state ratios. All states remain in the admitted pool.', '',
             'The normal comes from native one-control endpoint-clearance proxy finite differences '
             'at zero action, after sampled actions were frozen. It is not a learned cost-Q normal '
             'or a minimum-distance-over-the-control-interval certificate.', '',
             '| Actor | States whose event rates differ across samplers | All-zero states | All-positive states |',
             '|---|---:|---:|---:|']
    for method, value in diagnostic.items():
        notes.append(f"|{method}|{value['states_with_different_cost_event_rates']}/64|"
                     f"{value['states_all_zero_every_sampler']}|{value['states_all_positive_every_sampler']}|")
    notes.extend(['', 'Identical event rates do not prove equal native cost magnitudes. If event rates '
                  'do not change, this fixed pool cannot establish an event-safety benefit. No pool '
                  'replacement, extra sampling or retraining is authorized by this report.', '',
                  '![Native final-model mechanism](mechanism.png)', ''])
    (args.output / 'README.md').write_text('\n'.join(notes))
    print(json.dumps(dict(rows=8, new_native_controls=0, training_transitions=0)))


if __name__ == '__main__':
    main()
