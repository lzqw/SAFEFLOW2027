"""Run the locked Payload open-loop offset route diagnostic.

This consumes only route-calibration witnesses from the exact selected policy
bytes. It never selects actions from shifted geometry and never changes the
selection freeze.
"""
import argparse
import json
from pathlib import Path


def main():
    import numpy as np
    from dataclasses import asdict
    from scripts.paper_crax_payload.payload_environment import PayloadConfig, SafePayloadTransport
    from scripts.paper_crax_payload.payload_route_qualification import qualify_offset_route, route_sector
    from scripts.paper_crax_payload.payload_rollouts import replay_open_loop_witness, run_predeclared_alternative

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--training-config', type=Path, required=True)
    parser.add_argument('--selection-freeze', type=Path, required=True)
    parser.add_argument('--calibration-root', type=Path, required=True)
    parser.add_argument('--trace-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    config = json.loads(args.training_config.read_text())
    freeze = json.loads(args.selection_freeze.read_text())
    if freeze.get('status') != 'complete_frozen_selection':
        raise ValueError('Route proof requires a complete common-validation freeze')
    selected = {}
    selected_hashes = {}
    for method, record in freeze['methods'].items():
        candidate = record.get('selected')
        if candidate is None:
            raise ValueError(f'No selected candidate for {method}')
        selected[method] = candidate['candidate']
        selected_hashes[method] = candidate['policy_sha256']

    nominal_records = []
    initial_by_seed = {}
    task = dict(config['task'])
    offset = plan['route_qualification']['fixed_obstacle_offset']
    for method, candidate in selected.items():
        candidate_id = candidate['candidate_id']
        stem = candidate_id.replace(':', '_')
        result_path = args.calibration_root / f'{stem}.json'
        document = json.loads(result_path.read_text())
        if document.get('split') != 'route_calibration':
            raise ValueError('Calibration result has the wrong split')
        records = document['result']['conditions'][0]['records']
        for record in records:
            witness = Path(record['action_witness_file'])
            lines = [json.loads(line) for line in witness.read_text().splitlines()]
            header = lines[0]
            initial = header['initial_state']
            initial_by_seed.setdefault(record['layout_seed'], initial)
            env = SafePayloadTransport(PayloadConfig(**task))
            try:
                replay = replay_open_loop_witness(witness, env, offset)
            finally:
                env.close()
            sector = route_sector(record['trajectory_payload_xyz'], initial['obstacle'])
            nominal_records.append(dict(
                method=method,
                candidate_id=candidate_id,
                layout_seed=record['layout_seed'],
                source_safe_success=bool(record['safe_success']),
                source_violation=bool(record['violation']),
                sector=sector,
                shifted_minimum_distance=replay['shifted_minimum_distance'],
                shifted_safe_success=replay['shifted_safe_success'],
                shifted_violation=replay['shifted_violation'],
                initial_unsafe_after_declared_offset=replay['initial_unsafe_after_declared_offset'],
                native_interval_distance_labels=replay['native_interval_geometry_covers_all_physical_substeps'],
                replay_mode=replay['replay_mode'],
                oracle_used_for_action_selection=replay['oracle_used_for_action_selection'],
                new_control_intervals=replay['new_control_intervals'],
            ))

    alternatives = []
    for seed in sorted(initial_by_seed):
        env = SafePayloadTransport(PayloadConfig(**task))
        try:
            row = run_predeclared_alternative(env, initial_by_seed[seed], offset)
        finally:
            env.close()
        alternatives.append(dict(layout_seed=seed, **row))

    evidence = dict(
        source_split='route_calibration_validation_only',
        selection_freeze_digest=freeze['freeze_digest'],
        selected_policy_sha256=selected_hashes,
        obstacle_offset=offset,
        safety_distance_m=float(PayloadConfig(**task).safety_distance),
        oracle_used_for_action_selection=False,
        native_interval_geometry_covers_all_physical_substeps=True,
        replay_mode='real_open_loop_native_physics',
        nominal_trajectories=nominal_records,
        predeclared_alternative_rollouts=alternatives,
        alternative_controller_changed_after_results=False,
    )
    qualification = qualify_offset_route(plan['route_qualification'], evidence,
                                         selected_hashes, freeze['freeze_digest'])
    output = dict(schema='payload_route_proof_v1', status='completed', source_commit=args.source_commit,
                  selection_freeze_digest=freeze['freeze_digest'], qualification=qualification,
                  evidence=evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(qualification, sort_keys=True))


if __name__ == '__main__':
    main()
