"""Compact publication tables from completed real receipts; no new experiments."""
import csv
import json
from pathlib import Path


def write(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    report=Path('reports/paper_crax_payload');root=report/'arm-mechanism-fe04b21'
    receipt=json.loads((root/'receipt.json').read_text())
    if receipt['status']!='completed':raise ValueError('No completed mechanism receipt')
    rows=[];parameters=[]
    for raw in csv.DictReader((root/'MECHANISM_RESULTS.csv').open()):
        rows.append(dict(task='Reach',source_commit=receipt['source_commit'],checkpoint_steps=1000000,**raw))
        if raw['experiment']=='frozen_parameter_sensitivity':
            parameters.append(dict(task='Reach',phase='frozen_sampler_sensitivity',method=raw['variant'],training_seed=raw['training_seed'],
                configuration=raw['sampler'],rho=raw['rho'],flow_steps=raw['flow_steps'],retrained=False,
                sampler_batch_size=128*64,sampler_batch_seconds=raw['sampling_batch_seconds'],condition=raw['condition'],
                true_interval_risk=raw['true_interval_risk'],full_training_sps='',
                source_commit=receipt['source_commit'],scope='frozen_model_sampler_not_training_speed'))
    for raw in csv.DictReader((root/'PARAMETER_EFFICIENCY.csv').open()):
        parameters.append(dict(task='Reach',phase='frozen_single_action_inference',method=raw['variant'],training_seed=raw['training_seed'],
            configuration=raw['sampler'],flow_steps=4,retrained=False,compile_seconds=raw['compile_seconds'],
            single_action_p50_ms=float(raw['p50_seconds'])*1000,single_action_p95_ms=float(raw['p95_seconds'])*1000,
            full_training_sps='',source_commit=receipt['source_commit'],scope=raw['scope']))
    theory=json.loads((report/'THEORY_NUMERICS.json').read_text())
    for name in ('endpoint_pullback','direction_misalignment','constrained_logdet','same_covariance_relative_entropy'):
        if name in theory:
            value=theory[name]
            rows.append(dict(task='synthetic_linear_flow',experiment='M4',formula=name,
                training_transitions=0,retrained=False,true_interval_risk='',
                numerical_evidence_json=json.dumps(value,separators=(',',':')),
                scope='synthetic_formula_check_not_robot_or_general_dynamics_proof'))
    write(report/'MECHANISM_RESULTS.csv',rows);write(report/'PARAMETER_EFFICIENCY.csv',parameters)
    print(json.dumps(dict(raw_arm_rows=sum(r['task']=='Reach' for r in rows),mechanism_rows=len(rows),parameter_efficiency_rows=len(parameters),new_queries=0,new_training=0)))


if __name__=='__main__':main()
