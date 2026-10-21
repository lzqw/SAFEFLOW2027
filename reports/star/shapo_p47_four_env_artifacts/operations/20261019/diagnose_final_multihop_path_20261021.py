#!/usr/bin/env python3
"""One tiny storage-path fixture plus stat-only confirmation; no model/audit run."""
import ast
import hashlib
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

FROZEN = Path('/home/lzqw/SAFEFLOW2027_delivery/portable-validation-20261013')
COMMIT = '427079c7674ba5e9f5eae417700d6bec2a553aa4'
JOB = 'walker2d-2028'
OLD_ROOT = '/home/autodrive/LZQW/SAFEFLOW2027/runs/shapo_p47_four_env/runs/' + JOB
LAST_ROOT = '/home/carla/SAFEFLOW2027_4080/runs/shapo_p47_four_env/runs/' + JOB
OLD_REF = OLD_ROOT + '/training/STAR-{Walker2d-v4}/seed-2028-2026-10-13-12-03-31/checkpoints/latest_full.pt'


def main():
    assert subprocess.check_output(['git', '-C', str(FROZEN), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    source = (FROZEN / 'scripts/star/validate_shapo_p47_run.py').read_text()
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('load_json', 'resolve_run_path')]
    assert len(functions) == 2
    subset = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + functions, type_ignores=[])
    namespace = {'Path': Path, 'json': json}
    exec(compile(ast.fix_missing_locations(subset), '<exact-frozen-storage-resolver-functions>', 'exec'), namespace)
    origin = {'format': 'shapo-artifact-relocation-v1', 'job': JOB, 'original_host': '4080',
              'target_host': 'tjucta-a100', 'original_container': LAST_ROOT,
              'scientific_protocol_changed': False, 'original_evidence_modified': False}
    previous = dict(origin, original_host='5080', target_host='4080', original_container=OLD_ROOT)
    relative = Path(OLD_REF).relative_to(OLD_ROOT)
    with tempfile.TemporaryDirectory(prefix='safeflow-path-only-fixture-') as temporary:
        d = Path(temporary) / JOB
        (d / relative).parent.mkdir(parents=True)
        (d / relative).write_bytes(b'PATH-ONLY-FIXTURE-NOT-A-MODEL\n')
        (d / 'artifact_relocation.json').write_text(json.dumps(origin))
        (d / 'artifact_relocation.before_collection_fixture.json').write_text(json.dumps(previous))
        single_hop = namespace['resolve_run_path'](d, str(Path(LAST_ROOT) / relative))
        assert single_hop == d / relative
        try:
            namespace['resolve_run_path'](d, OLD_REF)
        except ValueError as error:
            rejected = {'error_type': type(error).__name__, 'error': str(error)}
        else:
            rejected = None
    code = f'''
import json
from pathlib import Path
d=Path({LAST_ROOT!r});r=Path({OLD_REF!r}).relative_to(Path({OLD_ROOT!r}));p=d/r
p.resolve().relative_to(d.resolve())
print(json.dumps({{'copied_old_parent_exists':p.is_file(),'current_copied_path':str(p),
 'current_copied_bytes':p.stat().st_size if p.is_file() else None,'tensor_bytes_or_model_loaded':False}}))
'''
    actual = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', '4080',
                             'taskset -c 16-27 nice -n 10 python3 -c ' + shlex.quote(code)],
                            check=True, text=True, capture_output=True, timeout=45)
    print(json.dumps({'kind': 'targeted_final_import_multihop_storage_path_diagnostic_not_scientific_audit',
        'frozen_controller_commit_unchanged': COMMIT,
        'frozen_validator_file_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'actual_retained_checkpoint_metadata_reference': OLD_REF,
        'actual_reference_obtained_by_safe_pickle_opcode_string_read_bytes': 46203,
        'real_parent_stat_only': json.loads(actual.stdout),
        'future_collector_last_hop_manifest_from_existing_code': origin,
        'previous_relocation_manifest_preserved_but_not_used_by_frozen_resolver': previous,
        'tiny_fixture_last_hop_path_resolves': True,
        'tiny_fixture_old_hop_rejected_even_though_relative_file_exists': rejected,
        'fixture_no_real_checkpoint_payloads_or_torch_imported': True,
        'training_parameters_or_original_artifacts_or_frozen_controller_changed': False,
        'new_model_or_evaluation_runs': 0, 'full_final_scientific_audit_executed': False,
        'missing2027_original_parent_gap_repaired': False,
        'proposed_action_not_executed': 'Request approval for storage-only multi-hop locator update; retain missing-file, same-run, traversal, checkpoint/SHA,20-run/60-eval/80-provenance/60-trajectory/120-identity requirements.'}, indent=2))


if __name__ == '__main__':
    main()
