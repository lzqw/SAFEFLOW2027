"""Focused tests for the explicit one-seed 40% preliminary snapshot."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/star/build_shapo_p47_preliminary_40pct.py'


def load_module() -> ModuleType:
    """Load the standalone builder without changing package layout."""
    spec = importlib.util.spec_from_file_location('preliminary_builder', SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    """Return one small fixture digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_bundle(module: ModuleType, root: Path, target: object) -> Path:
    """Create one internally checksummed passing target fixture."""
    bundle = root / target.relative_bundle
    (bundle / 'artifacts').mkdir(parents=True)
    (bundle / 'provenance').mkdir(parents=True)
    evaluation = {
        'milestone': target.milestone,
        'episodes': module.EVALUATION_EPISODES,
        'mean_return': 1.25,
        'mean_cost': 2.5,
        'cost_cvar_20': 4.5,
        'training_cumulative_cost': 30.0,
        'success_rate': 0.5,
        'failure_count': 2,
        'failure_rate': 0.02,
        'time_limit_count': 98,
        'time_limit_rate': 0.98,
        'high_risk_vn_vt': 0.8,
        'high_risk_dangerous_outward_velocity': 0.1,
    }
    audit = {
        'status': 'passed',
        'environment': target.environment,
        'training_seed': module.TRAINING_SEED,
        'milestone': target.milestone,
        'source_commit': module.SOURCE_COMMIT,
        'protocol_sha256': module.PROTOCOL_SHA256,
        'matrix_sha256': module.MATRIX_SHA256,
        'frozen_method_canonical_sha256': module.FROZEN_METHOD_SHA256,
        'training_seed_is_statistical_unit': True,
        'used_for_checkpoint_or_configuration_selection': False,
        'diagnostic_only': True,
        'formal_cost_budget': 10.0,
        'evaluation': evaluation,
    }
    files = {
        'milestone_audit.json': json.dumps(audit, sort_keys=True) + '\n',
        'artifacts/evaluation.csv': 'milestone,mean_return\n1,1.25\n',
        f'artifacts/evaluation_step_{target.milestone}.json': (
            json.dumps(evaluation, sort_keys=True) + '\n'
        ),
        'provenance/resolved_formal_config.yaml': 'env_id: fixture\n',
        'provenance/source_commit.txt': module.SOURCE_COMMIT + '\n',
    }
    for relative, value in files.items():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding='utf-8')
    checksum_rows = [
        f'{digest(bundle / relative)}  {relative}'
        for relative in sorted(files)
    ]
    (bundle / 'checksums.sha256').write_text('\n'.join(checksum_rows) + '\n', encoding='utf-8')
    (bundle / 'milestone_audit_complete').write_text('sealed\n', encoding='utf-8')
    return bundle


def test_waits_without_mutating_output(tmp_path: Path) -> None:
    """Missing targets remain waiting and do not create a partial report."""
    module = load_module()
    output = tmp_path / 'output'
    result = module.build_once(tmp_path / 'milestones', output)
    assert result['state'] == 'waiting'
    assert result['available_targets'] == 0
    assert len(result['missing_targets']) == 4
    assert not output.exists()


def test_builds_sealed_non_formal_bundle_and_rejects_tampering(tmp_path: Path) -> None:
    """All four fixtures produce an immutable n=1 report with strict source checks."""
    module = load_module()
    milestone_root = tmp_path / 'milestones'
    bundles = [create_bundle(module, milestone_root, target) for target in module.TARGETS]
    output = tmp_path / 'output'
    result = module.build_once(milestone_root, output)
    assert result['state'] == 'complete'
    assert result['generated'] is True
    summary = json.loads((output / 'preliminary_summary.json').read_text(encoding='utf-8'))
    assert summary['formal_result'] is False
    assert summary['n_training_seeds'] == 1
    assert summary['snapshot_budget_fraction'] == pytest.approx(0.4)
    assert len(summary['results']) == 4
    report = (output / 'PRELIMINARY_REPORT.md').read_text(encoding='utf-8')
    assert 'PRELIMINARY ONLY' in report
    assert 'not a formal result' in report.lower()
    repeated = module.build_once(milestone_root, output)
    assert repeated['state'] == 'complete'
    assert repeated['generated'] is False

    tampered_root = tmp_path / 'tampered'
    bundle = create_bundle(module, tampered_root, module.TARGETS[0])
    (bundle / 'artifacts/evaluation.csv').write_text('changed\n', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum mismatch'):
        module.collect_targets(tampered_root)
