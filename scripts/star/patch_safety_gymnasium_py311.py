#!/usr/bin/env python3
"""Apply the audited Python 3.11 compatibility patch to Safety-Gymnasium 1.0.0."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path


SUPPORTED_VERSION = '1.0.0'
ASSET_FIELDS = {
    'assets/free_geoms/push_box.py': 'push_box',
    'assets/free_geoms/vases.py': 'vase',
    'assets/geoms/apples.py': 'apple',
    'assets/geoms/buttons.py': 'button',
    'assets/geoms/circle.py': 'circle',
    'assets/geoms/goal.py': 'goal',
    'assets/geoms/hazards.py': 'hazard',
    'assets/geoms/oranges.py': 'orange',
    'assets/geoms/pillars.py': 'pillar',
    'assets/geoms/sigwalls.py': 'sigwall',
    'assets/geoms/walls.py': 'wall',
    'assets/mocaps/gremlins.py': 'gremlin',
}


def _replace_once(text: str, old: str, new: str, path: Path) -> tuple[str, bool]:
    """Replace one audited source fragment, while remaining idempotent."""
    if new in text:
        return text, False
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{path}: expected exactly one {old!r}, found {count}')
    return text.replace(old, new), True


def patch_package(package_root: Path) -> int:
    """Patch unhashable numpy dataclass defaults rejected by Python 3.11."""
    changed = 0
    for relative_path, asset_name in ASSET_FIELDS.items():
        path = package_root / relative_path
        text = path.read_text(encoding='utf-8')
        text, did_change = _replace_once(
            text,
            'from dataclasses import dataclass\n',
            'from dataclasses import dataclass, field\n',
            path,
        )
        changed += int(did_change)
        old = f"    color: np.array = COLOR['{asset_name}']"
        new = (
            f'    color: np.ndarray = field('
            f"default_factory=lambda: COLOR['{asset_name}'].copy())"
        )
        text, did_change = _replace_once(text, old, new, path)
        changed += int(did_change)

        # GROUP values are integers and therefore valid dataclass defaults. Undo an early
        # compatibility-patch revision that treated every subscripted value as mutable.
        patched_group = (
            f'    group: np.ndarray = field('
            f"default_factory=lambda: GROUP['{asset_name}'].copy())"
        )
        original_group = f"    group: np.array = GROUP['{asset_name}']"
        if patched_group in text:
            text = text.replace(patched_group, original_group)
            changed += 1
        elif original_group not in text:
            raise RuntimeError(f'{path}: expected group declaration was not found')
        path.write_text(text, encoding='utf-8')
    return changed


def patch_metadata(site_packages: Path) -> int:
    """Record the Python 3.11-compatible pygame wheel constraint for pip check."""
    candidates = sorted(site_packages.glob('safety_gymnasium-*.dist-info/METADATA'))
    if len(candidates) != 1:
        raise RuntimeError(f'expected one Safety-Gymnasium METADATA file, found {candidates}')
    path = candidates[0]
    text = path.read_text(encoding='utf-8')
    replacements = (
        ('Requires-Dist: pygame (==2.1.0)', 'Requires-Dist: pygame (>=2.1.0,<3)'),
        ('Requires-Dist: pygame==2.1.0', 'Requires-Dist: pygame>=2.1.0,<3'),
    )
    for old, new in replacements:
        if new in text:
            return 0
        if old in text:
            path.write_text(text.replace(old, new), encoding='utf-8')
            return 1
    raise RuntimeError(f'{path}: expected pygame 2.1.0 requirement was not found')


def main() -> None:
    """Patch an installed Safety-Gymnasium package."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--package-root',
        type=Path,
        help='Optional safety_gymnasium package directory; defaults to the active environment.',
    )
    parser.add_argument(
        '--skip-metadata',
        action='store_true',
        help='Do not relax the pygame wheel metadata constraint.',
    )
    args = parser.parse_args()

    distribution = importlib.metadata.distribution('safety-gymnasium')
    if distribution.version != SUPPORTED_VERSION:
        raise RuntimeError(
            f'expected safety-gymnasium {SUPPORTED_VERSION}, got {distribution.version}',
        )
    package_root = args.package_root or Path(distribution.locate_file('safety_gymnasium'))
    package_root = package_root.resolve()
    source_changes = patch_package(package_root)
    metadata_changes = 0
    if not args.skip_metadata:
        metadata_changes = patch_metadata(package_root.parent)
    print(
        f'patched safety-gymnasium {SUPPORTED_VERSION}: '
        f'source_changes={source_changes}, metadata_changes={metadata_changes}',
    )


if __name__ == '__main__':
    main()
