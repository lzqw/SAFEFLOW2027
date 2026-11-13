"""Fetch pinned official model assets into an external cache, never run vendor scripts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

LOCK_PATH = Path(__file__).with_name('model_lock.json')


def prepare_model(cache: Path) -> dict:
    lock = json.loads(LOCK_PATH.read_text())
    cache = Path(cache).resolve()
    repository = cache / ('piper_ros-' + lock['model_commit'])
    cache.mkdir(parents=True, exist_ok=True)
    if not repository.exists():
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        '--single-branch', '--branch', 'noetic',
                        lock['model_repository'], str(repository)], check=True)
        subprocess.run(['git', '-C', str(repository), 'sparse-checkout', 'set',
                        lock['model_package']], check=True)
        subprocess.run(['git', '-C', str(repository), 'checkout', '--detach',
                        lock['model_commit']], check=True)
    actual = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    if actual != lock['model_commit']:
        raise RuntimeError('Existing asset cache has a different revision; preserve it and use a fresh cache')
    source = repository / lock['model_package'] / lock['provisional_urdf']
    subprocess.run(['git', '-C', str(repository), 'diff', '--exit-code', 'HEAD', '--',
                    lock['model_package']], check=True, stdout=subprocess.DEVNULL)
    if hashlib.sha256(source.read_bytes()).hexdigest() != lock['source_urdf_sha256']:
        raise ValueError('Pinned official URDF digest mismatch; preserve this cache for diagnosis')
    root = ET.parse(source)
    meshes = []
    for mesh in root.iter('mesh'):
        value = mesh.attrib['filename']
        prefix = 'package://piper_description/'
        if not value.startswith(prefix):
            raise ValueError('Unexpected model asset URI')
        path = repository / lock['model_package'] / value[len(prefix):]
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set('filename', str(path))
        meshes.append({'path': str(path.relative_to(repository)),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    generated = cache / ('piper-resolved-' + actual + '.urdf')
    root.write(generated, encoding='utf-8', xml_declaration=True)
    return dict(model_repository=lock['model_repository'], model_commit=actual,
                source_urdf=str(source), resolved_urdf=str(generated),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                resolved_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
                meshes=meshes, hardware_matched=False, asset_edits='package URI resolution only',
                simulator='PyBullet', real_interactions=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error('Preserve the previous asset receipt')
    receipt = prepare_model(args.cache)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'meshes'}))


if __name__ == '__main__':
    main()
