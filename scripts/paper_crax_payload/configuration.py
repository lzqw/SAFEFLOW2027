"""Small explicit experiment-config inheritance, not runtime silent mutation."""
import json
from pathlib import Path


def load_configuration(path,seen=None):
    path=Path(path).resolve();seen=set() if seen is None else seen
    if path in seen:raise ValueError('Cyclic experiment configuration')
    seen.add(path);data=json.loads(path.read_text())
    if 'base_config' not in data:return data
    base=load_configuration(path.parent/data['base_config'],seen)
    def merge(target,changes):
        for key,value in changes.items():
            if isinstance(value,dict) and isinstance(target.get(key),dict):merge(target[key],value)
            else:target[key]=value
    merge(base,data['overrides'])
    base['explicit_revision']=dict(config_file=str(path),reason=data['reason'],base_config=data['base_config'])
    return base
