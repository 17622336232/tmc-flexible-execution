"""Include the unchanged v15 source and every v16 implementation file."""
import hashlib
import json
from pathlib import Path
import shutil

from experiments.task_attack_value.audit import implementation_hash as parent_hash
from experiments.task_attack_value.audit import snapshot_implementation as snapshot_parent

ROOT=Path(__file__).resolve().parents[2]


def variant_files():
    return sorted(Path(__file__).resolve().parent.glob('*.py'))+[ROOT/'scripts/run_attack_consistent.sh']


def implementation_hash():
    digest=hashlib.sha256(parent_hash().encode()+b'\0')
    for path in variant_files():digest.update(str(path.relative_to(ROOT)).encode()+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest()


def snapshot_implementation(destination):
    destination=Path(destination);snapshot_parent(destination);path=destination/'snapshot_manifest.json';metadata=json.loads(path.read_text())
    metadata['parent_implementation_hash']=metadata['implementation_hash'];metadata['implementation_hash']=implementation_hash()
    for source in variant_files():
        relative=source.relative_to(ROOT);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        metadata['files'].append(dict(path=str(relative),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),copied=True))
    path.write_text(json.dumps(metadata,indent=2)+'\n')
