"""Freeze a DEVELOPMENT observer candidate, without replacing frozen v19."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from experiments.task_attack.runtime import write_json
from experiments.task_attack_sensor_proxy.audit import implementation_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--stopping',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Preserve candidate publications')
    a.output.mkdir(parents=True);a.output=a.output.resolve();base_path=Path('models/task_attack/integrity_v19/bundle.json');base=json.loads(base_path.read_text());summary=json.loads((a.stopping/'summary.json').read_text())
    if set(summary['arms'])!={'ours_critic','blind','direct','known_critic'}:raise ValueError('Incomplete comparison training')
    shutil.copy2(a.data/'conditional_critic_candidate.pkl',a.output/'model.pkl');paths={}
    for arm in summary['arms']:
        target=a.output/(arm+'_stopping.pkl');shutil.copy2(a.stopping/(arm+'_stopping.pkl'),target);paths[arm]=str(target)
    seen=set(base['all_development_scene_seeds'])
    for path in Path('artifacts/task_attack').rglob('manifest.json'):
        try:d=json.loads(path.read_text())
        except (OSError,ValueError):continue
        candidates=[d]+d.get('rows',[])+d.get('blocks',[])
        for item in candidates:
            if isinstance(item,dict) and isinstance(item.get('seed'),int):seen.add(item['seed'])
    data_contract=json.loads((a.data/'contract.json').read_text());base['all_development_scene_seeds']=sorted(seen)
    base['conditional_critic_candidate']=dict(status='UNPROMOTED DEVELOPMENT; no actual superiority or near-known evidence',model=str(a.output/'model.pkl'),stopping=paths,
        method='Amortized structured read-dependency hypotheses, task-conditioned counterfactual critic and learned receding-horizon stop/wait',
        full_paper_KDE_reproduction=False,critic_is_clean_route_proxy=True,legacy_fields='Original v19 bundle retained for native schema compatibility; observer override actually uses only this candidate model and stopping assets',
        development_proxy_summary=summary,allowed_policies=['belief','belief_blind','direct_effect','belief_known_task','redvla_external'])
    write_json(a.output/'bundle.json',base)
    contract=json.loads(base_path.with_name('runtime_contract.json').read_text());contract['model_sha256']=hashlib.sha256((a.output/'bundle.json').read_bytes()).hexdigest();contract['observer_extension']='conditional_critic_native_runner; original actor/control/sensors/scorer unchanged';write_json(a.output/'runtime_contract.json',contract)
    source=Path(__file__).parent;dest=a.output/'source_snapshot';dest.mkdir();hashes={}
    for path in source.glob('*.py'):
        shutil.copy2(path,dest/path.name);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    assets={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in a.output.glob('*.pkl')}
    write_json(a.output/'candidate_contract.json',dict(status='DEVELOPMENT frozen candidate; old negative405/v19 unchanged',native_sha256=implementation_hash(),extension_source_sha256=hashes,asset_sha256=assets,
        bundle_sha256=contract['model_sha256'],old_v19_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),all_exposed_worlds_blocked=len(seen),training_data_contract=data_contract,
        formal_superiority_proven=False,near_known_proven=False))
    print('FROZEN DEVELOPMENT CANDIDATE',a.output,'blocked exposed worlds',len(seen),'native unchanged',implementation_hash(),flush=True)


if __name__=='__main__':main()
