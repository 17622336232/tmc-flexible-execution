"""Fit equally resourced comparison rankers; no replacement is automatic."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.task_attack.runtime import write_json
from experiments.task_attack_value.forest import predict
from research.attack_learning.train_fixed_slot_models import fit_one,matrix,assess


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Preserve comparison candidates')
    a.output.mkdir(parents=True)
    with gzip.open(a.dataset/'training_values.json.gz','rt') as f:data=json.load(f)
    train=[c for c in data['cases'] if c['train']];valid=[c for c in data['cases'] if c['calibration']]
    old=json.loads(Path('models/task_attack/integrity_v19/bundle.json').read_text())
    if set(old['training_scene_seeds'])&{c['seed'] for c in valid}:raise RuntimeError('Old comparison training leaks into these validation worlds')
    models={};choices={};metrics={};old_metrics={}
    for kind in ('static','dynamic','known_static','known_dynamic'):
        models[kind],choices[kind],metrics[kind]=fit_one(train,valid,kind)
        x,y,_,groups=matrix(valid,kind);old_metrics[kind]=assess(predict(old['value_models'][kind],x),y,groups)
        write_json(a.output/'progress.json',dict(completed=list(models),new_metrics=metrics,old_metrics=old_metrics))
    write_json(a.output/'ranker_candidates.json',dict(status='COMMON-RESOURCE comparison candidates; not deployed, not actual ASR',
        value_models=models,selected_configurations=choices,calibration=metrics,old_v19_on_same_prefixes=old_metrics,
        training_worlds=sorted({c['seed'] for c in train}),calibration_worlds=sorted({c['seed'] for c in valid}),
        dataset_sha256=hashlib.sha256((a.dataset/'training_values.json.gz').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_automatic_replacement=True,remaining='Matched full-policy crossfit continuation and actual paired engineering checks required. Preserve stronger old known/reference as an alternative.'))
    print('COMMON RANKER FIT COMPLETE; no online policy replaced',flush=True)


if __name__=='__main__':main()
