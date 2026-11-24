"""Decision-directed OWN proxy training; current test episodes never used."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from experiments.task_attack.runtime import write_json
from experiments.task_attack_value.forest import export,predict
from experiments.task_attack_value.value_features import action_features,source_features
from experiments.task_attack_value.audit import snapshot_implementation,implementation_hash
from .value_dataset import build


def split(cases,parent):
    fresh=[c for c in cases if c['batch']=='balanced_proxy_01']
    if not fresh:return set(parent['calibration_scene_seeds'])&{c['seed'] for c in cases}
    selected=set()
    for family in sorted({c['family'] for c in fresh}):
        rows=[c for c in fresh if c['family']==family]
        key=lambda c:hashlib.sha256(('operation-value-v15:'+str(c['seed'])).encode()).hexdigest()
        if any(c['training_target_station'] is not None for c in rows):
            for station in range(4):
                stratum=sorted([c for c in rows if c['training_target_station']==station],key=key)
                if stratum:selected.add(stratum[0]['seed'])
        else:selected.update(c['seed'] for c in sorted(rows,key=key)[:max(2,len(rows)//4)])
    return selected


def matrix(cases,kind):
    x=[];y=[];weights=[];groups=[];counts=Counter(c['family'] for c in cases)
    known=kind.startswith('known_');dynamic=kind.endswith('dynamic')
    for case in cases:
        start=len(x)
        # At a given clock time all alive cases retain their prior mass. Do
        # NOT divide by episode length: doing that changes the pre-task prior
        # in favor of short tasks before any informative motion is observed.
        prior=1./counts[case['family']]
        for sample in case['samples']:
            values,_=action_features(np.array(sample['x' if dynamic else 'static']),case['perception'],case['actions'],case['known'] if known else None)
            x.extend(values);y.extend(sample['y']);weights.extend([prior/len(case['actions'])]*len(values))
        groups.append((case,start,len(x)))
    return np.array(x,dtype=np.float32),np.array(y),np.array(weights),groups


def assess(prediction,y,groups):
    errors=[];selected=[];best=[]
    for case,start,end in groups:
        count=len(case['actions']);p=prediction[start:end].reshape(-1,count);truth=y[start:end].reshape(-1,count)
        errors.append(float(np.mean((p-truth)**2)))
        selected.append(float(np.mean(truth[np.arange(len(p)),p.argmax(axis=1)])))
        best.append(float(np.mean(truth.max(axis=1))))
    return dict(mse=float(np.mean(errors)),mean_selected_counterfactual_value=float(np.mean(selected)),
        mean_clairvoyant_counterfactual_value=float(np.mean(best)),regret=float(np.mean(np.array(best)-selected)),
        independent_scenes=len(groups),not_actual_attack_success=True)


def fit_one(training,validation,kind,configuration=None):
    x,y,w,_=matrix(training,kind);vx,vy,_,vg=matrix(validation,kind);choices=[]
    configurations=[configuration] if configuration else [dict(max_depth=d,min_samples_leaf=l) for d,l in ((6,20),(10,20),(10,80))]
    for conf in configurations:
        estimator=ExtraTreesRegressor(n_estimators=96,n_jobs=2,random_state=1515,max_features=.8,**conf)
        estimator.fit(x,y,sample_weight=w);prediction=estimator.predict(vx);metrics=assess(prediction,vy,vg)
        portable=export(estimator);delta=float(np.max(np.abs(prediction-predict(portable,vx))))
        if delta>1e-12:raise RuntimeError('Portable tree export differs from actual fitted predictions')
        metrics['feature_importance']=estimator.feature_importances_.tolist()
        choices.append((metrics['regret'],metrics['mse'],portable,conf,metrics,delta))
        print('FITTED',kind,conf,{k:v for k,v in metrics.items() if k!='feature_importance'},'portable_max_delta',delta,flush=True)
    best=min(choices,key=lambda item:(item[0],item[1]));return best[2],best[3],best[4]


def feature_ablations(training,validation,configuration):
    x,y,w,_=matrix(training,'dynamic');vx,vy,_,groups=matrix(validation,'dynamic');result={}
    schemas={'without_orientation':[i for i in range(x.shape[1]) if not 29<=i<55 and not 71<=i<75],
        'without_semantic_motion_context':[i for i in range(x.shape[1]) if not 61<=i<71]}
    for name,columns in schemas.items():
        estimator=ExtraTreesRegressor(n_estimators=96,n_jobs=2,random_state=1515,max_features=.8,**configuration)
        estimator.fit(x[:,columns],y,sample_weight=w);result[name]=dict(columns=columns,metrics=assess(estimator.predict(vx[:,columns]),vy,groups))
        print('FEATURE ABLATION',name,result[name]['metrics'],flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--batches',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise ValueError('Model version already exists; preserve it and choose a new version')
    feature_files=(Path('experiments/task_attack_value/vision.py'),Path('experiments/task_attack_value/value_features.py'))
    watched=feature_files+(Path('experiments/task_attack_value/forest.py'),Path(__file__),Path(__file__).with_name('value_dataset.py'),Path(__file__).with_name('prepare_fixed_slot_dataset.py'))
    frozen_source={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in watched}
    snapshot=a.work/'source_snapshot'
    if snapshot.exists():
        old=json.loads((snapshot/'snapshot_manifest.json').read_text())
        if old['implementation_hash']!=implementation_hash():raise ValueError('Existing training snapshot belongs to another implementation; choose a new work version')
    else:snapshot_implementation(snapshot)
    cache=a.work/'training_values.json.gz'
    if cache.exists():
        with gzip.open(cache,'rt',encoding='utf-8') as stream:data=json.load(stream)
        expected=[hashlib.sha256((b/'manifest.json').read_bytes()).hexdigest() for b in a.batches]
        if expected!=data['input_manifest_sha256']:raise RuntimeError('Dataset manifest changed; choose a new versioned cache')
        if data.get('image_feature_source_sha256')!=hashlib.sha256(feature_files[0].read_bytes()).hexdigest():raise RuntimeError('Image feature schema differs from cached data; choose a new versioned cache')
    else:data=build(a.batches,a.work)
    cases=data['cases'];parent=json.loads(Path('models/task_attack/forward_v14/bundle.json').read_text());calibration=split(cases,parent)
    training=[c for c in cases if c['seed'] not in calibration];validation=[c for c in cases if c['seed'] in calibration]
    models={};configurations={};metrics={};kinds=('static','dynamic','known_static','known_dynamic')
    for kind in kinds:models[kind],configurations[kind],metrics[kind]=fit_one(training,validation,kind)
    vx,vy,_,vg=matrix(validation,'dynamic');sx,_,_,_=matrix(validation,'static')
    dynamic_prediction=predict(models['dynamic'],vx);static_prediction=predict(models['static'],sx)
    probability=np.concatenate([np.repeat(s['informative_probability'],len(c['actions'])) for c in validation for s in c['samples']])
    blends=[]
    for alpha in (0.,.5,1.):
        weight=(1-alpha)+alpha*probability
        scores=assess((1-weight)*static_prediction+weight*dynamic_prediction,vy,vg)
        blends.append(dict(alpha=alpha,metrics=scores))
    blend=min(blends,key=lambda b:(b['metrics']['regret'],b['metrics']['mse']))
    ablations=feature_ablations(training,validation,configurations['dynamic'])
    # Three world-disjoint fold models evaluate the possible future branches.
    fold={c['seed']:int(hashlib.sha256(('operation-value-fold:'+str(c['seed'])).encode()).hexdigest(),16)%3 for c in training};crossfit={}
    for index in range(3):
        tr=[c for c in training if fold[c['seed']]!=index];held=[c for c in training if fold[c['seed']]==index]
        crossfit[str(index)]={}
        for kind in kinds:crossfit[str(index)][kind]=fit_one(tr,held,kind,configurations[kind])[0]
    library=[];descriptors=[];training_counts=Counter(c['family'] for c in training)
    for case in training:
        samples=[];description=[];static_description=[]
        for sample in case['samples']:
            x=np.array(sample['x']);description.append(np.concatenate([source_features(x,case['perception'],i) for i in range(len(x))]).tolist())
            static=np.array(sample['static']);static_description.append(np.concatenate([source_features(static,case['perception'],i) for i in range(len(static))]).tolist())
            samples.append({k:sample[k] for k in ('t','x','static','informative_probability')})
        library.append(dict(seed=case['seed'],fold=fold[case['seed']],prior_weight=1./training_counts[case['family']],
            authorized_program=case['known'][0],authorized_target=case['known'][1],samples=samples,descriptor=description,static_descriptor=static_description));descriptors.extend(description)
    descriptors=np.array(descriptors);model=dict(version=19,value_models=models,crossfit_value_models=crossfit,
        trajectory_library=library,metric_scale=np.maximum(1e-4,descriptors.std(axis=0)).tolist(),
        informative_prefix=parent['informative_prefix'],legacy_onset=parent['legacy_onset'],programs=parent['programs'],
        parameters=parent['parameters'],endpoint=parent['endpoint'],calibration=metrics,selected_configurations=configurations,
        offline_feature_ablations=ablations,
        training_scene_seeds=sorted(c['seed'] for c in training),calibration_scene_seeds=sorted(calibration),
        all_development_scene_seeds=sorted(set(parent['all_development_scene_seeds'])|{c['seed'] for c in cases}|{e['seed'] for e in data['excluded']}|
            {row['seed'] for batch in a.batches for row in json.loads((batch/'manifest.json').read_text())['rows']}),
        use_dynamic_value=blend['metrics']['regret']<metrics['static']['regret'],
        information_blend=blend['alpha'],information_blend_calibration=blends,
        use_known_dynamic_value=metrics['known_dynamic']['regret']<metrics['known_static']['regret'],
        feature_source_sha256={str(f):frozen_source[str(f)] for f in feature_files},
        training_source_sha256=frozen_source,
        trained_operation_model=True,whole_task_classification_required=False,current_target_truth_used=False,
        external_rgb_and_public_clock_only_at_deployment=True,actual_endpoint_calibrated=True,
        input_manifest_sha256=data['input_manifest_sha256'],source_result_sha256=[c['source_result_sha256'] for c in cases],
        training_quality_exclusions=data['excluded'],
        training_target=data['target'],effect_label_dispatch_delay_seconds=data['effect_label_dispatch_delay_seconds'],
        effect_label_implementation_delay_seconds=data['effect_label_implementation_delay_seconds'],
        external_dataset_constructor_sha256=data['external_dataset_constructor_sha256'],full_paper_fine_grained_KDE_reproduction=False,
        limitations='Counterfactual labels use OWN whole-query semantics and clean future physical read opportunities with a small empirical adoption transition. They are not actual attack outcomes for every action; victim policy changes/recovery require fresh online validation. Paper baselines remain adaptations.')
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha for path,sha in frozen_source.items()):raise RuntimeError('Training source changed mid-fit; preserve logs but do not publish mismatched model')
    write_json(a.output,model);print('MODEL SAVED',str(a.output),'behavior enabled',model['use_dynamic_value'],flush=True)


if __name__=='__main__':main()
