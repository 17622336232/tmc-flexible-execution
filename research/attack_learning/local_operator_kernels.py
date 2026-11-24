"""Learned conditional distributions over LOCAL outcomes, without task IDs.

Forest leaves retain joint empirical outcomes rather than rounding a predicted
mean into a rule. A sampled READ outcome may retain an old binding, be uncertain
or bind a wrong value. A sampled SUBMIT may be wrong, partial or refused. The
authorized predicate is never used to repair generated records.
"""
import math

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from operator_features import FIELDS,local_features

ACTIVE=('cracked','temperature')
READ_CODES=('unmentioned','retain','capture','other','uncertain','conflict','unreadable')


def latest(bindings,source,field,certain_only=False):
    for b in reversed(bindings):
        if b['object_source']==source and b['field']==field and (not certain_only or b['status']=='certain'):
            return b
    return None


def read_alias(value,source,field,bindings,captures):
    old=latest(bindings,source,field,True)
    if old is not None and value==old['declared_value']:return dict(kind='previous')
    compatible=[(i,c) for i,c in enumerate(captures) if c['source']==source and c['field']==field and c['carrier_value']==value]
    if compatible:
        i,c=compatible[-1]
        age=sum(x['source']==source and x['field']==field for x in captures[i+1:])
        return dict(kind='capture',age=age)
    # An erroneous declaration remains erroneous; do not replace with truth.
    return dict(kind='literal',value=value)


def resolve(alias,source,field,bindings,captures):
    if alias['kind']=='previous':
        b=latest(bindings,source,field,True)
        return b['declared_value'] if b else None
    if alias['kind']=='capture':
        rows=[c for c in captures if c['source']==source and c['field']==field]
        age=alias['age']
        return rows[-1-age]['carrier_value'] if len(rows)>age else None
    return alias.get('value')


def resolve_sequence(alias,source,field,bindings,captures):
    if alias['kind']=='previous':
        b=latest(bindings,source,field,True)
        return b['capture_sequence'] if b else None
    sequences=[i for i,c in enumerate(captures) if c['source']==source and c['field']==field]
    age=alias.get('age',0) if alias['kind']=='capture' else 0
    return sequences[-1-age] if len(sequences)>age else None


def read_template(example):
    templates=[];codes=[]
    for source in range(4):
        for field in ACTIVE:
            facts=[f for f in example['declared'] if f['source']==source and f['field']==field]
            outputs=[]
            for f in facts:
                value=f['value'];status=f['status']
                alias=read_alias(value,source,field,example['before_bindings'],example['captures'])
                outputs.append(dict(status=status,alias=alias))
            templates.append(outputs)
            if not outputs:code=0
            elif any(o['status']=='conflict' for o in outputs):code=5
            elif all(o['status']=='unreadable' for o in outputs):code=6
            elif not any(o['status']=='certain' for o in outputs):code=4
            else:
                kinds=[o['alias']['kind'] for o in outputs if o['status']=='certain']
                code=1 if all(k=='previous' for k in kinds) else 2 if all(k in ('previous','capture') for k in kinds) else 3
            codes.append(code)
    return templates,np.eye(len(READ_CODES))[codes].ravel()


def read_inputs(features,captures):
    """Only the READ head can access latched content, with its visibility.

    At zero acquisition quality, changing raw contents cannot change this
    extra READ input. DECIDE never receives these extra content features.
    """
    values=[]
    for source in range(4):
        for field in ACTIVE:
            rows=[c for c in captures if c['source']==source and c['field']==field]
            rows=rows[-2:]
            for age in (0,1):
                c=rows[-1-age] if len(rows)>age else None
                if c and c['quality_measured']:
                    q=c['view_quality'];v=c['carrier_value']
                    is_numeric=isinstance(v,(bool,int,float))
                    numeric=float(v) if is_numeric else 0.
                    values.extend([q,q*numeric/(100. if field=='temperature' else 1.),float(is_numeric)])
                else:values.extend([0.,0.,0.])
    return np.r_[features,values]


class ConditionalDistribution:
    """A fitted forest partition and weighted empirical joint leaf law."""
    def __init__(self,trees=48,leaf=4,depth=14,seed=1611):
        self.forest=ExtraTreesRegressor(n_estimators=trees,min_samples_leaf=leaf,
            max_depth=depth,max_features=.75,n_jobs=2,random_state=seed)

    def fit(self,x,y,weights):
        self.mean=np.mean(y,axis=0);self.scale=np.maximum(.15,np.std(y,axis=0))
        self.forest.fit(x,(y-self.mean)/self.scale,sample_weight=weights)
        self.weights=np.asarray(weights);leaves=self.forest.apply(x);self.tables=[]
        for j in range(leaves.shape[1]):
            table={}
            for leaf in np.unique(leaves[:,j]):
                ids=np.flatnonzero(leaves[:,j]==leaf);p=self.weights[ids];table[int(leaf)]=(ids,np.cumsum(p/p.sum()))
            self.tables.append(table)
        return self

    def mean_outcome(self,x):
        # sklearn squeezes a one-output regressor. Preserve the outcome axis;
        # otherwise an (N,) quality vector broadcasts against (N,1) labels.
        return np.asarray(self.forest.predict(x)).reshape(len(x),len(self.mean))*self.scale+self.mean

    def probabilities(self,x):
        """Exact empirical mixture law used by sample(), without draw noise.

        Marginalize the uniformly selected forest tree and its weighted leaf
        outcomes. This does not replace or alter any fitted/sampled outcome.
        It permits a proper mixture likelihood without8-draw log-density bias.
        """
        x=np.asarray(x,dtype=np.float32);leaves=self.forest.apply(x);result=[]
        for row in leaves:
            probabilities=np.zeros(len(self.weights))
            for j,leaf in enumerate(row):
                ids,cdf=self.tables[j][int(leaf)];probabilities[ids]+=np.diff(cdf,prepend=0.)/len(self.tables)
            ids=np.flatnonzero(probabilities>0)
            result.append((ids,probabilities[ids]))
        return result

    def sample(self,x,rng,trees=None):
        x=np.asarray(x,dtype=np.float32);out=np.empty(len(x),dtype=int)
        choices=rng.randint(len(self.tables),size=len(x)) if trees is None else np.asarray(trees)
        for j in np.unique(choices):
            rows=np.flatnonzero(choices==j);leaves=self.forest.estimators_[j].apply(x[rows])
            for leaf in np.unique(leaves):
                selected=rows[leaves==leaf];ids,cdf=self.tables[j][int(leaf)]
                out[selected]=ids[np.minimum(len(ids)-1,np.searchsorted(cdf,rng.random_sample(len(selected))))]
        return out


def report_template(example,episode):
    identities={s['identity']:i for i,s in enumerate(episode['perception']['stations'])}
    records=[];targets=[];report=example['report'];bindings=example['after_bindings']
    for record in report.get('records') or []:
        source=identities.get(record.get('object_id'))
        # Preserve unknown object claims as such, never silently rebind them.
        fields=[]
        for field,value in (record.get('facts') or {}).items():
            if field not in ACTIVE:continue
            alias=read_alias(value,source,field,bindings,example['captures']) if source is not None else dict(kind='literal',value=value)
            fields.append(dict(field=field,alias=alias))
        records.append(dict(source=source,unknown_object=record.get('object_id') if source is None else None,fields=fields))
    for source in range(4):
        for field in ACTIVE:
            targets.append(float(any(r['source']==source and any(f['field']==field for f in r['fields']) for r in records)))
    accepted=report.get('outcome')=='accepted'
    # Disclosure annotations are separate human-reviewed labels where present;
    # no keyword or absence inference substitutes for model awareness.
    disclosed=example.get('explicit_conflict_disclosure_label')
    targets.append(float(accepted))
    return dict(records=records,accepted=accepted,disclosed=disclosed,
        outcome=report.get('outcome')),np.array(targets)


def teacher_matrices(episodes):
    choices=[];reads=[];reports=[];quality=[]
    family_count={f:sum(e['family']==f for e in episodes) for f in {e['family'] for e in episodes}}
    for episode in episodes:
        examples=episode['examples'];weight=1./family_count[episode['family']]/len(examples)
        for ex in examples:
            updates,encoded=read_template(ex)
            register_example=dict(ex,report=dict(outcome='incomplete',records=ex['records_returned']))
            writes,write_targets=report_template(register_example,episode)
            reads.append(dict(x=read_inputs(ex['before_features'],ex['captures']),y=np.r_[encoded,write_targets[:-1]],
                outcome=dict(updates=updates,record_writes=writes['records']),weight=weight,episode=episode['key'],step=ex['step']))
            if not ex['censored']:
                endpoint=ex['next_canonical_event'] if ex['next_canonical_event'] is not None else ex['canonical_event']
                interval=ex['next_t']-ex['t'] if ex['next_t'] is not None else ex['call_seconds']
                travel=max(0.,interval-ex['call_seconds'])
                kind=0 if ex['terminal'] else 1 if np.linalg.norm(np.array(endpoint)[:4]-ex['canonical_event'][:4])>.01 else 2
                target=np.r_[np.eye(3)[kind]*2,endpoint]
                choices.append(dict(x=ex['after_features'],y=target,weight=weight,
                    outcome=dict(terminal=ex['terminal'],endpoint=endpoint,
                        origin=ex['canonical_event'],travel_seconds=travel,
                        tool=ex['action_type'],call_seconds=ex['call_seconds']),
                    episode=episode['key'],step=ex['step']))
            if ex['terminal']:
                outcome,target=report_template(ex,episode)
                reports.append(dict(x=ex['after_features'],y=target,outcome=outcome,
                    weight=weight*len(examples),episode=episode['key'],step=ex['step']))
            width=len(ex['before_features'])//4
            for source in range(4):
                for field_index,field in enumerate(ACTIVE):
                    q=ex['acquisition_quality'][source*2+field_index]
                    if q is not None:
                        local=ex['before_features'][width*source:width*(source+1)]
                        # Quality depends on physical image/view, not goals or
                        # remembered semantic contents. Use event and actual
                        # static source geometry only.
                        anchor=episode['perception']['anchors'][source];ev=ex['event'];scale=max(.005,anchor['w'])
                        x=[(ev[0]-anchor['u'])/scale,(ev[1]-anchor['v'])/scale,
                            ev[2]/scale,ev[3]/scale,*ev[4:6],float(field_index)]
                        quality.append(dict(x=x,y=[q],weight=weight/8,episode=episode['key'],step=ex['step']))
    return dict(decide=choices,read_bind=reads,submit=reports,quality=quality)


def fit_kernels(episodes,configuration=None):
    config=configuration or dict(trees=48,leaf=4,depth=14)
    data=teacher_matrices(episodes);models={}
    for key,rows in data.items():
        if not rows:raise ValueError('No training support for '+key)
        fit=ConditionalDistribution(**config,seed=1611+len(key))
        fit.fit(np.array([r['x'] for r in rows],dtype=np.float32),
                np.array([r['y'] for r in rows]),np.array([r['weight'] for r in rows]))
        models[key]=dict(distribution=fit,outcomes=[r.get('outcome',r['y']) for r in rows])
    # Actual completed READ latencies are pooled. Goal-specific complete-route
    # duration is not allowed to manufacture a likelihood ratio by itself.
    durations=[ex['call_seconds'] for ep in episodes for ex in ep['examples'] if ex['call_seconds'] is not None and ex['call_seconds']>0]
    models['completed_read_latencies']=np.array(durations)
    models['limitations']=['Empirical local conditional distributions, not a claim of a complete proposal reproduction.',
        'READ/BIND adoption under cache conflict needs edited-proxy support; clean data alone cannot identify it.',
        'Only completed READ durations are currently sampled. Censor-aware duration calibration must pass before deployment.',
        'Submit disclosure remains unlabelled unless separately reviewed; unlabelled is not false.']
    return models,data


def validation_metrics(models,episodes):
    matrices=teacher_matrices(episodes);result={}
    for key,rows in matrices.items():
        if not rows:result[key]=dict(examples=0);continue
        x=np.array([r['x'] for r in rows],dtype=np.float32);y=np.array([r['y'] for r in rows]);weights=np.array([r['weight'] for r in rows]);weights/=weights.sum()
        prediction=models[key]['distribution'].mean_outcome(x)
        result[key]=dict(examples=len(rows),weighted_squared_error=float(np.sum(weights[:,None]*(prediction-y)**2)/y.shape[1]))
        if key=='decide':
            result[key]['terminal_brier']=float(weights.dot((prediction[:,0]/2-y[:,0]/2)**2))
            result[key]['next_event_absolute_error']=np.sum(weights[:,None]*np.abs(prediction[:,3:]-y[:,3:]),axis=0).tolist()
        elif key=='read_bind':
            dimensions=8*len(READ_CODES)
            p=prediction[:,:dimensions].reshape(-1,8,len(READ_CODES));truth=y[:,:dimensions].reshape(p.shape)
            result[key]['declared_transition_accuracy']=float(np.sum(weights[:,None]*(p.argmax(axis=-1)==truth.argmax(axis=-1)))/8)
            mask=truth[:,:,0]==0
            result[key]['declared_transition_accuracy_excluding_unmentioned']=float(np.sum(weights[:,None]*mask*(p.argmax(axis=-1)==truth.argmax(axis=-1)))/max(1e-12,np.sum(weights[:,None]*mask)))
    return result
