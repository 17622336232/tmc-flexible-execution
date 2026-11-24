"""Source-relative learned K/D/E components, with no task-family feature.

Every head is shared across physical sources. Actor choices and reports remain
labels; neither a prescribed route nor an answer filter is installed in the
victim. This replaces joint whole-row output stencils in the first local pilot.
"""
import numpy as np

from operator_features import FIELDS
from local_operator_kernels import ConditionalDistribution,ACTIVE,READ_CODES,read_template,report_template,latest
from local_operator_kernels import read_inputs,resolve,resolve_sequence
from build_local_operator_dataset import canonical,global_features
from local_operator_rollout import make_state,capture,quality_input,factual_error


def relative_features(global_vector,source,bindings):
    width=len(global_vector)//4;x=list(global_vector[width*source:width*(source+1)])
    # Generic relations between DECLARED numeric facts. This is an input to
    # a learned choice, not a predicate that filters submitted records.
    for field in FIELDS:
        own=latest(bindings,source,field,True)
        values=[latest(bindings,i,field,True) for i in range(4)]
        nums=[b['declared_value'] for b in values if b and isinstance(b['declared_value'],(bool,int,float))]
        if own and isinstance(own['declared_value'],(bool,int,float)) and nums:
            unit=100. if field=='temperature' else 10. if field=='box_count' else 1.
            v=own['declared_value'];x.extend([(v-min(nums))/unit,(v-max(nums))/unit,1.])
        else:x.extend([0.,0.,0.])
    return x


def read_relative_inputs(local,captures,source):
    # Reindex only this operation's source port; raw values are exclusive to
    # this READ head and become invisible at zero measured acquisition quality.
    rows=[dict(c,source=0) for c in captures if c['source']==source and c['field'] in ACTIVE]
    return read_inputs(local,rows)


def local_records(records,source):
    return [dict(r,source=0) for r in records if r['source']==source]


def fit(episodes,config=None):
    config=config or dict(trees=48,leaf=4,depth=14)
    datasets={k:[] for k in ('read','gate','motion','register','accept')}
    family_count={f:sum(e['family']==f for e in episodes) for f in {e['family'] for e in episodes}}
    for ep in episodes:
        examples=ep['examples'];anchors=ep['perception']['anchors']
        for j,ex in enumerate(examples):
            weight=1./family_count[ep['family']]/len(examples)
            updates,codes=read_template(ex);codes=codes.reshape(8,len(READ_CODES))
            writes,write_y=report_template(dict(ex,report=dict(records=ex['records_returned'],outcome='incomplete')),ep)
            after_bindings=ex['after_bindings'];before_bindings=ex['before_bindings']
            if not ex['terminal'] and ex['next_canonical_event'] is not None:
                nxt=examples[j+1] if j+1<len(examples) and examples[j+1]['t']==ex['next_t'] else None
                quality=np.array(nxt['acquisition_quality']).reshape(4,2).max(axis=1) if nxt else np.zeros(4)
                if quality.max()>0:destination=int(quality.argmax());basis='actual next acquisition opportunity'
                else:
                    # This associates a geometry sample with a coordinate
                    # anchor; it is not a proposed robot destination rule.
                    u=ex['next_event'][0];destination=int(np.argmin([abs(a['u']-u) for a in anchors]));basis='nearest image anchor; source purpose unknown'
            else:destination=None;basis=None
            for source in range(4):
                before=relative_features(ex['before_features'],source,before_bindings)
                after=relative_features(ex['after_features'],source,after_bindings)
                row=dict(x=read_relative_inputs(before,ex['captures'],source),
                    y=np.r_[codes[source*2:source*2+2].ravel(),write_y[source*2:source*2+2]],weight=weight/4,
                    outcome=dict(updates=updates[source*2:source*2+2],record_writes=local_records(writes['records'],source)))
                datasets['read'].append(row)
                if not ex['censored']:
                    gate=0 if ex['terminal'] else 1 if source==destination else 2
                    datasets['gate'].append(dict(x=after,y=np.eye(3)[gate],weight=weight/4,outcome=gate))
                if source==destination:
                    endpoint=np.array(ex['next_canonical_event']);anchor=canonical([anchors[source]['u'],anchors[source]['v'],0,0,0,0],ep['perception'])
                    endpoint[:2]-=anchor[:2]
                    travel=max(0.,ex['next_t']-ex['t']-ex['call_seconds'])
                    datasets['motion'].append(dict(x=after,y=endpoint,weight=weight,
                        outcome=dict(relative_endpoint=endpoint.tolist(),travel_seconds=travel,source_assignment_basis=basis)))
                if ex['terminal']:
                    report,y=report_template(ex,ep)
                    datasets['register'].append(dict(x=after,y=y[source*2:source*2+2],weight=weight*len(examples)/4,
                        outcome=local_records(report['records'],source)))
            if ex['terminal']:
                report,y=report_template(ex,ep)
                datasets['accept'].append(dict(x=ex['after_features'],y=y[-1:],weight=weight*len(examples),
                    outcome=dict(accepted=report['accepted'],disclosed=report['disclosed'],outcome=report['outcome'])))
    models={}
    for key,rows in datasets.items():
        distribution=ConditionalDistribution(**config,seed=1600+len(key))
        distribution.fit(np.array([r['x'] for r in rows],dtype=np.float32),np.array([r['y'] for r in rows]),np.array([r['weight'] for r in rows]))
        models[key]=dict(distribution=distribution,outcomes=[r['outcome'] for r in rows])
    models['completed_read_latencies']=np.array([ex['call_seconds'] for ep in episodes for ex in ep['examples'] if ex['call_seconds'] and ex['call_seconds']>0])
    models['limitations']=['Source-relative empirical kernels; edited-cache support and censor-aware duration still need validation before deployment.',
        'Nearest-anchor motion labels with zero measured next-field opportunity do not establish semantic purpose.',
        'Independent local factors may miss correlated errors; complete-trajectory validation is required.']
    return models,datasets


def sample_reads(state,features,models,rng):
    previous=[vars(b) for b in state.bindings];captures=[vars(c) for c in state.captures]
    xs=[read_relative_inputs(relative_features(features,i,previous),captures,i) for i in range(4)]
    indices=models['read']['distribution'].sample(np.array(xs),rng)
    for source,index in enumerate(indices):
        outcome=models['read']['outcomes'][index]
        for j,outputs in enumerate(outcome['updates']):
            field=ACTIVE[j]
            for out in outputs:
                value=resolve(out['alias'],source,field,previous,captures)
                sequence=resolve_sequence(out['alias'],source,field,previous,captures)
                if sequence is not None:state=state.bind(source,field,value,sequence,out['status'])
        for row in outcome['record_writes']:
            for f in row['fields']:
                value=resolve(f['alias'],source,f['field'],[vars(b) for b in state.bindings],captures)
                sequence=resolve_sequence(f['alias'],source,f['field'],[vars(b) for b in state.bindings],captures)
                if sequence is not None:state=state.register(source,f['field'],value,sequence)
    return state


def decide(state,features,perception,models,rng):
    bindings=[vars(b) for b in state.bindings]
    xs=np.array([relative_features(features,i,bindings) for i in range(4)])
    probabilities=np.clip(models['gate']['distribution'].mean_outcome(xs),0,1)
    terminal=float(np.mean(probabilities[:,0]))
    if rng.rand()<terminal:return dict(terminal=True)
    weights=probabilities[:,1]+1e-12;weights/=weights.sum();source=int(rng.choice(4,p=weights))
    index=int(models['motion']['distribution'].sample(xs[source:source+1],rng)[0])
    outcome=models['motion']['outcomes'][index];endpoint=np.array(outcome['relative_endpoint'],copy=True)
    a=perception['anchors'][source];anchor=canonical([a['u'],a['v'],0,0,0,0],perception);endpoint[:2]+=anchor[:2]
    return dict(terminal=False,endpoint=endpoint.tolist(),travel_seconds=outcome['travel_seconds'])


def submit(state,features,models,rng):
    bindings=[vars(b) for b in state.bindings];captures=[vars(c) for c in state.captures]
    xs=np.array([relative_features(features,i,bindings) for i in range(4)])
    ids=models['register']['distribution'].sample(xs,rng);records=[]
    for source,index in enumerate(ids):
        for template in models['register']['outcomes'][index]:
            facts={f['field']:resolve(f['alias'],source,f['field'],bindings,captures) for f in template['fields']}
            records.append(dict(source=source,facts=facts,unknown_object=None))
    k=int(models['accept']['distribution'].sample(np.array([features]),rng)[0]);outcome=models['accept']['outcomes'][k]
    return dict(records=records,accepted=outcome['accepted'],explicit_disclosure=outcome['disclosed'],outcome=outcome['outcome'])


def rollout(models,program,target,perception,world,first_body,parameters,rng,budget=900.,max_operations=120):
    from build_local_operator_dataset import denormalize
    state=make_state(world);body=list(first_body);t=0.;path=[]
    for _ in range(max_operations):
        if t>=budget:break
        state=capture(state,body,t,perception,models)
        features=global_features(program,target,state,perception,body,parameters,t)
        state=sample_reads(state,features,models,rng)
        features=global_features(program,target,state,perception,body,parameters,t)
        action=decide(state,features,perception,models,rng)
        duration=float(rng.choice(models['completed_read_latencies']));path.append(dict(t=t,event=body,terminal=action['terminal']));t+=duration
        if action['terminal']:
            if t>=budget:break
            report=submit(state,features,models,rng)
            return dict(report=report,t=t,path=path,state=state,terminal=True,factual_error=factual_error(report,program,target,world,parameters))
        body=denormalize(action['endpoint'],perception);t+=action['travel_seconds']
    return dict(report=dict(records=[],accepted=False,explicit_disclosure=None,outcome='model_censored'),t=t,path=path,state=state,terminal=False,factual_error=False)
