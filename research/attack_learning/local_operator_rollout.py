"""Use the SAME local kernels to generate candidate executions.

No output is corrected with the goal predicate. This is a model, not a robot
controller: generated body events never go to ROS or to a real vehicle.
"""
from dataclasses import replace
import math

import numpy as np

from primitive_state import CandidateState,Fact
from local_operator_kernels import ACTIVE,read_inputs,resolve,resolve_sequence
from build_local_operator_dataset import global_features,canonical,denormalize


def make_state(world):
    facts=tuple(Fact(i,k,v) for i,row in enumerate(world) for k,v in row.items())
    return CandidateState(facts,facts)


def quality_input(body,perception,source,field_index):
    a=perception['anchors'][source];width=max(.005,a['w'])
    return [(body[0]-a['u'])/width,(body[1]-a['v'])/width,body[2]/width,body[3]/width,*body[4:6],float(field_index)]


def capture(state,body,t,perception,models):
    xs=[quality_input(body,perception,i,j) for i in range(4) for j in range(2)]
    qs=np.clip(models['quality']['distribution'].mean_outcome(np.array(xs))[:,0],0,1)
    # Match the training state schema exactly. Other physical fields have an
    # explicitly UNMEASURED acquisition, not an invented zero-quality read and
    # not an absent capture counter that changes the learned phase inputs.
    for fact in state.carriers:
        q=float(qs[fact.source*2+ACTIVE.index(fact.field)]) if fact.field in ACTIVE else None
        state=state.capture(fact.source,fact.field,t,q)
    return state


def resolve_records(templates,state):
    bindings=[vars(b) for b in state.bindings];captures=[vars(c) for c in state.captures];records=[]
    for row in templates:
        source=row['source'];facts={}
        for f in row['fields']:
            value=resolve(f['alias'],source,f['field'],bindings,captures)
            facts[f['field']]=value
        records.append(dict(source=source,unknown_object=row.get('unknown_object'),facts=facts))
    return records


def interpret(state,features,models,rng):
    snapshots=[vars(c) for c in state.captures]
    x=read_inputs(features,snapshots)[None]
    index=int(models['read_bind']['distribution'].sample(x,rng)[0])
    result=models['read_bind']['outcomes'][index]
    previous=[vars(b) for b in state.bindings]
    for slot,outputs in enumerate(result['updates']):
        source=slot//2;field=ACTIVE[slot%2]
        for out in outputs:
            value=resolve(out['alias'],source,field,previous,snapshots)
            sequence=resolve_sequence(out['alias'],source,field,previous,snapshots)
            if sequence is not None:state=state.bind(source,field,value,sequence,out['status'])
    for row in resolve_records(result['record_writes'],state):
        if row['source'] is None:continue
        for field,value in row['facts'].items():
            seq=[c.sequence for c in state.captures if c.source==row['source'] and c.field==field]
            if seq:state=state.register(row['source'],field,value,seq[-1])
    return state,index


def choose(state,program,target,body,t,perception,parameters,models,rng):
    features=global_features(program,target,state,perception,body,parameters,t)
    index=int(models['decide']['distribution'].sample(np.array([features]),rng)[0])
    return models['decide']['outcomes'][index],index


def submit(state,program,target,body,t,perception,parameters,models,rng):
    features=global_features(program,target,state,perception,body,parameters,t)
    index=int(models['submit']['distribution'].sample(np.array([features]),rng)[0])
    outcome=models['submit']['outcomes'][index]
    # Report selection belongs to the learned kernel, not to the independent
    # factual scorer. Contradictory selected records remain separate records.
    records=resolve_records(outcome['records'],state)
    return dict(records=records,accepted=outcome['accepted'],explicit_disclosure=outcome['disclosed'],outcome=outcome['outcome']),index


def factual_error(report,program,target,world,parameters):
    """Independent score AFTER the model has generated its report."""
    from experiments.task_attack_value.programs import answer
    from experiments.task_attack_value.semantic_fact_audit import numeric_representation
    if not report['accepted']:return False
    expected=dict(answer(program,world,parameters,target));collected={};wrong=False
    for record in report['records']:
        source=record['source']
        if source not in expected:wrong=True;continue
        for field,value in record['facts'].items():
            value=numeric_representation(field,value)
            if field in program['roles']:
                if value!=world[source][field]:wrong=True
                elif value is not None:collected.setdefault(source,set()).add(field)
    if set(expected)!=set(collected):wrong=True
    for source in expected:
        if not set(program['roles']).issubset(collected.get(source,set())):wrong=True
    return wrong


def rollout(models,program,target,perception,world,first_body,parameters,rng,budget=900.,
            edit=None,edit_time=None,initial_state=None,initial_t=0.,max_operations=120):
    state=initial_state or make_state(world);body=list(first_body);t=float(initial_t);path=[];edit_applied=False
    # This cap is a model-computation censor only, never a new victim rule.
    for iteration in range(max_operations):
        if edit is not None and not edit_applied and edit_time is not None and t>=edit_time:
            state=state.dispatch_edit(edit['source'],edit['role'],edit['value']);edit_applied=True
        if t>=budget:break
        state=capture(state,body,t,perception,models)
        features=global_features(program,target,state,perception,body,parameters,t)
        state,read_index=interpret(state,features,models,rng)
        decision,index=choose(state,program,target,body,t,perception,parameters,models,rng)
        duration=float(rng.choice(models['completed_read_latencies']))
        path.append(dict(t=t,event=list(body),read_outcome=read_index,decision_outcome=index,terminal=decision['terminal']))
        # The current capture has been latched. An edit during processing can
        # update carriers, but it cannot replace this call's captured content.
        if edit is not None and not edit_applied and edit_time is not None and t<edit_time<=t+duration:
            state=state.dispatch_edit(edit['source'],edit['role'],edit['value']);edit_applied=True
        t+=duration
        if decision['terminal']:
            if t>=budget:break
            report,report_index=submit(state,program,target,body,t,perception,parameters,models,rng)
            return dict(report=report,t=t,path=path,state=state,terminal=True,report_outcome=report_index,
                factual_error=factual_error(report,program,target,world,parameters))
        travel=float(decision['travel_seconds'])
        # The learned outcome is a destination in static-image coordinates;
        # no ground-truth world pose or native trajectory is queried here.
        body=denormalize(decision['endpoint'],perception);t+=travel
    return dict(report=dict(records=[],accepted=False,explicit_disclosure=None,outcome='model_censored'),
                t=t,path=path,state=state,terminal=False,factual_error=False)
