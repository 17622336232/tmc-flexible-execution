"""Declared local role inputs to a LEARNED operator, never a result filter.

The goal is a latent hypothesis at deployment. Only the task-known reference
receives real text. This encoder has no task name, physical truth, clock or
private target state. Relational inputs are computed from declared bindings.
"""
from operator_features import FIELDS,numeric,compare
from local_operator_kernels import latest


def register_inputs(program,target,source,bindings,perception,parameters):
    values={b['field']:b['declared_value'] for b in bindings if b['object_source']==source and b['status']=='certain' and b['declared_value'] is not None}
    roles=set(program['roles']);reduction=program.get('reduce') or {}
    relation,known=compare(program.get('filter'),values,parameters)
    x=[float(f in roles) for f in FIELDS]
    x.extend([float(program['scope']!='target' or target==source),relation,known,
        float(reduction.get('op')=='min'),float(reduction.get('op')=='max')])
    for field in FIELDS:
        value,is_number=numeric(values.get(field),field)
        all_values=[latest(bindings,i,field,True) for i in range(4)]
        numbers=[numeric(b['declared_value'],field)[0] for b in all_values if b and numeric(b['declared_value'],field)[1]]
        x.extend([float(field in values),value,is_number,
            value-min(numbers) if is_number and numbers else 0.,value-max(numbers) if is_number and numbers else 0.,
            sum(b is not None for b in all_values)/4.,
            float(any(b['object_source']==source and b['field']==field and b['status']=='conflict' for b in bindings)),
            float(reduction.get('field')==field)])
        seen=perception['stations'][source].get(field)
        n,k=numeric(seen,field);x.extend([float(seen is not None),n,k])
    return x
