"""Admit a grouped work clause only when its plausible scopes are equivalent."""
from itertools import product

from .natural_frontier import ASK, discover, read_ordinary_records


def residual(plan, facts):
    signature=[]
    for field in plan.fields:
        remaining={tuple(i for i in clause if i not in facts)
                   for clause in field.any_of if not any(facts.get(i) is False for i in clause)}
        signature.append((field.name,field.view,((),) if () in remaining else tuple(sorted(remaining))))
    return (plan.direction,plan.angle_degrees,plan.finish,plan.early_capture,tuple(signature))


def resolve_grouped_conditions(clauses, reply, sources):
    """Require equivalent remaining permissions and obligations under both scopes.

    A returned finding can resolve a scope ambiguity only if it has the same
    actual image and prerequisite status under every candidate interpretation.
    Ambiguities outside this bounded conditional-clause grammar stay native.
    """
    alternatives=[]
    ambiguous=0
    for clause in clauses:
        pieces=clause.split("，")
        if len(pieces)>2 and pieces[0].startswith(("若","如果")):
            condition,first,*following=pieces
            if not all(ASK.match(s) for s in following):
                return None
            ambiguous+=1
            first=condition+"，"+first
            independent="。".join([first,*following])
            inherited="。".join([first,*(condition+"，"+s for s in following)])
            alternatives.append((independent,inherited))
        else:
            alternatives.append((clause,))
    if not 0<ambiguous<=2:
        return None
    common=None
    selected=None
    for ordered in product(*alternatives):
        plan=discover("。".join(ordered))
        if plan is None:
            return None
        observed=read_ordinary_records(plan,reply,{},sources)
        signature=(observed,residual(plan,observed))
        if common is not None and common!=signature:
            return None
        common=signature
        # Keep the more restrictive inherited form when its residual is identical.
        selected=plan
    return selected
