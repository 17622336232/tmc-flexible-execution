"""Validate cold text parsing against the off-controller scoring specifications."""
import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

from experiments.flexible_execution.revision_audit import read, save, write_table
from experiments.flexible_execution.text_requirements import compile_text
from experiments.flexible_execution.requirement_program import parse


def reference(truth, instruction):
    if truth.get("oracle_requirement_program"):
        return parse(truth["oracle_requirement_program"], instruction)
    spec = truth["score_spec"]
    fields = {}
    from experiments.flexible_execution.material_schema import INSPECTION_FIELDS
    for key in spec["required"]:
        fields[key] = INSPECTION_FIELDS[key]
    if set(spec)-{"required","target","predicate","greater","less_equal"}:
        raise ValueError("Unexpected reference operation on recognized input")
    predicates = [{"op":op,"field":k,"right":{"literal":v}} for key,op in
                  [("predicate","eq"),("greater","gt"),("less_equal","le")] for k,v in spec.get(key,{}).items()]
    rule = {"op":"and","children":predicates} if len(predicates)>1 else predicates[0] if predicates else {"op":"keep"}
    scope = {"mode":"all","description":"reference"} if "target" not in spec else {
        "mode":"target","description":"reference","identity":spec["target"]}
    return parse(dict(scope=scope,fields=fields,filter=rule,extreme=None,report=spec["required"]), instruction)


def exercise(program, ref):
    assert program.scope["mode"]==ref.scope["mode"] and program.scope.get("identity")==ref.scope.get("identity")
    assert program.fields==ref.fields and set(program.report)==set(ref.report)
    # Vary the observed facts independently, including partial evidence. Numeric
    # boundary values are checked explicitly; no unobserved value enters a run.
    domain = {"flag":[None,False,True],"number":[None,0,69.9,70.,70.1,71.],
              "count":[None,0,1,4],"label":[None,"red","blue"]}
    values = [domain[t] for t in program.fields.values()]
    names = [program.scope.get("identity","subject"),"other"]
    count=0
    for left,right in product(product(*values),repeat=2):
        records={n:dict(zip(program.fields,v)) for n,v in zip(names,(left,right))}
        for confirmed in (True,False):
            assert program.evaluate(records,scope_confirmed=confirmed)==ref.evaluate(records,scope_confirmed=confirmed)
            count+=1
    return count


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,default=Path.cwd())
    repo=parser.parse_args().repo
    out=repo/"outputs/revision-study/round3"
    out.mkdir(parents=True,exist_ok=True)
    sources=list((repo/"outputs/natural-pilot").glob("pair-*/paired-serial/attempt-*/episode/private/truth.json"))
    sources+=list((repo/"outputs/protocol-v2-optimization/custom-final").glob("custom-*/conditional-a/attempt-*/episode/private/truth.json"))
    unique,rows={},[]
    for p in sorted(sources):
        truth=read(p)
        scope=p.parents[1]/"clean_task_scope.json"
        instruction=read(scope)["authorized_task"] if scope.exists() else truth["task"]
        key=instruction,json.dumps(truth.get("oracle_requirement_program") or truth.get("score_spec"),sort_keys=True)
        if key in unique:
            continue
        unique[key]=True
        compiled=compile_text(instruction)
        n=exercise(compiled[0],reference(truth,instruction)) if compiled else 0
        for addition in ("不要采集编号。","也要核实另一排设备。","除非工作人员同意。"):
            assert compile_text(instruction+addition) is None
        rows.append(dict(family=truth["family"],recognized=compiled is not None,grid_checks=n,
                         instruction=instruction,example=str(p.relative_to(repo))))
    write_table(out/"text_sensitivity.csv",rows)
    summary=dict(unique_text_reference_pairs=len(rows),recognized=sum(r["recognized"] for r in rows),
                 recognized_families=sorted({r["family"] for r in rows if r["recognized"]}),
                 separate_semantic_checks=sum(r["grid_checks"] for r in rows),changed_inputs_rejected=len(rows)*3,
                 new_requests=0,score_changes=0)
    save(out/"text_validation.json",summary)
    print(json.dumps(summary))


if __name__=="__main__":
    main()
