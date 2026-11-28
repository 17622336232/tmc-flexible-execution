"""Balanced mechanism study; only the one-front/one-side cost calibration uses measured service times."""
import argparse
from collections import defaultdict
import gc
from pathlib import Path
from statistics import median
from time import perf_counter

from experiments.flexible_execution.revision_audit import read,save,write_table
from research.flexible_scheduling.workflow import Workflow,View,DecisionProblem,merged_demands


OUT=Path("outputs/continuing/workflow")
METHODS=("serial","async","prefetch","strong-prefetch","fixed-joint","generic","factored")


def balanced():
    # Explicit mechanism inputs, not artificial sleeps or estimates of deployment prevalence.
    q=lambda field,p:((0,1-p),(field,p))
    singles=(q(1,.5),)
    cases=[
        ("single-permitted",singles,(View("side",1,4,True,True),),0),
        ("rare-long-trip",(q(1,.04),),(View("distant",1,32,True,True),),0),
        ("capture-only",singles,(View("label",1,4,True,False),),0),
        ("wait-for-positive",singles,(View("protected-side",1,4,False,False),),0),
        ("nothing-else",(((0,1.),),),(),0),
        ("verified-reuse",singles,(View("revisit",1,7,True,True),),1),
        ("raw-on-hand",singles,(View("previous-photo",1,6,False,True,True),),0),
        ("shared-view",(q(1,.65),q(2,.50)),(View("overview",3,5,True,True),View("left",1,4,True,True),View("right",2,4,True,True)),0),
        ("alternatives",(q(1,.60),q(3,.35)),(View("long-shared",3,12,True,True),View("short-left",1,2,True,False),View("short-right",2,3,False,False)),0),
        ("independent-obligations",(((1,.4),(2,.5),(4,.1)),q(8,.3)),(View("pair",3,4,True,True),View("review",4,6,False,False),View("record",8,2,True,False)),0),
        ("unreadable-requires-review",(((0,.4),(1,.4),(2,.2)),),(View("body",1,6,True,False),View("recheck",2,4,False,False)),0),
        ("always-required",(((1,1.),),q(2,.35)),(View("panorama",3,6,True,True),View("single",1,3,True,False)),0),
    ]
    for name,factors,views,verified in cases:
        for state,(front,combined,follow1,follow2,report) in {
            "cheap-combined":(7.,8.,6.,8.,4.),"costly-combined":(6.,16.,5.,14.,4.)}.items():
            yield name,state,Workflow(factors,views,verified,(front,combined),(0.,follow1,follow2),report)


def summarize_tasks(cases):
    data=[]
    for name,state,task in cases:
        reference=None
        for method in METHODS:
            result=DecisionProblem(task).solve(method)
            if method=="generic":
                reference=result
            if method=="factored":
                if abs(result["expected_seconds"]-reference["expected_seconds"])>1e-8 or (
                    result["overlap"],result["joint"])!=(reference["overlap"],reference["joint"]):
                    raise ValueError("Optimal policy mismatch")
            data.append(dict(case=name,cost_profile=state,method=method,model_expected_s=result["expected_seconds"],
                   joint="|".join(task.views[i].name for i in result["joint"]),overlap="|".join(task.views[i].name for i in result["overlap"]),
                   decision_ms=result["solve_seconds"]*1000,actions_evaluated=result["evaluated"]))
    return data


def stress():
    rows=[]
    for graph in ("shared","up-to-five-sources","mixed"):
        for n in (2,4,8,16,32):
            groups=min(n, 2 if graph=="shared" else 4 if graph=="mixed" else 5)
            factors=tuple(((0,.5),(1<<(i%groups),.5)) for i in range(n))
            views=tuple(View("source-%d"%i,1<<i,2+i,True,i%2==0) for i in range(groups))
            task=Workflow(factors,views,0,(7.,9.),(0.,6.,9.),3.)
            observed=defaultdict(list)
            for trial in range(5):
                for mode in (("generic","factored") if trial%2==0 else ("factored","generic")):
                    # Entire solver instance is recreated at each run.
                    t=perf_counter()
                    instance=DecisionProblem(task)
                    result=instance.solve(mode)
                    secs=perf_counter()-t
                    observed[mode].append((secs,result))
                    instance.finish.cache_clear()
                    instance.read_duration.cache_clear()
                gc.collect()
            g=observed["generic"][0][1]
            f=observed["factored"][0][1]
            if abs(g["expected_seconds"]-f["expected_seconds"])>1e-8 or (g["joint"],g["overlap"])!=(f["joint"],f["overlap"]):
                raise AssertionError((graph,n))
            for method,m in observed.items():
                rows.append(dict(graph=graph,predicates=n,distinct_requirements=groups,method=method,repeats=len(m),
                       median_ms=median(x[0] for x in m)*1000,support=len(merged_demands(factors,0)),
                       candidate_plans=m[0][1]["evaluated"],expected_model_s=m[0][1]["expected_seconds"]))
    return rows


def calibrated():
    # These new service samples are acquired specifically for marginal cost.
    import csv
    assay=Path("outputs/continuing/paired-rgb-assay01")
    inputs=read(assay/"inputs.json")["cases"]
    pairs=list(csv.DictReader((assay/"paired.csv").open()))
    for case in inputs:
        own=case["house"]
        # A house's measured run is never used to choose that house's plan.
        others=[p for p in pairs if p["house"]!=own and p["separate_minus_joint_service_s"]]
        if not others:
            continue
        front,joint,side=(median(float(p[k]) for p in others) for k in ("front_s","joint_s","side_s"))
        source=read(Path(case["source"])/"source.json")
        motion=source["motion_estimates"]
        trip=2*motion["turn"]+motion["photo"]
        # New role authorizes both semantic readings. Probability varied explicitly, never read from that house or policy answer.
        for p in (.1,.5,.9):
            factors=(((0,1-p),(1,p)),)
            yield own,"p=%.1f"%p,Workflow(factors,(View("side",1,trip,True,True),),0,(front,joint),(0.,side),0.)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--calibrated",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.calibrated:
        rows=summarize_tasks(calibrated())
        if not rows:
            raise ValueError("No complete RGB service-time pairs")
        write_table(OUT/"leave_house_out_costs.csv",rows)
        print("paired cost predictions",len(rows))
    else:
        rows=summarize_tasks(balanced())
        runtimes=stress()
        write_table(OUT/"balanced.csv",rows)
        write_table(OUT/"solver_cpu.csv",runtimes)
        bycase=defaultdict(dict)
        for row in rows:
            bycase[(row["case"],row["cost_profile"])][row["method"]]=row["model_expected_s"]
        exact=len(bycase)
        matches={m:sum(abs(cell[m]-cell["generic"])<1e-8 for cell in bycase.values()) for m in METHODS}
        save(OUT/"balanced_summary.json",dict(task_profiles=exact,method_runs=len(rows),optimal_cost_matches=matches,
             structural_timing_rows=len(runtimes),stress_scope="Solver CPU only; no claim about VLM service for many simultaneous predicates.",
             limitations="Static return-to-anchor section; fixed observation coverage and factor distributions. Real perception accuracy is a separate paired assay."))
        print("task profiles",exact,"optimal cost matches",matches,"timing rows",len(runtimes))


if __name__=="__main__":
    main()
