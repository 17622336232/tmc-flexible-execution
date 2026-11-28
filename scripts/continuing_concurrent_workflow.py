"""Run the empirical-clock event study. No transport, renderer, hidden labels, or synthetic waiting."""
import argparse
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

from experiments.flexible_execution.revision_audit import digest,read,save,write_table
from research.flexible_scheduling.event_workflow import ALWAYS,EventModel,Mission,Query
from research.flexible_scheduling.event_policy import Planner
from scripts.continuing_crossworld_views import assert_snapshot

CFG=Path("config/concurrent_workflow_dev01.json")
ROOT=Path("outputs/continuing/concurrent-workflow-dev01")


def tick(v):return max(1,round(v*10))


def clocks():
    routes=[]
    for p in sorted((ROOT.parent/"local-view-navigation01").glob("*/*/navigation.json")):
        d=read(p)
        if d["outward_arrived"] and d["return_arrived"]:
            routes.append((d["outward_s"],d["return_s"],digest(p)))
    measured=read(ROOT.parent/"observed-return-dev02/schedule_reference.json")["rows"][0]["component_times"]
    service={}
    for domain in ("equipment","campus"):
        d=ROOT.parent/"same-frame-region-assay01"/domain
        values={m:read(d/(m+".receipt.json"))["service_s"] for m in ("first","second","group")}
        service[domain]=dict(single=tick(median([values["first"],values["second"]])),pair=tick(values["group"]),raw_s=values)
    return dict(arc=[tick(median(v[i] for v in routes)) for i in (0,1)],
                capture=[tick(measured["photo-a"]),tick(measured["photo-b"])],
                routes=[h for a,b,h in routes],service=service)


def query(label,site,clauses=ALWAYS,early=False):
    return Query(label,(site,) if type(site) is int else site,clauses,early)


def families(name,n=2):
    # A child may share evidence with its parent, a sibling, or several other branches.
    yes=lambda i:(((i,True),),)
    either=lambda a,b:(((a,True),),((b,True),))
    if name=="paired-screen":
        qs=(query("two primary screens: first",0),query("two primary screens: second",0),
            query("side evidence for first",1,yes(0)),query("side evidence for second",1,yes(1)))
    elif name=="relay-inspection":
        qs=(query("primary screen",0),query("independent site check",1),query("conditional second-site field",1,yes(0)),
            query("conditional original-site field",0,yes(1)),query("shared final finding",1,either(2,3)))
    elif name=="shared-followup":
        qs=(query("first priority",0),query("second priority",0),query("one downstream shared check",1,either(0,1)),
            query("original view followup",0,yes(2),True))
    elif name=="permission-gated-transit":
        qs=(query("first screen",0),query("followup at restricted bay",1,yes(0)),
            query("second fact at restricted bay",1,yes(0)),query("original-view report item",0,yes(1)))
    elif name=="existing-archive":
        qs=(query("front observation",0),query("archived detail",1),query("second archived detail",1,yes(0)),
            query("reuse original observation",0,yes(1)))
    elif name=="no-flexibility":
        qs=(query("primary",0),query("strict successor",0,yes(0)),query("strict final successor",0,yes(1)))
    elif name=="fan-in-scale":
        qs=tuple(query("independent screen "+str(i),0) for i in range(n))
        qs+=(query("common downstream question",1,tuple(((i,True),) for i in range(n))),)
    else:raise ValueError(name)
    return qs


def mission(name,domain,slots,cl,n=2):
    q=families(name,n)
    raw=3 if name=="existing-archive" else 1
    stop=1 if name=="relay-inspection" else 0
    early=(True,name!="permission-gated-transit")
    t=cl["service"][domain]
    return Mission(("original","second"),q,((0,cl["arc"][0]),(cl["arc"][1],0)),tuple(cl["capture"]),
                   early,0,raw,(0,t["single"],t["pair"]),0,stop,slots)


def register():
    cfg=read(CFG);cl=clocks()
    code=Path("research/flexible_scheduling")
    desc=dict(config_sha256=digest(CFG),model_sha256=digest(code/"event_workflow.py"),
              planner_sha256=digest(code/"event_policy.py"),driver_sha256=digest(Path(__file__)),clocks=cl)
    ROOT.mkdir(parents=True,exist_ok=True)
    assert_snapshot(ROOT/"registration.json",desc)
    return cfg,cl


def solve_case(name,domain,slots,n,mode,budget,cl):
    m=mission(name,domain,slots,cl,n)
    ablate={"serial":"serial","ordinary-async":"no-speculative","single-read":"single-read","no-early":"no-speculative"}.get(mode)
    model=EventModel(m)
    engine=Planner(model,quotient=mode!="generic",ablation=ablate)
    s=model.initial();start=perf_counter()
    if mode in ("ordinary-async","strong-prefetch"):
        if mode=="ordinary-async":
            val,act=engine.incumbent(s)
        else:val,act=engine.strong(s)
        result=dict(value_ticks=val,action=act,proved=False,expanded=0,cache_hits=0,candidate_evaluations=0,
                    rollout_states=engine.rollout_nodes,root_lower_ticks=engine.cost_lb(s),strong_ticks=val)
    else:
        result=engine.solve(s,budget=budget)
    elapsed=perf_counter()-start
    result["first_dispatch"]=asdict(result.pop("action"))
    result.update(case=name,domain=domain,slots=slots,roots=n if name=="fan-in-scale" else None,method=mode,
                  budget=budget,expected_s=result["value_ticks"]/10,cpu_wall_s=elapsed)
    return result


def jobs(cfg):
    for name in cfg["balanced"]:
        scales=cfg["scale_roots"] if name=="fan-in-scale" else [2]
        for n in scales:
            for domain in ("equipment","campus"):
                for slots in (1,2):
                    # Alternate the first search to limit systematic host-order effects.
                    for mode in ("ordinary-async","strong-prefetch"):
                        yield name,domain,slots,n,mode,0
                    for lim in cfg["budgets"]:
                        for mode in (("generic","joint") if (slots==1)==(domain=="equipment") else ("joint","generic")):
                            yield name,domain,slots,n,mode,lim
                    if name!="fan-in-scale":
                        for mode in ("serial","single-read","no-early"):
                            yield name,domain,slots,n,mode,cfg["budgets"][-1]
                    if name in ("no-flexibility","paired-screen") and slots==1:
                        for mode in ("generic","joint"):
                            yield name,domain,slots,n,mode,cfg["full_small_budget"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("register","run","summary"))
    p.add_argument("--case");p.add_argument("--limit",type=int);p.add_argument("--domain");p.add_argument("--slots",type=int)
    args=p.parse_args()
    cfg,cl=register()
    if args.mode=="register":
        print(cl);return
    folder=ROOT/"solves";folder.mkdir(exist_ok=True)
    count=0
    for case,domain,slots,n,mode,budget in jobs(cfg):
        if args.mode=="summary":break
        if (args.case and case!=args.case) or (args.domain and domain!=args.domain) or (args.slots and slots!=args.slots):continue
        dst=folder/f"{case}_{domain}_{slots}_{n}_{mode}_{budget}.json"
        if dst.exists():continue
        value=solve_case(case,domain,slots,n,mode,budget,cl)
        save(dst,value);count+=1
        if args.limit and count>=args.limit:break
    rows=[read(p) for p in sorted(folder.glob("*.json"))]
    rr={(v["case"],v["domain"],v["slots"],v["roots"],v["method"],v["budget"]):v for v in rows}
    pairs=[]
    for (case,domain,k,n,mode,budget),g in rr.items():
        if mode!="generic":continue
        j=rr.get((case,domain,k,n,"joint",budget))
        if not j:continue
        if j["proved"] and g["proved"] and abs(g["expected_s"]-j["expected_s"])>1e-6:
            raise ValueError("Certified same-information solvers disagree")
        pairs.append(dict(case=case,domain=domain,slots=k,roots=n,budget=budget,generic_s=g["expected_s"],
           joint_s=j["expected_s"],generic_exact=g["proved"],joint_exact=j["proved"],
           generic_nodes=g["expanded"],joint_nodes=j["expanded"],generic_cpu_s=g["cpu_wall_s"],joint_cpu_s=j["cpu_wall_s"]))
    if rows:write_table(ROOT/"solvers.csv",rows)
    if pairs:write_table(ROOT/"equal_budget.csv",pairs)
    totals=dict(registered=len(list(jobs(cfg))),finished=len(rows),paired_equal_budgets=len(pairs),
         both_certified=sum(p["generic_exact"] and p["joint_exact"] for p in pairs),
         jointly_certified_only=sum(p["joint_exact"] and not p["generic_exact"] for p in pairs),
         generic_certified_only=sum(p["generic_exact"] and not p["joint_exact"] for p in pairs),
         different_feasible_costs=sum(abs(p["generic_s"]-p["joint_s"])>1e-7 for p in pairs),new_model_requests=0)
    save(ROOT/"summary.json",totals)
    print(totals,flush=True)


if __name__=="__main__":
    main()
