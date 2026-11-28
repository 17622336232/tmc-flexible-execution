"""Upper bounds under the calibrated house service model, prior to any whole-task field results."""
from research.flexible_scheduling.event_workflow import EventModel
from research.flexible_scheduling.event_lookahead import LookaheadPlanner
from research.flexible_scheduling.live_event_room import mission,time_reference
from experiments.flexible_execution.revision_audit import digest,read,save
from scripts.continuing_room_event_calibration import CAL,ROOT,prepare


def main():
    cfg,_=prepare()
    cal=read(CAL/"summary.json");rows=[]
    for case in cfg["cases"]:
        cost=time_reference(case,cal,ROOT.parent/"room-workflow-dev01/smoke")
        m=mission(case,cost)
        values={}
        for name,ablation in (("joint",None),("generic",None),("serial","serial"),("single-read","single-read"),
                              ("no-early","no-speculative")):
            model=EventModel(m);p=LookaheadPlanner(model,quotient=name!="generic",ablation=ablation)
            s=model.initial();opt=p.solve(s,budget=100000)
            values[name]=dict(expected_s=opt["value_ticks"]/10,proved=opt["proved"],expansions=opt["expanded"],
                              strong_first_two_dispatches_s=opt["strong_ticks"]/10,root_lower_s=opt["root_lower_ticks"]/10)
        serial=values["serial"];joint=values["joint"]
        canonical=joint["expected_s"] if joint["proved"] else joint["root_lower_s"]
        pct=100*(serial["expected_s"]-canonical)/serial["expected_s"]
        rows.append(dict(case=case["id"],cost=cost,estimates=values,
            possible_fractional_improvement_over_optimized_serial_pct=pct,
            comparable="Mission clocks measured from the initial photo, current declared prior of independent 0.5. This is a conditional model capacity; live correctness and delivery are unmeasured here."))
    result=dict(calibration_sha256=digest(CAL/"summary.json"),cases=rows,new_model_requests=0)
    save(ROOT/"prior_bounds.json",result)
    for r in rows:
        print(r["case"],round(r["possible_fractional_improvement_over_optimized_serial_pct"],2),r["estimates"]["joint"])


if __name__=="__main__":
    main()
