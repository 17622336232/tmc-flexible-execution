"""Feed ordinary native records to the discovered conditional task rather than a handwritten graph."""
from dataclasses import replace

from experiments.flexible_execution.revision_audit import digest,read,save
from experiments.flexible_execution.timeline import read_trace_rows
from research.flexible_scheduling.cross_source_workflow import CrossSourcePlanner,CrossSourceWorkflow
from research.flexible_scheduling.event_workflow import indices
from research.flexible_scheduling.native_frontier_adapter import NativeFrontierObserver,scheduling_state
from scripts.continuing_natural_frontier_probe import ROOT as NATIVE
from scripts.continuing_online_cost_replay import BASE,CROSS,OUT,ticks


def main():
    probe = read(NATIVE/"summary.json")
    reg = read(NATIVE/"registration.json")
    action = read(BASE.parent/"native-acquisition-clocks-dev01"/"summary.json")
    service = read(CROSS)
    plans = []
    for c in reg["cases"]:
        receipt = next(v for v in probe["rows"] if v["case"]==c["id"])
        clocks = next(v for v in action["cases"] if v["case"]==c["id"])
        path = NATIVE/c["id"]/"model_calls.jsonl"
        call, = read_trace_rows(path)[0]
        if digest(path) != receipt["response_sha256"]:
            raise ValueError("Native first-decision packet changed")
        watch = NativeFrontierObserver(c["payload"]["authorized_task"])
        inspected = watch.consume(call["input"],call["image_sha256"],call.get("output") or {})
        available = {v["view"] for v in inspected["available_sources"].values()}
        if not receipt["ready_for_new_closed_loop"] or not clocks["phases"]:
            plans.append(dict(case=c["id"],outcome="insufficient-native-evidence",
                 fields=[f.name for f in watch.plan.fields],activity=inspected,local_motion_failure=clocks["error"]))
            continue
        measured = clocks["phases"][0]
        if watch.plan.finish!="side" or measured["view"]!="side" or available!={"front"}:
            raise ValueError("This probe has only measured the outward trip and the front camera")
        motion, capture = ticks(measured["motion_s"]),ticks(measured["after_move_capture_s"])
        existing = ticks(receipt["service_s"])
        for observed in service["cases"]:
            # Use measured price ratios for sensitivity, not as a fit transferred to the warehouse.
            v = observed["observations"]
            ratio = v["joint"]["service_s"]/min(v["front"]["service_s"],v["side"]["service_s"])
            m,s = scheduling_state(watch,available,"front",((0,motion),(99999,0)),(1,capture),
                                    (0,existing,round(1.5*existing)))
            graph = CrossSourceWorkflow(m,max(existing,round(ratio*existing)))
            planner = CrossSourcePlanner(graph)
            best = planner.solve(s,budget=100000)
            generic = CrossSourcePlanner(graph,quotient=False).solve(s,budget=100000)
            if abs(best["value_ticks"]-generic["value_ticks"])>1e-6 or not (best["proved"] and generic["proved"]):
                raise ValueError("Matched generic feasibility differs")
            command = best["action"]
            # An archived, measured side capture licenses planning with its raw frame. It does
            # not set any semantic observation or release a descendant of an unresolved field.
            with_side = replace(s,raw=3,site=1)
            next_read = planner.solve(with_side,budget=100000)["action"]
            plans.append(dict(case=c["id"],outcome="legal-plan-available-in-replay",sensitivity_house=observed["house"],
                              planned_first_action=dict(move=command.motion,read=[m.queries[i].label for _,mask in command.reads for i in indices(mask)]),
                              after_camera_only_reads=[m.queries[i].label for _,mask in next_read.reads for i in indices(mask)],
                              expected_remaining_s=best["value_ticks"]/100,observation=inspected,
                              measured_outward_photo_s=measured["to_fresh_source_s"]))
    result = dict(rows=plans,new_model_requests=0,probe_sha256=digest(NATIVE/"summary.json"),calibration_sha256=digest(CROSS),
            caveat="Fixed original reply and actual source/clock receipts. Group latency ratios and unseen gate routes are sensitivity inputs. "
                   "The actual native position at dispatch is not replaced; this is a policy replay without new actions or semantic outcomes.")
    save(OUT/"native_schedule.json",result)
    print([(r["case"],r["outcome"],r.get("planned_first_action"),r.get("expected_remaining_s")) for r in plans])


if __name__=="__main__":
    main()
