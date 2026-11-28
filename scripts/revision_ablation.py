"""Post-audit coverage check and matched mechanism ablations; separate from primary 120."""
from collections import defaultdict
from dataclasses import asdict, replace
import json
from itertools import product
from statistics import mean

from experiments.flexible_execution.evaluate import score
from experiments.flexible_execution.revision_audit import OUT, save, write_table
from scripts.revision_decisions import reference
from scripts.revision_mechanisms import DevelopmentEpisode, fixture


class Ablated(DevelopmentEpisode):
    def __init__(self, case, variant):
        super().__init__(case,"common-anchor" if variant=="common-terminal" else "adaptive-progress")
        self.variant=variant
        if variant=="no-shared-view":
            self.task=replace(self.task,skills=tuple(s for s in self.task.skills if len(s.covers)==1))

    def snapshot(self):
        state=super().snapshot()
        if self.variant=="no-initial-reuse":
            reacquirable={key for s in self.task.skills for key in s.covers}
            keep={key:e for key,e in state.evidence.items() if key not in reacquirable or not e.frame_id.startswith("initial-")}
            return replace(state,evidence=keep)
        return state


def main():
    rows=[]
    archive=OUT/"ablation-verified"
    archive.mkdir(exist_ok=True)
    trace_rows=[]
    for kind,delay,move in product(("shared","reuse","multi_pending","expensive"),(2,7,13),(.5,1.5)):
        case=fixture(kind,delay,move,True)
        if kind=="expensive":
            # The original grid had a cheap alternative. Keep it intact and
            # explicitly test the missing case in which both paths are detours.
            costs={"front":0.,"far":-16.,"side":-15.,"alternate":-19.,"finish":5.}
            routes=tuple((a,b,abs(costs[a]-costs[b])*move) for a,b,_ in case.task.travel)
            case=replace(case,id=case.id+"-all-detours",task=replace(case.task,travel=routes))
        opt, options=reference(case.task,[k for k,_ in case.initial_material])
        for variant in ("full","no-initial-reuse","no-shared-view","common-terminal"):
            driver=Ablated(case,variant)
            events=driver.run()
            result=score(case,events)
            trace_rows.append(dict(case=case.id,kind=kind,delay=delay,move=move,variant=variant,
                                   fixture=asdict(case),events=events))
            acts=[e for e in events if e["event"]=="action_started" and e["skill_id"]]
            rows.append(dict(case=case.id,kind=kind,variant=variant,latency=delay,travel=move,
                             finite_best="|".join(k for k,v in options.items() if abs(v-opt)<1e-7),
                             pending_material_actions=sum(bool(e["queries_pending"]) for e in acts),
                             success=result["success"],trace_errors=result["trace_errors"],
                             time_s=result["restricted_completion_seconds"],photos=result["observations"]))
    with (archive/"events.jsonl").open("w",encoding="utf8") as stream:
        for entry in trace_rows:
            stream.write(json.dumps(entry,ensure_ascii=False)+"\n")
    write_table(archive/"mechanism_ablations.csv",rows)
    groups=defaultdict(list)
    for r in rows:
        groups[(r["kind"],r["variant"])].append(r)
    table=[dict(kind=k,variant=v,n=len(g),passes=sum(r["success"] for r in g),
                mean_time_s=mean(r["time_s"] for r in g),photos_total=sum(r["photos"] for r in g),
                pending_actions=sum(r["pending_material_actions"] for r in g)) for (k,v),g in groups.items()]
    write_table(archive/"ablation_summary.csv",table)
    print([(r["kind"],r["variant"],r["photos_total"],r["pending_actions"]) for r in table])


if __name__=="__main__":
    main()
