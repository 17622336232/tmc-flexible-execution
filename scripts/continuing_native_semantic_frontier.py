"""Audit the original decision loop for actual multi-field use and only source-supported grouping opportunities."""
from collections import Counter,defaultdict
from pathlib import Path

from experiments.flexible_execution.revision_audit import digest,read,save,write_table
from experiments.flexible_execution.timeline import read_trace_rows

ROOT=Path("outputs/continuing")
COHORTS=("native-views-dev01","native-evidence-dev02","coordinated-handoff-dev03")
FIELDS={"cracks":"cracked","large_inventory":"box_count","blocked_exit":"blocked"}


def audit(ep):
    field=FIELDS[read(ep/"clean_preflight.json")["family"]]
    robot=ep/"private/robot";log=robot/"model_calls.jsonl"
    if not log.exists():return []
    offered=defaultdict(list);known=set();out=[]
    for ordinal,c in enumerate(read_trace_rows(log)[0],1):
        inp=c["input"];current=inp.get("current_image_id")
        if not current:continue
        names=[r["image_id"] for r in inp["recent_images"]]+[current]
        if len(names)!=len(c["image_sha256"]):raise ValueError("Unmatched ordinary source list")
        hashes=dict(zip(names,c["image_sha256"]))
        for name,h in hashes.items():
            if digest(robot/(name+".jpg"))!=h:raise ValueError("Wrong ordinary camera evidence")
        reply=c.get("output") or {}
        records=list(reply.get("records") or [])+list((reply.get("report") or {}).get("records") or [])
        facts=set();unsupported=0
        for row in records:
            f=(row.get("facts") or {}).get(field);src=row.get("evidence_image_id");subject=row.get("object_id")
            ok=type(f) is bool if field!="box_count" else type(f) is int and f>=0
            if not ok or not subject:continue
            if src not in hashes:
                unsupported+=1;continue
            facts.add((subject,f,hashes[src]))
        new=facts-known
        groups=Counter(x[2] for x in new)
        late=[x for x in new if offered[x[2]]]
        earliest_joint=False
        if new:
            sources={x[2] for x in new}
            earliest_joint=bool(set.intersection(*(set(offered[h]) for h in sources)))
        action=(reply.get("action") or {}).get("type")
        # A fixed evidence input might support asking earlier. We never equate this with a
        # correct earlier answer or remove a regular navigation/coverage/submission decision.
        eligible=bool(action in ("observe","record") and new and earliest_joint and not unsupported)
        out.append(dict(case=ep.parent.name,cohort=ep.parent.parent.name,method=ep.name,step=ordinal,
            action=action,has_reply=bool(reply),field=field,new_bound_fields=len(new),
            multiple_new_fields_on_one_photo=any(v>1 for v in groups.values()),
            new_facts_with_an_earlier_supplied_photo=len(late),all_new_on_one_earlier_input=earliest_joint,
            nonmotion_field_turn_candidate=eligible,out_of_input_mentions=unsupported,
            service_s=c["return_monotonic_seconds"]-c["dispatched_monotonic_seconds"],source_log_sha256=digest(log)))
        for h in hashes.values():offered[h].append(ordinal)
        known|=facts
    return out


def summarize(rows):
    return dict(ordinary_requests=len(rows),valid_replies=sum(r["has_reply"] for r in rows),
                new_source_bound_primary_facts=sum(r["new_bound_fields"] for r in rows),
                turns_already_combining_new_fields_on_same_photo=sum(r["multiple_new_fields_on_one_photo"] for r in rows),
                newly_cited_facts_on_earlier_supplied_photos=sum(r["new_facts_with_an_earlier_supplied_photo"] for r in rows),
                nonmotion_candidate_turns=sum(r["nonmotion_field_turn_candidate"] for r in rows),
                service_on_candidates_s=sum(r["service_s"] for r in rows if r["nonmotion_field_turn_candidate"]))


def main():
    rows=[]
    for n in COHORTS:
        for rc in sorted((ROOT/n).glob("*/*.receipt.json")):
            d=read(rc)
            ep=rc.parent/d["method"]
            if d["state"] in ("scored","interrupted-after-dispatch","closed-failure"):
                rows+=audit(ep)
    dest=ROOT/"results"
    write_table(dest/"native_semantic_opportunities.csv",rows)
    out=dict(all=summarize(rows),original=summarize([r for r in rows if r["method"]=="original-gpt"]),
             by_cohort={n:summarize([r for r in rows if r["cohort"]==n]) for n in COHORTS},
             criterion="Only genuine in-input source hashes and correctly typed primary predicates are counted; counting a later observed fact is not proof the same answer was obtainable earlier. Candidate-only durations cannot be counted as task savings. Exact source-claim triples repeated in a later report count once per trajectory. No simulator labels or extra requests.")
    save(dest/"native_semantic_frontier.json",out)
    print(out["all"],out["original"])


if __name__=="__main__":
    main()
