"""Matched one-image perception experiment. Registration and post-receipt grading use no model requests."""
import argparse
from collections import defaultdict
import fcntl
from hashlib import sha256
import json
from pathlib import Path

from experiments.flexible_execution.continuing_jobs import unresolved_service_limits
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.flexible_execution.revision_audit import digest, read, save, write_table
from experiments.flexible_execution.timeline import read_trace_rows
from experiments.task_attack.model import EXTERNAL_RESPONSE_CODES, ModelClient
from research.flexible_scheduling.closed_scope import is_front_crack_question
from research.flexible_scheduling.grouped_regions import Region, PROMPT, answers, dispatch, request_binding


CFG=Path("config/same_frame_region_assay01.json")
ROOT=Path("outputs/continuing/same-frame-region-assay01")
ARCHIVE=Path("outputs/protocol-v2-optimization/custom-final")


def seal(path,value):
    if path.exists():
        if read(path)!=value:
            raise ValueError("Sealed source differs: "+str(path))
    else:
        save(path,value)


def order_key(salt,*values):
    return sha256(json.dumps([salt,*values],sort_keys=True,separators=(",",":")).encode()).hexdigest()


def register():
    config=read(CFG)
    candidates=defaultdict(list)
    for p in sorted(ARCHIVE.glob("custom-*/*/attempt-*/episode/private/robot/model_calls.jsonl")):
        groups=defaultdict(set)
        for c in read_trace_rows(p,allow_truncated=True)[0]:
            d=c["input"]
            if not d.get("bbox_in_attached_photo") or len(c["image_sha256"])!=1:
                continue
            q=d["question"]
            domain="equipment" if is_front_crack_question(q) else "campus" if q=="该点位是否同时配置了灭火器且通道没有被实物堵住？" else None
            if domain:
                groups[(domain,c["image_sha256"][0],q,d["frame_id"])].add(tuple(d["bbox_in_attached_photo"]))
        for (domain,h,q,frame_id),boxes in groups.items():
            if len(boxes)<2:
                continue
            key=order_key(config["salt"],domain,str(p),h,q)
            chosen=sorted(boxes,key=lambda b:order_key(config["salt"],h,b))[:2]
            candidates[domain].append((key,p,h,q,frame_id,chosen))
    cases=[]
    for arm in config["cases"]:
        domain=arm["domain"]
        if not candidates[domain]:
            raise ValueError("No existing image for "+domain)
        _,p,h,q,fid,boxes=min(candidates[domain])
        camera=None
        for raw in sorted(p.parent.glob("*.jpg")):
            meta=raw.with_suffix(".json")
            if meta.exists() and read(meta).get("frame_id")==fid and digest(raw)==h:
                camera=raw
                break
        if camera is None:
            raise ValueError("No source-matched camera file for the preselected question")
        regions=[Region(name,box,q).public() for name,box in zip(("first","second"),boxes)]
        cases.append(dict(domain=domain,order=arm["order"],photo=str(camera),frame_id=fid,sha256=h,
                     source_log=str(p),source_log_sha256=digest(p),regions=regions,eligible_sources=len(candidates[domain])))
    doc=dict(config_sha256=digest(CFG),code_sha256=digest(Path("research/flexible_scheduling/grouped_regions.py")),
             prompt_sha256=sha256(PROMPT.encode()).hexdigest(),cases=cases)
    ROOT.mkdir(parents=True,exist_ok=True)
    seal(ROOT/"registration.json",doc)
    return config,doc


def slots(reg):
    for case in reg["cases"]:
        for mode in case["order"]:
            regions=[Region(r["id"],r["bbox"],r["question"]) for r in case["regions"] if mode=="group" or r["id"]==mode]
            yield case,mode,regions,ROOT/case["domain"]/(mode+".receipt.json")


def reconcile(case,mode,regions,receipt,log):
    calls=read_trace_rows(log)[0]
    if len(calls)!=1:
        raise ValueError("Expected one immutable inference attempt")
    c=calls[0]
    binding=request_binding(case["frame_id"],case["sha256"],regions)
    if not (c["input"]==binding and c["image_sha256"]==[case["sha256"]] and c["model"]=="gpt-6.1-terra" and
            c["reasoning_effort"]=="max" and c["instructions_sha256"]==sha256(PROMPT.encode()).hexdigest()):
        raise ValueError("Mismatched source or model receipt")
    start=c.get("dispatched_monotonic_seconds")
    end=c.get("return_monotonic_seconds")
    ledger=ROOT/"sends.jsonl"
    if start is not None:
        matches=[r for r in read_trace_rows(ledger)[0] if r["reservation"]==c["logical_job_attempt"]]
        if len(matches)!=1 or matches[0]["image_sha256"]!=c["image_sha256"] or end is None or end<start:
            raise ValueError("Missing or inconsistent transport reservation")
    err=c.get("error") or {}
    code=err.get("code")
    external=code in EXTERNAL_RESPONSE_CODES or err.get("status") in (401,403,429)
    doc=dict(mode=mode,domain=case["domain"],sha256=digest(log),sent=start is not None,attempt=c.get("logical_job_attempt"),
             service_s=end-start if start is not None else None,cutoff_s=c["effective_timeout_seconds"],
             values=answers(binding,c["image_sha256"],c.get("output")),error=code,
             endpoint_block=(code or "external_service_error") if external else None)
    seal(receipt,doc)
    return doc


def predict(config,reg):
    from uav_bridge.chatgpt_auth import CredentialStore
    with (ROOT.parent/"online.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if unresolved_service_limits(ROOT.parent):
            raise SystemExit("The authorized plan has a recorded unresolved service limit")
        parent=ROOT.parent/"native-evidence-dev02/warehouse-filter-02/evidence-deferred.receipt.json"
        if not parent.exists() or read(parent).get("state") not in ("scored","interrupted-after-dispatch","closed-failure"):
            raise SystemExit("The previously reserved primary development slot is still pending")
        budget=RequestBudget(ROOT/"sends.jsonl",config["max_total_sends"])
        store=CredentialStore(directory=Path("/home/tdszy/uav/Diff-VLA/.runtime/chatgpt"))
        for case,mode,regions,receipt in slots(reg):
            log=receipt.with_name(mode+".jsonl")
            begun=receipt.with_name(mode+".started.json")
            if receipt.exists() or log.exists():
                r=reconcile(case,mode,regions,receipt,log)
                if not r["sent"]:
                    raise SystemExit("Preserve and inspect the pre-dispatch failure at "+str(receipt))
                if r["endpoint_block"] and unresolved_service_limits(ROOT.parent):
                    raise SystemExit("Observed service limit at "+str(receipt))
                continue
            if begun.exists():
                raise SystemExit("Original dispatch has no terminal log. Inspect the consumed reservation at "+str(begun))
            rgb=Path(case["photo"]).read_bytes()
            if len(regions)==0 or sha256(rgb).hexdigest()!=case["sha256"]:
                raise ValueError("Registered input changed")
            receipt.parent.mkdir(exist_ok=True)
            seal(begun,dict(config_sha256=digest(CFG),registration_sha256=digest(ROOT/"registration.json"),
                            domain=case["domain"],mode=mode))
            client=ModelClient(log,timeout=config["per_call_seconds"],credential_store=store,request_budget=budget)
            try:
                dispatch(client,case["frame_id"],case["sha256"],rgb,regions)
            except Exception as ex:
                if not log.exists():
                    raise RuntimeError("Pre-dispatch error, no complete model log. Preserve the attempt.") from ex
            r=reconcile(case,mode,regions,receipt,log)
            if r["endpoint_block"] or not r["sent"]:
                raise SystemExit("External or pre-dispatch failure: "+str(receipt))


def grade(reg):
    predictions={}
    receipt_hashes={}
    for case,mode,regions,path in slots(reg):
        if not path.exists():
            raise SystemExit("Grading is unavailable until all six registered slots have closed")
        r=reconcile(case,mode,regions,path,path.with_name(mode+".jsonl"))
        predictions[(case["domain"],mode)]=r
        receipt_hashes[str(path.relative_to(ROOT))]=digest(path)
    ledgers=read_trace_rows(ROOT/"sends.jsonl")[0]
    if sorted(r["attempt"] for r in predictions.values() if r["sent"])!=[r["reservation"] for r in ledgers]:
        raise ValueError("Study ledger has an unaccounted reservation")
    seal(ROOT/"sealed_receipts.json",receipt_hashes)
    from scripts.continuing_same_image_claims import predicate,reference_centers
    scores=[];cases=[]
    for c in reg["cases"]:
        raw=Path(c["photo"])
        if digest(raw)!=c["sha256"]:
            raise ValueError("Archived pixels no longer match study")
        centers,alignment,objects=reference_centers(raw.parent,raw.stem)
        domain=c["domain"]
        whole=predictions[(domain,"group")]
        for region in c["regions"]:
            name=region["id"]
            b=region["bbox"]
            ids=[i for i,(x,y) in centers.items() if b[0]<=x<=b[2] and b[1]<=y<=b[3]]
            obj=ids[0] if len(ids)==1 and alignment>=.90 else None
            truth=next((r for r in objects if r["identity"]==obj),{})
            value=predicate(region["question"],truth)
            one=predictions[(domain,name)]
            scores.append(dict(domain=domain,region=name,photo_sha256=c["sha256"],object=obj,alignment=alignment,
                expected=value,group=(whole["values"] or {}).get(name),single=(one["values"] or {}).get(name),
                group_error=whole["error"],single_error=one["error"]))
        rr=[x for x in scores if x["domain"]==domain]
        assigned=all(x["expected"] is not None for x in rr) and len({x["object"] for x in rr})==len(rr)
        single=[predictions[(domain,k)] for k in ("first","second")]
        complete=all(x["group"] in ("positive","negative") and x["single"] in ("positive","negative") for x in rr)
        paired=assigned and complete
        cases.append(dict(domain=domain,regions_independently_assigned=assigned,both_treatments_definite=complete,
                         group_all_correct=all(x["group"]==x["expected"] for x in rr) if assigned else None,
                         single_all_correct=all(x["single"]==x["expected"] for x in rr) if assigned else None,
                         paired_service_saving_s=sum(o["service_s"] for o in single)-whole["service_s"] if paired else None))
    write_table(ROOT/"regions.csv",scores)
    save(ROOT/"summary.json",dict(role="perception-only",registered=6,sends=len(ledgers),regions=scores,cases=cases))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("register","predict","grade"),default="register",nargs="?")
    args=parser.parse_args()
    config,reg=register()
    if args.action=="predict":
        predict(config,reg)
    elif args.action=="grade":
        grade(reg)
    else:
        print("Registered",[(c["domain"],len(c["regions"]),c["sha256"][:12]) for c in reg["cases"]],"requests made by registration: 0")


if __name__=="__main__":
    main()
