"""Eight single-decision calls on exact historical first cameras, bound and sealed before navigation."""
import argparse
import csv
import fcntl
from hashlib import sha256
import json
from pathlib import Path

from experiments.flexible_execution.adapters.original_controller import upstream
from experiments.flexible_execution.continuing_jobs import unresolved_service_limits
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.flexible_execution.revision_audit import digest,read,save
from experiments.flexible_execution.timeline import read_trace_rows
from experiments.task_attack.model import EXTERNAL_RESPONSE_CODES,ModelClient
from research.flexible_scheduling.conditional_view_move import INSTRUCTION
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_navigation_assay import geometry_hash


CFG=Path("config/first_view_contingency_dev01.json")
ROOT=Path("outputs/continuing/first-view-contingency-dev01")
AUDIT=ROOT.parent/"results/actual_approach_failures.csv"
ORDINARY=("authorized_task","current_image_id","geometry","scene_conditions","execution_clock","memory","records",
          "recent_actions","last_tool_feedback","recent_images","view_history")


def instructions(method):
    base=upstream().SYSTEM
    return base if method=="ordinary" else base+"\n"+INSTRUCTION


def register():
    cfg=read(CFG)
    candidates={}
    for row in csv.DictReader(AUDIT.open()):
        if (row["cohort"]=="historical-custom" or row["step"]!="1" or row["decision_bound"]!="True" or
              row["actual_approach_ok"]!="False" or row["error_type"]!="VisualGeometryError"):
            continue
        rc=[p for p in ROOT.parent.glob("*/"+row["case"]+"/"+row["method"]+".receipt.json")
            if (p.parent/row["method"]/"clean_preflight.json").is_file()]
        if len(rc)!=1:
            raise ValueError("Ambiguous native source "+str(rc))
        ep=rc[0].parent/row["method"]
        p=ep/"private/robot/model_calls.jsonl"
        if digest(p)!=row["source_log_sha256"]:
            raise ValueError("History of the actual initial decision differs")
        calls=read_trace_rows(p)[0]
        c=next(c for c in calls if c.get("logical_job_attempt")==int(row["attempt"]))
        ctx=c["input"];photo=p.parent/"onboard_0001.jpg"
        if (ctx["current_image_id"]!="onboard_0001" or ctx.get("recent_images") or c["image_sha256"]!=[digest(photo)]
            or c["image_sha256"]!=[row["source_rgb"]] or c["output"]["action"].get("type")!="approach"):
            raise ValueError("Not a genuine first camera and matched approach")
        s=read(ep/"clean_preflight.json")
        key=(s["family"],s["layout_seed"])
        rank=sha256(json.dumps([cfg["salt"],str(ep),digest(photo)],separators=(",",":")).encode()).hexdigest()
        candidates.setdefault(key,[]).append((rank,ep,c,photo,s))
    if len(candidates)!=cfg["max_layouts"]:
        raise ValueError("Unexpected number of exposed first-decision layouts; do not expand the budget")
    items=[]
    for index,key in enumerate(sorted(candidates)):
        _,ep,c,photo,s=min(candidates[key],key=lambda r:r[0])
        ctx={k:c["input"][k] for k in ORDINARY}
        methods=["ordinary","contingent"] if index%2==0 else ["contingent","ordinary"]
        items.append(dict(id=ep.parent.name,methods=methods,source_episode=str(ep),source_sha256=digest(photo),
            reference=str(photo),original_approach=c["output"]["action"],candidates=len(candidates[key]),
            original_call_sha256=digest(ep/"private/robot/model_calls.jsonl"),matching_world=geometry_hash(ep),
            public_input=ctx,settings={k:s[k] for k in ("family","seed","layout_seed","equipment_count","source_profile")}))
    reg=dict(config_sha256=digest(CFG),policy_sha256=digest(Path("research/flexible_scheduling/conditional_view_move.py")),
             prompts={m:sha256(instructions(m).encode()).hexdigest() for m in ("ordinary","contingent")},cases=items)
    ROOT.mkdir(parents=True,exist_ok=True)
    assert_snapshot(ROOT/"registration.json",reg)
    return reg


def slots(reg):
    return [(c,m,ROOT/c["id"]/m) for c in reg["cases"] for m in c["methods"]]


def reconcile(case,method,folder):
    log=folder/"model_calls.jsonl"
    rows=read_trace_rows(log)[0]
    if len(rows)!=1:
        raise ValueError("One ordinary inference attempt was registered")
    c=rows[0]
    if (c["input"]!=case["public_input"] or c["image_sha256"]!=[case["source_sha256"]]
            or c["instructions_sha256"]!=sha256(instructions(method).encode()).hexdigest() or
            c["model"]!="gpt-6.1-terra" or c["reasoning_effort"]!="max"):
        raise ValueError("Request differs from its source, model or protocol")
    start=c.get("dispatched_monotonic_seconds")
    end=c.get("return_monotonic_seconds")
    if start is not None:
        reserved=[b for b in read_trace_rows(ROOT/"sends.jsonl")[0] if b["reservation"]==c["logical_job_attempt"]]
        if len(reserved)!=1 or reserved[0]["image_sha256"]!=c["image_sha256"] or end is None or end<start:
            raise ValueError("Missing genuine transmission")
    error=c.get("error") or {}
    status=error.get("status");code=error.get("code")
    raw=c.get("output") or {}
    receipt=dict(case=case["id"],method=method,log_sha256=digest(log),sent=start is not None,
                 service_s=end-start if start is not None else None,actual_reply=c.get("output") is not None,
                 requested_action=(raw.get("action") or {}).get("type"),supplied_option=isinstance(raw.get("on_unresolved"),dict),
                 endpoint_block=code if code in EXTERNAL_RESPONSE_CODES or status in (401,403,429) else None,
                 error=code)
    assert_snapshot(folder.with_suffix(".receipt.json"),receipt)
    return receipt


def predict(reg):
    from uav_bridge.chatgpt_auth import CredentialStore
    cfg=read(CFG)
    with (ROOT.parent/"online.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=RequestBudget(ROOT/"sends.jsonl",cfg["max_sends"])
        for case,method,folder in slots(reg):
            rc=folder.with_suffix(".receipt.json")
            if (folder/"model_calls.jsonl").exists():
                old=reconcile(case,method,folder)
                if old["endpoint_block"]:
                    raise SystemExit("Original service interruption at "+str(rc))
                continue
            if folder.exists() or rc.exists():
                raise SystemExit("Original attempt is unsettled; inspect "+str(folder))
            if unresolved_service_limits(ROOT.parent):
                raise SystemExit("The authorized service has a recorded restriction")
            photo=Path(case["reference"])
            data=photo.read_bytes()
            if sha256(data).hexdigest()!=case["source_sha256"]:
                raise ValueError("Camera source changed")
            folder.mkdir(parents=True)
            save(folder/"start.json",dict(registration_sha256=digest(ROOT/"registration.json"),method=method,
                                           source_sha256=case["source_sha256"]))
            client=ModelClient(folder/"model_calls.jsonl",timeout=cfg["timeout_s"],
                 credential_store=CredentialStore(directory=Path("/home/tdszy/uav/Diff-VLA/.runtime/chatgpt")),
                 request_budget=budget)
            try:
                client.request(instructions(method),case["public_input"],
                               [dict(label="CURRENT onboard RGB onboard_0001; current geometry belongs to this image",data=data)],attempt_limit=1)
            except Exception:
                if not (folder/"model_calls.jsonl").exists():
                    raise
            result=reconcile(case,method,folder)
            print(case["id"],method,result["requested_action"],result["supplied_option"],result["error"],flush=True)
            if result["endpoint_block"] or not result["sent"]:
                raise SystemExit("Original service failure at "+str(rc))
        seal(reg)


def seal(reg):
    hashes={}
    for c,m,folder in slots(reg):
        reconcile(c,m,folder)
        rc=folder.with_suffix(".receipt.json")
        hashes[c["id"]+"/"+m]=dict(receipt=digest(rc),model_log=digest(folder/"model_calls.jsonl"))
    assert_snapshot(ROOT/"sealed_replies.json",hashes)
    return hashes


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["register","predict","status","seal"])
    mode=p.parse_args().mode
    reg=register()
    if mode=="predict":predict(reg)
    elif mode=="seal":print(len(seal(reg)))
    else:
        for c,m,f in slots(reg):
            rc=f.with_suffix(".receipt.json")
            print(c["id"],m,read(rc)["requested_action"] if rc.exists() else "started" if f.exists() else "unsent")


if __name__=="__main__":
    main()
