"""Measure whether a normal original-GPT camera decision can propose the user's causal work order."""
import argparse
import fcntl
from hashlib import sha256
from pathlib import Path

from experiments.flexible_execution.adapters.original_controller import upstream
from experiments.flexible_execution.continuing_jobs import unresolved_service_limits
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.flexible_execution.revision_audit import digest,read,save
from experiments.flexible_execution.timeline import read_trace_rows
from experiments.task_attack.model import ModelClient,EXTERNAL_RESPONSE_CODES
from research.flexible_scheduling.native_section import SECTION_PROMPT,describe
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_first_views import ORDINARY

CFG=Path("config/native_section_dev01.json")
ROOT=Path("outputs/continuing/native-section-dev01")


def prompt(method):
    return upstream().SYSTEM+("\n"+SECTION_PROMPT if method=="section" else "")


def register():
    cfg=read(CFG);cases=[]
    for c in cfg["cases"]:
        ep=Path(c["source"]);robot=ep/"private/robot"
        log=robot/"model_calls.jsonl"
        call=next(v for v in read_trace_rows(log)[0] if v["input"].get("current_image_id")=="onboard_0001")
        f=robot/"onboard_0001.jpg";h=digest(f)
        if call["image_sha256"]!=[h] or call["input"]["recent_images"]:raise ValueError("Unmatched published first camera")
        inp={k:call["input"][k] for k in ORDINARY}
        inp["authorized_task"]=c["instruction"]
        cases.append(dict(id=c["id"],source=str(f),sha256=h,source_log_sha256=digest(log),input=inp,order=c["order"]))
    reg=dict(config_sha256=digest(CFG),cases=cases,prompts={m:sha256(prompt(m).encode()).hexdigest() for m in ("original","section")},
             tool_sha256=digest(Path("research/flexible_scheduling/native_section.py")))
    ROOT.mkdir(exist_ok=True)
    assert_snapshot(ROOT/"registration.json",reg)
    return cfg,reg


def jobs(cfg,reg):
    cases={c["id"]:c for c in cfg["cases"]}
    for c in reg["cases"]:
        for mode in c["order"]:
            yield cases[c["id"]],c,mode,ROOT/c["id"]/mode


def reconcile(row,mode,folder):
    f=folder/"model_calls.jsonl";calls=read_trace_rows(f)[0]
    if len(calls)!=1:raise ValueError("Single original decision required")
    c=calls[0]
    if (c["input"]!=row["input"] or c["image_sha256"]!=[row["sha256"]] or
        c["instructions_sha256"]!=sha256(prompt(mode).encode()).hexdigest() or c["model"]!="gpt-6.1-terra" or c["reasoning_effort"]!="max"):
        raise ValueError("Original camera, instructions or model changed")
    t=c.get("dispatched_monotonic_seconds")
    if t is not None:
        reservations=[v for v in read_trace_rows(ROOT/"sends.jsonl")[0] if v["reservation"]==c["logical_job_attempt"]]
        if len(reservations)!=1 or reservations[0]["image_sha256"]!=[row["sha256"]]:raise ValueError("Unbound request")
    err=c.get("error") or {};code=err.get("code");external=code in EXTERNAL_RESPONSE_CODES or err.get("status") in (401,403,429)
    rec=dict(case=row["id"],method=mode,sha256=digest(f),sent=t is not None,valid_reply=isinstance(c.get("output"),dict),
             service_s=c["return_monotonic_seconds"]-t if t is not None else None,
             error=code,endpoint_block=(code or "external") if external else None)
    assert_snapshot(folder.with_suffix(".receipt.json"),rec)
    return rec


def grade(cfg,reg):
    entries=list(jobs(cfg,reg))
    receipts=[reconcile(row,m,f) for _,row,m,f in entries]
    assert_snapshot(ROOT/"sealed.json",{v["case"]+"/"+v["method"]:v["sha256"] for v in receipts})
    results=[]
    for (case,row,mode,folder),rc in zip(entries,receipts):
        raw=read_trace_rows(folder/"model_calls.jsonl")[0][0].get("output") or {}
        info=describe(raw,case,"onboard_0001")
        results.append({**rc,**info})
    admitted=[r["case"] for r in results if r["method"]=="section" and r["accepted"]]
    out=dict(registered=4,sends=sum(r["sent"] for r in results),cases=results,eligible_full_cases=admitted,
             sealed_source_sha256=digest(ROOT/"sealed.json"),
             scope="Supplied source, ordinary action, task proposal and field grammar only; independent visual correctness and complete task delivery require new physical episodes.")
    save(ROOT/"proposal_summary.json",out)
    return out


def run(cfg,reg):
    from uav_bridge.chatgpt_auth import CredentialStore
    with (ROOT.parent/"online.lock").open("a+") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for case,row,mode,folder in jobs(cfg,reg):
            if (folder/"model_calls.jsonl").exists():
                rec=reconcile(row,mode,folder)
                if rec["endpoint_block"]:raise SystemExit("Original service interruption is preserved")
                continue
            if folder.exists():raise SystemExit("Unsettled original decision at "+str(folder))
            if unresolved_service_limits(ROOT.parent):raise SystemExit("Authorized model service has a recorded limit")
            data=Path(row["source"]).read_bytes()
            if sha256(data).hexdigest()!=row["sha256"]:raise ValueError("Original camera changed")
            folder.mkdir(parents=True)
            save(folder/"start.json",dict(registration_sha256=digest(ROOT/"registration.json"),case=case["id"],method=mode))
            client=ModelClient(folder/"model_calls.jsonl",timeout=cfg["per_request_s"],
                credential_store=CredentialStore(directory=Path("/home/tdszy/uav/Diff-VLA/.runtime/chatgpt")),
                request_budget=RequestBudget(ROOT/"sends.jsonl",cfg["standalone_max"]))
            try:
                client.request(prompt(mode),row["input"],[dict(label="CURRENT onboard RGB onboard_0001; current geometry belongs to this image",data=data)],attempt_limit=1)
            except Exception:
                if not (folder/"model_calls.jsonl").is_file():raise
            rec=reconcile(row,mode,folder)
            print(case["id"],mode,rec["sent"],rec["error"],flush=True)
            if rec["endpoint_block"] or not rec["sent"]:raise SystemExit("External or startup restriction at "+str(folder))
        return grade(cfg,reg)


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("run","register","grade","status"));a=p.parse_args()
    cfg,reg=register()
    if a.mode in ("register","status"):
        for case,row,mode,folder in jobs(cfg,reg):print(case["id"],mode,folder.with_suffix(".receipt.json").is_file())
    else:
        out=run(cfg,reg) if a.mode=="run" else grade(cfg,reg)
        print(out["sends"],out["eligible_full_cases"])


if __name__=="__main__":
    main()
