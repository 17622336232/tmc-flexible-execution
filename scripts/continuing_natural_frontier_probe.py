"""Probe reuse from the exact native controller schema on unnumbered Chinese instructions."""
import argparse
from dataclasses import asdict
import fcntl
from hashlib import sha256
import json
from pathlib import Path

from experiments.flexible_execution.adapters.original_controller import upstream
from experiments.flexible_execution.continuing_jobs import unresolved_service_limits
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.flexible_execution.revision_audit import digest, read, save
from experiments.flexible_execution.timeline import read_trace_rows
from experiments.task_attack.model import EXTERNAL_RESPONSE_CODES, ModelClient
from research.flexible_scheduling.natural_frontier import discover, read_ordinary_records, released, source_map
from research.flexible_scheduling.native_frontier_adapter import NativeFrontierObserver
from research.flexible_scheduling.ordinary_frontier import camera_goals, pose_matches
from scripts.continuing_crossworld_views import assert_snapshot

CFG = Path("config/native_natural_frontier_dev01.json")
ROOT = Path("outputs/continuing/native-natural-frontier-dev01")
SOURCE = ROOT.parent / "native-section-dev01" / "registration.json"


def register():
    cfg = read(CFG)
    old_cases = {c["id"]:c for c in read(SOURCE)["cases"]}
    rows = []
    for c in cfg["cases"]:
        old = old_cases[c["id"]]
        plan = discover(c["task"])
        if plan is None or digest(Path(old["source"])) != old["sha256"]:
            raise ValueError("Declared task or historical source failed the discovery check")
        rows.append(dict(id=c["id"],photo=old["source"],image_sha256=old["sha256"],
              payload={**old["input"],"authorized_task":c["task"]}, discovered=json.loads(json.dumps(asdict(plan)))))
    ROOT.mkdir(exist_ok=True)
    scenarios = dict(config_sha256=digest(CFG),old_sources_sha256=digest(SOURCE),cases=rows,
                     prompt_sha256=sha256(upstream().SYSTEM.encode()).hexdigest(),
                     discovery_sha256=digest(Path("research/flexible_scheduling/natural_frontier.py")))
    assert_snapshot(ROOT/"registration.json",scenarios)
    return cfg, scenarios


def reconcile(cfg,reg):
    records = []
    sends = read_trace_rows(ROOT/"sends.jsonl")[0]
    for c in reg["cases"]:
        path = ROOT/c["id"]/"model_calls.jsonl"
        call, = read_trace_rows(path)[0]
        dispatch = call.get("dispatched_monotonic_seconds")
        reservation = call.get("logical_job_attempt")
        if (not dispatch or call["instructions_sha256"]!=reg["prompt_sha256"] or call["input"]!=c["payload"]
            or call["image_sha256"]!=[c["image_sha256"]] or call["reasoning_effort"]!="max" or
            call["model"]!="gpt-6.1-terra" or sends[reservation-1]["image_sha256"]!=call["image_sha256"]):
            raise ValueError("Unbound ordinary request")
        records.append((c,call,digest(path)))
    if [call["logical_job_attempt"] for _,call,_ in records]!=list(range(1,len(sends)+1)):
        raise ValueError("Unexpected real request")
    assert_snapshot(ROOT/"sealed.json",{c["id"]:h for c,_,h in records})
    rows = []
    for c, call, seal in records:
        data = c["payload"]
        plan = discover(data["authorized_task"])
        geometry, frame = data["geometry"], data["current_image_id"]
        sources = source_map(plan, geometry,[dict(image_id=frame,position=geometry["position"],yaw=geometry["yaw"])])
        output = call.get("output") or {}
        values = read_ordinary_records(plan,output,{},sources)
        action = output.get("action") or {}
        goals = camera_goals(asdict(plan),geometry["position"],geometry["yaw"])
        match = action.get("type")=="move" and pose_matches(action.get("goal"),goals["side"])
        front = {i for i in released(plan,{}) if plan.fields[i].view=="front"}
        shadow = NativeFrontierObserver(data["authorized_task"]).consume(data,call["image_sha256"],output)
        rows.append(dict(case=c["id"], fields_discovered=len(plan.fields), action=action.get("type"),
                         exact_source_fields=[dict(field=plan.fields[i].name,visible=v) for i,v in sorted(values.items())],
                         expected_front_roots=len(front), complete_front_assignment=set(values)==front,
                         planned_turn_matches=match, ready_for_new_closed_loop=match and set(values)==front,
                         shadow=shadow,error=(call.get("error") or {}).get("code"), response_sha256=seal,
                         service_s=call["return_monotonic_seconds"]-call["dispatched_monotonic_seconds"]))
    save(ROOT/"summary.json",dict(rows=rows,closed=len(records),raw_sha256=digest(ROOT/"sealed.json")))
    print(rows)


def run(cfg,reg):
    from uav_bridge.chatgpt_auth import CredentialStore
    store = CredentialStore(directory=Path("/home/tdszy/uav/Diff-VLA/.runtime/chatgpt"))
    budget = RequestBudget(ROOT/"sends.jsonl",cfg["max_sends"])
    with (ROOT.parent/"online.lock").open("a+") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for c in reg["cases"]:
            folder = ROOT/c["id"]
            path = folder/"model_calls.jsonl"
            if not path.exists():
                if folder.exists() or unresolved_service_limits(ROOT.parent):
                    raise SystemExit("Inspect unsettled existing attempt or recorded service limit")
                blob = Path(c["photo"]).read_bytes()
                if sha256(blob).hexdigest()!=c["image_sha256"]:
                    raise ValueError("Registered input changed")
                folder.mkdir()
                client = ModelClient(path,timeout=cfg["per_call_s"],credential_store=store,request_budget=budget)
                try:
                    client.request(upstream().SYSTEM,c["payload"],[dict(data=blob,
                        label="CURRENT onboard RGB onboard_0001; current geometry belongs to this image")],attempt_limit=1)
                except Exception:
                    if not path.is_file():
                        raise
            call, = read_trace_rows(path)[0]
            err = call.get("error") or {}
            if not call.get("dispatched_monotonic_seconds") or err.get("code") in EXTERNAL_RESPONSE_CODES or err.get("status") in (401,403,429):
                raise SystemExit("Preserve original service restriction")
            print(c["id"],bool(call.get("output")),err.get("code"),flush=True)
    reconcile(cfg,reg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode",choices=("register","run","grade"))
    a = p.parse_args()
    cfg, reg = register()
    if a.mode=="run":
        run(cfg,reg)
    elif a.mode=="grade":
        reconcile(cfg,reg)
    else:
        print([(c["id"],len(c["discovered"]["fields"])) for c in reg["cases"]])
