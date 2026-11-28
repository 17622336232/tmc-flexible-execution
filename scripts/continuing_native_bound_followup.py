"""One ordinary GPT decision after a true archived condition and measured source become available."""
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
from research.flexible_scheduling.native_frontier_adapter import NativeFrontierObserver
from scripts.continuing_crossworld_views import assert_snapshot,copied_renderer
from scripts.continuing_native_action_binding import visual_instances
from scripts.continuing_natural_frontier_probe import ROOT as FIRST

BASE = FIRST.parent
CLOCK = BASE/"native-acquisition-clocks-dev01"
ROOT = BASE/"native-natural-followup-dev01"


def packet():
    probe = read(FIRST/"summary.json")
    eligible = [r for r in probe["rows"] if r["ready_for_new_closed_loop"]]
    if len(eligible)!=1:
        raise ValueError("This registered probe admits exactly one source-ready task")
    name = eligible[0]["case"]
    log = FIRST/name/"model_calls.jsonl"
    first, = read_trace_rows(log)[0]
    if digest(log)!=eligible[0]["response_sha256"]:
        raise ValueError("Previous real response differs")
    shot = CLOCK/name/"end-side.jpg"
    meta = read(shot.with_suffix(".json"))
    source = read(CLOCK/(name+".receipt.json"))
    if not read(CLOCK/name/"measures.json")["world_matches"] or source[shot.name]!=meta["sha256"] or digest(shot)!=meta["sha256"]:
        raise ValueError("Archived camera or registered world check differs")
    observer = NativeFrontierObserver(first["input"]["authorized_task"])
    prior = observer.consume(first["input"],first["image_sha256"],first["output"])
    ready = [r["field"] for r in prior["authorized_next"] if r["view"]=="side"]
    if len(ready)!=2 or any(r["photos"] for r in prior["authorized_next"]):
        raise ValueError("Expected two legitimate same-side reads and no available action-side photo in the first turn")
    frame=dict(image_id="onboard_0002",position=meta["position"],yaw=meta["yaw"],capture_ros=meta["ros_stamp"])
    payload = {**first["input"],"current_image_id":frame["image_id"],"geometry":dict(position=meta["position"],yaw=meta["yaw"]),
        "memory":"","records":first["output"]["records"],"view_history":[*first["input"]["view_history"],frame],
        "recent_actions":[dict(action=first["output"]["action"],feedback=dict(ok=True,position=meta["position"],yaw=meta["yaw"]))],
        "this_request_scope":dict(image=frame["image_id"],fields=ready,
            instruction="Only record the listed currently permitted side findings in this reply. Existing bound records remain valid. "
                        "Wait for the return of this request before considering any new conditional requests.")}
    ROOT.mkdir(exist_ok=True)
    reg = dict(case=name,source_photo=str(shot),photo_sha256=digest(shot),first_reply_sha256=digest(log),
        first_return_monotonic=first["return_monotonic_seconds"],payload=payload,allowed_fields=ready,
        model="gpt-6.1-terra",effort="max",timeout=120,cap=1,prompt=sha256(upstream().SYSTEM.encode()).hexdigest(),
        role="Single fixed-camera read of two already released fields on a reconstructed native observation. "
             "The simulator motion is an archived matched-layout measurement, not a live episode in this test.")
    assert_snapshot(ROOT/"registration.json",reg)
    return reg,observer


def grade(reg,observer):
    path = ROOT/"model_calls.jsonl"
    call, = read_trace_rows(path)[0]
    send, = read_trace_rows(ROOT/"sends.jsonl")[0]
    if (send["reservation"]!=call["logical_job_attempt"] or call["input"]!=reg["payload"]
        or send["image_sha256"]!=[reg["photo_sha256"]] or call["image_sha256"]!=send["image_sha256"]
        or call["dispatched_monotonic_seconds"]<=reg["first_return_monotonic"]
        or call["instructions_sha256"]!=reg["prompt"] or call["model"]!=reg["model"] or call["reasoning_effort"]!=reg["effort"]):
        raise ValueError("Real request does not match source, model and permit chronology")
    assert_snapshot(ROOT/"sealed.json",dict(call_sha256=digest(path),ledger_sha256=digest(ROOT/"sends.jsonl")))
    expected = set(reg["allowed_fields"])
    reply = call.get("output") or {}
    seen = observer.consume(call["input"],call["image_sha256"],reply)
    fields = {}
    scope_violations = []
    for r in reply.get("records") or []:
        if type((r.get("facts") or {}).get("visible")) is bool:
            name = r.get("object_id")
            if name not in expected or r.get("evidence_image_id")!="onboard_0002":
                scope_violations.append(name)
            else:
                fields[name] = r["facts"]["visible"]
    # Independent camera grading imports the warehouse classes only after sealing returned decisions.
    photo = Path(reg["source_photo"])
    ep = photo.parent
    with copied_renderer(ep) as (scene,renderer):
        alignment,pixels = visual_instances(scene,renderer,read(photo.with_suffix(".json")), photo,
                                            read(ep/"private/truth.json")["devices"])
    colors = {"蓝色货箱":"blue_carton","白色货箱":"white_carton"}
    truth = {v:max(pixels.get(colors[v],{}).values(),default=0)>=36 for v in expected}
    pass_all = (alignment>=.90 and fields==truth and set(seen.get("newly_bound",[]))==expected and not scope_violations)
    outcome = dict(registration_sha256=digest(ROOT/"registration.json"),receipt_sha256=digest(ROOT/"sealed.json"),
        actual_sends=1,independent_correct_combination=pass_all,observed_fields=fields,source_alignment=alignment,
        truth=truth,scope_violations=scope_violations,reported_action=(reply.get("action") or {}).get("type"),
        live_service_s=call["return_monotonic_seconds"]-call["dispatched_monotonic_seconds"],
        error=(call.get("error") or {}).get("code"),next_authorized=seen.get("authorized_next",[]))
    save(ROOT/"summary.json",outcome)
    print(outcome)


def run(reg,observer):
    from uav_bridge.chatgpt_auth import CredentialStore
    photo = Path(reg["source_photo"])
    log = ROOT/"model_calls.jsonl"
    with (BASE/"online.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not log.exists():
            if (ROOT/"start.json").exists() or unresolved_service_limits(BASE):
                raise ValueError("Existing unresolved service gate")
            save(ROOT/"start.json",dict(registration_sha256=digest(ROOT/"registration.json")))
            client = ModelClient(log,timeout=reg["timeout"],
                credential_store=CredentialStore(directory=Path("/home/tdszy/uav/Diff-VLA/.runtime/chatgpt")),
                request_budget=RequestBudget(ROOT/"sends.jsonl",reg["cap"]))
            image = dict(label="CURRENT onboard RGB onboard_0002; current geometry belongs to this image",data=photo.read_bytes())
            try:
                client.request(upstream().SYSTEM,reg["payload"],[image],attempt_limit=1)
            except Exception:
                if not log.exists():
                    raise
        call, = read_trace_rows(log)[0]
        err = call.get("error") or {}
        if (not call.get("dispatched_monotonic_seconds") or err.get("code") in EXTERNAL_RESPONSE_CODES
            or err.get("status") in (401,403,429)):
            raise ValueError("Original external restriction retained")
    grade(reg,observer)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("register","run","grade"))
    a = parser.parse_args()
    reg, watch = packet()
    if a.mode=="run":
        run(reg,watch)
    elif a.mode=="grade":
        grade(reg,watch)
    else:
        print(reg["case"],reg["allowed_fields"])
