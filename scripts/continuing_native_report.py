"""Independently link actual dispatches, scoped nominations, motion, RGB and report citations."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from urllib.parse import unquote

from experiments.flexible_execution.adapters.native_views import INSTRUCTIONS, close_pose
from experiments.flexible_execution.adapters.original_controller import upstream
from experiments.flexible_execution.clean_coverage import audit_run
from experiments.flexible_execution.revision_audit import digest, read, save, write_table
from experiments.flexible_execution.timeline import read_trace_rows


def rows(path):
    return read_trace_rows(path)[0] if path.exists() else []


def merged(intervals):
    result = []
    for start, finish in sorted(intervals):
        if finish < start:
            raise ValueError("Negative interval")
        if result and start <= result[-1][1]:
            result[-1][1] = max(finish, result[-1][1])
        else:
            result.append([start, finish])
    return result


def length(intervals):
    return sum(b-a for a,b in merged(intervals))


def chain_checks(ep, calls, events, allowed, coverage):
    robot = ep/"private/robot"
    offers = {e["id"]:e for e in events if e["event"] == "native_offer"}
    shots = [e for e in events if e["event"] == "native_raw_photo"]
    inputs = {c["input"]["current_image_id"]:c for c in calls}
    report = read(ep/"result.json").get("report") or {}
    claim_frames = {r.get("evidence_image_id") for r in report.get("records", [])}
    supported = {frame for obj in coverage.get("supported_by_frame", {}).values()
                 for frames in obj.values() for frame in frames}
    proofs, refusals = [], []
    for e in events:
        if e["event"] != "native_early_start":
            continue
        offer = offers[e["candidate"]]
        current = inputs["onboard_%04d" % e["step"]]
        scope_ok = offer["field"] in allowed
        if not scope_ok:
            refusals.append(dict(step=e["step"], id=offer["id"], field=offer["field"]))
        if all((s["candidate"],s["step"]) != (offer["id"],e["step"]) for s in shots):
            proofs.append(dict(case=ep.parent.name, method=ep.name, image=None, candidate=offer["id"],
                early=True, field_authorized=scope_ok, chosen_while_pending=
                current["dispatched_monotonic_seconds"] <= e["monotonic_seconds"] < current["return_monotonic_seconds"],
                stage="started_without_saved_photo"))
    for shot in shots:
        step, cid, image = shot["step"], shot["candidate"], shot["image_id"]
        offer = offers[cid]
        prior = inputs[offer["source"]]
        call = inputs["onboard_%04d" % step]
        following = inputs.get("onboard_%04d" % (step+1))
        nomination = (prior.get("output") or {}).get("next_views") or []
        nominated = any(isinstance(n,dict) and (n.get("field"),n.get("subject"),n.get("bbox")) ==
                        (offer["field"],offer["subject"],offer["source_bbox"]) for n in nomination)
        def event(name, label, **match):
            return next((e for e in events if e["event"] == name and e["step"] == label and
                         all(e.get(k) == v for k,v in match.items())), {})
        early = image.endswith("_early")
        started = event("native_early_start", step, candidate=cid) if early else {}
        go = event("action_started", f"{step}-side", candidate=cid, early=early)
        reached = event("action_finished", f"{step}-side", candidate=cid, early=early)
        # One turn can first fetch candidate X, then fetch a different one requested by GPT.
        restored = next((e for e in events if e["event"] == "native_anchor_checked" and e["step"] == step
                         and e["candidate"] == cid and e["monotonic_seconds"] >= shot["monotonic_seconds"]), {})
        meta = read(robot/(image+".json"))
        raw_hash = digest(robot/(image+".jpg"))
        start, end = (call[k] for k in ("dispatched_monotonic_seconds","return_monotonic_seconds"))
        image_names = [i["image_id"] for i in (following or {}).get("input",{}).get("recent_images",[])]
        next_bound = bool(following and any(n == image and h == raw_hash for n,h in zip(image_names,following["image_sha256"])))
        decision = call.get("output") or {}
        listed=decision.get("use_views")
        if ep.name.startswith("evidence-") and isinstance(listed,list):
            chosen_by_gpt=len(listed)<=2 and all(isinstance(v,str) for v in listed) and cid in listed
        else:
            chosen_by_gpt=decision.get("use_view")==cid
        requested = chosen_by_gpt and (decision.get("action") or {}).get("type") in ("observe","record")
        release = event("native_material_release",step)
        reference_bound = (offer["reference_sha256"] == digest(robot/(offer["source"]+".jpg")) ==
                           prior["image_sha256"][-1] == shot["reference_sha256"] and
                           meta["source_reference"] == offer["source"])
        offered = any(o.get("id") == cid and o.get("source_bbox") == offer["source_bbox"]
                      and o.get("source") == offer["source"] for o in call["input"].get("offered_views",[]))
        scope_ok = offer["field"] in allowed
        pose_ok = close_pose({**meta["position"],"yaw":meta["yaw"]},offer["goal"])
        raw_bound = meta["image_id"] == image and raw_hash == meta["sha256"] == shot["image_sha256"]
        concurrent = bool(early and start <= started.get("monotonic_seconds",-1) <= go.get("monotonic_seconds",-1) <
                          reached.get("monotonic_seconds",-1) <= shot["monotonic_seconds"] < end)
        proves = all((early, concurrent, scope_ok, nominated, offered, reference_bound, pose_ok, raw_bound,
                      requested, image in release.get("released",[]), next_bound, restored.get("restored")))
        proofs.append(dict(case=ep.parent.name,method=ep.name,image=image,sha256=raw_hash,
            candidate=cid,early=early,field=offer["field"],field_authorized=scope_ok,nomination_matches=nominated,
            offered_in_frozen_request=offered,reference_bound=reference_bound,actual_goal_reached=pose_ok,
            raw_bound=raw_bound,within_transport=concurrent,returned_to_anchor=restored.get("restored"),
            actual_gpt_requested=requested,next_input_contains_exact_rgb=next_bound,
            final_report_cites=image in claim_frames,independent_field_support=image in supported,
            proven_early_handoff=proves))
    return proofs, refusals


def audit(ep, receipt, case):
    robot = ep/"private/robot"
    result = read(ep/"result.json")
    assert result["policy"] == "clean" and result["attack_budget"] == 0
    assert digest(ep/"result.json") == receipt["result_sha256"]
    calls = rows(robot/"model_calls.jsonl")
    ledger = rows(ep/"request_attempts.jsonl")
    assert len(calls) == len(ledger) == receipt["sends"]
    suffix="" if ep.name=="original-gpt" else INSTRUCTIONS
    if ep.name.startswith("evidence-"):
        from research.flexible_scheduling.native_controller import EVIDENCE_INSTRUCTIONS
        suffix+=EVIDENCE_INSTRUCTIONS
    nominal = sha256((upstream().SYSTEM + suffix).encode()).hexdigest()
    for ix, (call, entry) in enumerate(zip(calls,ledger),1):
        assert call["instructions_sha256"] == nominal and call["model"] == entry["model"] == "gpt-6.1-terra"
        assert call["reasoning_effort"] == entry["reasoning_effort"] == "max"
        assert entry["reservation"] == call["logical_job_attempt"] == ix
        rgb = [f["image_id"] for f in call["input"]["recent_images"]] + [call["input"]["current_image_id"]]
        assert [digest(robot/(n+".jpg")) for n in rgb] == entry["image_sha256"] == call["image_sha256"]
    events = rows(robot/"timeline.jsonl")
    cov = audit_run(ep,recheck_legacy_fragments=True) if isinstance(result.get("report"),dict) else {}
    task = read(ep/"clean_task_scope.json")["authorized_task"]
    assert all(c["input"]["authorized_task"] == task for c in calls)
    allowed = set((case.get("early_fields") or "").split(","))
    if allowed - {""}:
        assert task.endswith("采集规则：允许在判读结果返回前采集以下材料："+"、".join(case["early_fields"].split(","))+"。")
    proofs, violations = chain_checks(ep,calls,events,allowed,cov)
    intervals = read(ep/"clean_timing.json")["intervals"]
    transport = merged((c["dispatched_monotonic_seconds"],c["return_monotonic_seconds"]) for c in calls)
    action_ends = {(e["step"],e["monotonic_seconds"]):e for e in events if e["event"]=="action_finished"}
    motion, validated, returns = [], [], []
    for a in intervals.get("action",[]):
        e = action_ends[(a["step"],a["finish"])]
        if e.get("action") in ("observe","record") or e.get("error_type")=="ValueError" and a["seconds"] < .1:
            continue
        # Original wrapper instruments backend.move directly, including real failed moves.
        if not (isinstance(a["step"],str) or e.get("action") in ("approach","move") or e.get("purpose")=="move"):
            continue
        motion.append((a["start"],a["finish"]))
        if a["ok"]:
            validated.append(a["start"])
        if str(a["step"]).endswith("-return"):
            returns.append((a["start"],a["finish"]))
    intersect = [(max(a,c),min(b,d)) for a,b in transport for c,d in merged(motion) if max(a,c)<min(b,d)]
    successes = receipt["strict"] is True and cov.get("complete") is True and result["score"].get("correct") is True
    times = sorted(c["return_monotonic_seconds"]-c["dispatched_monotonic_seconds"] for c in calls)
    world = unquote((ep/"inspection.world").read_text()).replace(str(ep.resolve()),"{episode}")
    textures = [(f.name,digest(f)) for f in sorted((ep/"assets/materials/textures").glob("*.png"))]
    origin = events[0]["monotonic_seconds"] if events else None
    return dict(case=case["id"],method=ep.name,strict=successes,legacy=result["score"]["outcome"],
        whole_s=receipt["seconds"],restricted_s=receipt["seconds"] if successes else result["episode_timeout_seconds"],
        requests=len(calls),request_union_s=length(transport),motion_command_s=length(motion),
        transport_with_motion_s=length(intersect),transport_without_motion_s=length(transport)-length(intersect),
        first_completed_motion_start_s=min(validated)-origin if validated else None,return_move_s=length(returns),
        call_p50_s=median(times) if times else None,call_max_s=max(times,default=None),
        offers=sum(e["event"]=="native_offer" for e in events),branches_with_options=sum(bool(e.get("options")) for e in events if e["event"]=="native_policy_checked"),
        early_starts=sum(e["event"]=="native_early_start" for e in events),
        early_saved=sum(x["early"] and bool(x.get("raw_bound")) for x in proofs),
        early_handoffs=sum(bool(x.get("proven_early_handoff")) for x in proofs),
        early_final_citations=sum(x["early"] and bool(x.get("final_report_cites")) for x in proofs),
        early_evidence_support=sum(x["early"] and bool(x.get("independent_field_support")) for x in proofs),
        field_grant_violation_count=len(violations),receipt_sha256=receipt["result_sha256"],
        world_sha256=sha256(world.encode()).hexdigest(),texture_sha256=sha256(json.dumps(textures).encode()).hexdigest(),
        task_sha256=sha256(task.encode()).hexdigest(),source_sha256=receipt["source_sha256"]), proofs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=Path("config/native_views_dev01.json"))
    parser.add_argument("--admit-comparators",action="store_true")
    args = parser.parse_args()
    config = read(args.config)
    root = Path("outputs/continuing")/config["id"]
    dest = root/"audit"
    dest.mkdir(parents=True,exist_ok=True)
    runs, proofs = [], []
    for case in config["cases"]:
        for arm in case["primary"]+case["comparators"]:
            receipt = root/case["id"]/(arm+".receipt.json")
            if not receipt.exists() or read(receipt).get("state")!="scored":
                continue
            run, photos = audit(receipt.with_name(arm),read(receipt),case)
            runs.append(run)
            proofs.extend(photos)
    for case in config["cases"]:
        subset=[r for r in runs if r["case"]==case["id"]]
        if len({(r["world_sha256"],r["texture_sha256"],r["task_sha256"]) for r in subset})>1:
            raise ValueError("Unpaired assets or instructions: "+case["id"])
    if runs:
        write_table(dest/"episodes.csv",runs)
    save(dest/"photo_chains.json",proofs)
    save(dest/"summary.json",dict(scored=len(runs),sends=sum(r["requests"] for r in runs),
        successes=sum(r["strict"] for r in runs),early_handoffs=sum(r["early_handoffs"] for r in runs),
        final_early_citations=sum(r["early_final_citations"] for r in runs),
        verified_model_and_photos=True, token_counts=None,energy_joules=None,
        note="Transport without logged motion is a descriptive blocking proxy. Handoff alone does not imply a benefit or a verified final fact."))
    if args.admit_comparators:
        primary={(c["id"],m) for c in config["cases"] for m in c["primary"]}
        completed={(r["case"],r["method"]) for r in runs}
        admissible=[p for p in proofs if p.get("proven_early_handoff") and p["method"] in ("native-deferred","evidence-deferred")]
        if not primary<=completed or not admissible or any(r["field_grant_violation_count"] for r in runs):
            raise ValueError("Primary comparisons or independent mechanism evidence missing")
        endpoint_errors=[read(root/c/(m+".receipt.json")).get("endpoint_block") for c,m in primary]
        if any(endpoint_errors):
            raise ValueError("Primary block has an endpoint error")
        gate=dict(admit=True,config_sha256=digest(args.config),
                  reason="Complete primary block with actual nomination, permitted motion, source-bound RGB and subsequent ordinary request. The house-independent inventory task with zero nominations remains as the negative opportunity control.",
                  proven_photos=[p["sha256"] for p in admissible],
                  primary_result_sha256=sorted(r["receipt_sha256"] for r in runs if (r["case"],r["method"]) in primary))
        frozen=root/"comparator_gate.json"
        if frozen.exists() and read(frozen)!=gate:
            raise ValueError("Comparator admission evidence changed")
        save(frozen,gate)
    for r in runs:
        print(r["case"],r["method"],r["strict"],round(r["whole_s"],2),r["requests"],r["early_handoffs"],r["early_final_citations"])


if __name__ == "__main__":
    main()
