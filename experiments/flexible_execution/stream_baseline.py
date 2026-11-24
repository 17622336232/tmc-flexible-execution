"""Client-side segmented generation for the simulated inspection baseline."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from math import isfinite
from time import monotonic, sleep

from experiments.task_attack.agent import ModelDecisionInvalid, SYSTEM


NAVIGATION = {"move", "approach"}
ACTION_PACKET = SYSTEM + """
For this controller, emit a complete first JSON packet. If the next action is
move or approach, put the fully specified action and brief reason in that packet
and leave visual records and memory to the continuation packet. Choose a move
only when the current photograph and measured geometry allow you to propose it.
If the next action is record, observe, or submit, use the full response format
above. The rest of the task still requires fresh camera data where appropriate.
"""
CONTINUATION_PACKET = SYSTEM + """
The first, complete plan packet already proposed one movement from this exact
image. The input shows the proposed action, not its outcome. On the same frozen
photograph, document only facts visibly supported now and prepare the next
inspection after the movement. Make that plan conditional on the eventual tool
feedback and next photo. Do not guess where the robot arrived or what another
view will show. This packet may not authorize a new action or final report.
Return JSON only with keys memory (string), records (array in the above record
format), and next_view_check (string describing the next visual work).
"""


def _finite(terms):
    try:
        return all(type(term) in (int, float) and isfinite(term) for term in terms)
    except OverflowError:
        return False


def _action(first):
    action = first.get("action") if isinstance(first, dict) else None
    kind = action.get("type") if isinstance(action, dict) else None
    if kind not in ("move", "approach", "record", "observe", "submit"):
        raise ModelDecisionInvalid("A complete action packet must declare a known task tool")
    if kind == "move":
        goal = action.get("goal")
        if not isinstance(goal, dict) or set(goal) != {"x", "y", "z", "yaw"} or not _finite(goal.values()):
            raise ModelDecisionInvalid("The movement packet must specify a finite camera pose")
    elif kind == "approach":
        box, standoff = action.get("bbox"), action.get("standoff_m", 2.)
        if (not isinstance(box, list) or len(box) != 4 or not _finite([*box, standoff])
                or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1 or not 1.2 <= standoff <= 3.):
            raise ModelDecisionInvalid("The approach packet must specify a valid visible region and stand-off")
    return action


def decide(model, context, images, *, execute, trace, step, image_sha256, stream):
    image_id = context["current_image_id"]
    with trace.measure("model_request", f"{step}-packet-one", purpose="segmented_action", image_id=image_id,
                       image_sha256=image_sha256):
        first = model.request(ACTION_PACKET, context, images)
    action = _action(first)
    if action["type"] not in NAVIGATION:
        return first, None
    trace.event("segmented_action_ready", step, action=action, image_id=image_id, image_sha256=image_sha256)
    pending_input = {**context, "segmented_navigation": {"first_action": action, "executed_pose_known": False}}

    def continuation():
        with trace.measure("model_request", f"{step}-packet-two", purpose="segmented_continuation", image_id=image_id,
                           image_sha256=image_sha256):
            return model.request(CONTINUATION_PACKET, pending_input, images)

    feedback = None
    if stream:
        previous_count = model.calls
        deadline = model.deadline_monotonic
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="inspection-plan-packet") as worker:
            pending = worker.submit(continuation)
            while model.calls == previous_count and not pending.done() and monotonic() < deadline:
                sleep(0.015)
            if model.calls > previous_count and not pending.done() and monotonic() < deadline:
                trace.event("segmented_action_released", step, action=action, frozen_image_id=image_id)
                feedback = execute(action)
            followup = pending.result()
    else:
        followup = continuation()
    if (not isinstance(followup, dict) or set(followup) != {"memory", "records", "next_view_check"}
            or not isinstance(followup["memory"], str) or not isinstance(followup["records"], list)
            or not isinstance(followup["next_view_check"], str)):
        raise ModelDecisionInvalid("The segmented continuation did not return grounded records and a next view check")
    return {"action": action, "reason": first.get("reason"), "records": followup["records"],
            "memory": followup["memory"] + "\nProvisional next view check: " + followup["next_view_check"]}, feedback


def audit(root):
    def rows(name):
        path = root / "private/robot" / name
        return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()] if path.is_file() else []

    events = rows("timeline.jsonl")
    calls = rows("model_calls.jsonl")
    starts = {row["step"]: row for row in events if row["event"] == "action_started"}
    finishes = {row["step"]: row for row in events if row["event"] == "action_finished"}
    action_prompt = sha256(ACTION_PACKET.encode()).hexdigest()
    continuation_prompt = sha256(CONTINUATION_PACKET.encode()).hexdigest()
    result = []
    for row in events:
        if row["event"] != "segmented_action_ready":
            continue
        step = row["step"]
        first = next((call for call in calls if call["instructions_sha256"] == action_prompt
                      and call["input"].get("current_image_id") == row["image_id"]
                      and call.get("output", {}).get("action") == row["action"]), None)
        following = [call for call in calls if call["instructions_sha256"] == continuation_prompt
                     and call["input"].get("current_image_id") == row["image_id"]
                     and "dispatched_monotonic_seconds" in call]
        action_start, action_end = starts.get(step), finishes.get(step)
        matches = bool(first and following and first["image_sha256"][-1] == row["image_sha256"]
                       and all(call["image_sha256"] == first["image_sha256"]
                               and call["input"]["segmented_navigation"] == {"first_action": row["action"],
                                   "executed_pose_known": False}
                               and first["return_monotonic_seconds"] <= call["dispatched_monotonic_seconds"]
                               for call in following))
        overlap = 0.
        if matches and action_start and action_end and action_end.get("ok") is True:
            start, end = action_start["monotonic_seconds"], action_end["monotonic_seconds"]
            overlap = sum(max(0., min(end, call["return_monotonic_seconds"])
                              - max(start, call["dispatched_monotonic_seconds"])) for call in following)
        result.append({"step": step, "matching_frozen_inputs": matches, "continuation_attempts": len(following),
                       "overlap_seconds": overlap})
    return {"navigation_packets": len(result), "moves_during_model_continuation": sum(r["overlap_seconds"] > 0 for r in result),
            "overlap_seconds": sum(r["overlap_seconds"] for r in result), "packets": result}
