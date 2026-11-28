"""Recompute the shared fixes on original replies and frozen photo/identity chains."""
import json
from collections import Counter
from pathlib import Path

from experiments.flexible_execution.adapters.v2_execution import compile_requirements
from experiments.flexible_execution.clean_coverage import audit_run
from experiments.flexible_execution.frontend_evidence import project
from experiments.flexible_execution.requirement_program import parse
from experiments.flexible_execution.revision_audit import BASE, OUT, digest, read, save, write_table
from experiments.flexible_execution.timeline import read_trace_rows


class SavedResponse:
    def __init__(self, call):
        self.call = call

    def request(self, instructions, payload, photos):
        if photos or payload != self.call["input"]:
            raise ValueError("The replay requested different input")
        return self.call["output"]


def compile_replies():
    repaired = []
    registry = read(BASE / "formal-progress/compiler-branch-replays.json")
    for r in registry["attempts"]:
        p = BASE / "custom-final" / r["case"] / r["method"] / ("attempt-%03d" % r["attempt"])
        path = p / "episode/private/robot/model_calls.jsonl"
        assert digest(path) == r["raw_log_sha256"]
        calls, _ = read_trace_rows(path, allow_truncated=True)
        call = next(c for c in calls if c["logical_job_attempt"] == r["logged_request"])
        program, intent = compile_requirements(SavedResponse(call), call["input"]["authorized_task"])
        assert intent is None and program.fields == r["local_replay_with_null_conditional"]["typed_fields"]
        repaired.append(dict(case=r["case"], method=r["method"], original_reply_sha256=digest(path),
                             program_fields=program.fields))
    return repaired


def project_camera():
    old = read(BASE / "formal-progress/frontend-question-172.json")
    job = read(BASE / "custom-final" / old["case"] / old["method"] / "job.json")
    ep = Path(job["artifact_dir"])
    robot = ep / "private/robot"
    assert all(digest(Path(p)) == h for p, h in old["log_sha256"].items())
    ev, _ = read_trace_rows(robot / "timeline.jsonl")
    handoff = next(e for e in ev if e["event"] == "full_inspection_started")
    instruction = read(ep / "clean_task_scope.json")["authorized_task"]
    program = parse(read(robot / "typed_task_program.json"), instruction)
    result, provenance = project(robot, program, instruction, Path(job["request_ledger"]),
                                model=job["model"], effort=job["reasoning_effort"],
                                cutoff=handoff["evidence_cutoff_monotonic"], expires=handoff["episode_deadline_monotonic"])
    assert result == old["projection_with_only_prefix_normalized"] and provenance == old["bound_evidence"]
    return dict(original_state=job["state"], original_records=handoff["seeded_records"],
                fresh_projection=result, verified_sources=provenance)


def pilot():
    rows = []
    for path in sorted(Path("outputs/natural-pilot").glob("pair-*/paired-serial/attempt-*/episode/result.json")):
        source = read(path)
        if source.get("policy") != "clean":
            continue
        score = audit_run(path.parent) if isinstance(source.get("report"), dict) else {}
        rows.append(dict(path=str(path), family=source["family"], seed=source["seed"],
                         old=source["score"]["outcome"], strict=score.get("complete"),
                         missing=str(score.get("missing_fields_by_object", "report unavailable")), source_sha256=digest(path)))
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cases = compile_replies()
    photo = project_camera()
    pilot_rows = pilot()
    write_table(OUT / "augmented_serial_regraded.csv", pilot_rows)
    save(OUT / "shared_fixes.json", dict(new_requests=0, parser_replays=cases, evidence_projection=photo,
                                         augmented_serial=dict(n=len(pilot_rows), families=len({x["family"] for x in pilot_rows}),
                                         old=Counter(x["old"] for x in pilot_rows), strict_regradable=sum(x["strict"] is not None for x in pilot_rows),
                                         strict=sum(x["strict"] is True for x in pilot_rows))))
    print("Recovered", len(cases), "compiler replies and", len(photo["fresh_projection"]), "bound fact.")
    print("Augmented serial controls:", len(pilot_rows), "old correct:", sum(x["old"]=="correct" for x in pilot_rows),
          "strict:", sum(x["strict"] is True for x in pilot_rows))
