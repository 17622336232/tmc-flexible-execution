"""Read-only historical audit and independent single-view reference calculation.

Run from the repository root. New products go under outputs/revision-study only.
"""
import csv
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

from .clean_coverage import audit_run
from .timeline import read_trace_rows

BASE = Path("outputs/protocol-v2-optimization")
OUT = Path("outputs/revision-study")
ORIGINAL = Path("outputs/existing-gpt-pilot")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_table(path, rows):
    with path.open("w", encoding="utf8", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)


def save(path, obj):
    with NamedTemporaryFile("w", dir=path.parent, encoding="utf8", delete=False) as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
    Path(tmp.name).replace(path)


def one_view_reference(contract, start):
    """Enumerate all saved legal, one-view, single-query continuations.

    Uniform branches and perfect readability are design assumptions. The
    resulting gain is a forecast over this finite support, never an observed gain.
    """
    t = contract["task"]
    if len(t["queries"]) != 1 or start is None:
        return None
    q = t["queries"][0]
    travel = {(a, b): v for a, b, v in t["travel"]}
    def trip(a, b):
        return 0 if a == b else travel[a, b]
    worlds = q["outcomes"]
    candidates = [None] + [s for s in t["skills"] if s["permission"] is None]
    costs = []
    for before in candidates:
        site = before["site"] if before else t["start"]
        taken = set(before["covers"]) if before else set()
        ready = start + (trip(t["start"], site) + before["seconds"] if before else 0)
        times = []
        for outcome in worlds:
            need = {r["id"] for r in t["requirements"] if not r["when"] or outcome in r["when"]["outcomes"]} - taken
            after = [s for s in t["skills"] if need <= set(s["covers"]) and
                     (s["permission"] is None or outcome in s["permission"]["outcomes"])]
            tail = trip(site, t["terminal"]) if not need else min(
                (trip(site, s["site"]) + s["seconds"] + trip(s["site"], t["terminal"]) for s in after),
                default=float("inf"))
            if max(ready, q["latest_ready"]) + tail + t["report_seconds"] > t["deadline"]:
                break
            times.append(max(ready, q["estimated_ready"]) + tail + t["report_seconds"])
        costs.append(sum(times) / len(times) if len(times) == len(worlds) else None)
    if costs[0] is None:
        return None
    return max([0.] + [costs[0] - c for c in costs[1:] if c is not None])


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = []
    for domain in ("custom", "public"):
        directory = BASE / "reports" / (domain + "-final")
        integrity = read(BASE / "formal-progress/final-integrity.json")["domains"][domain]
        assert all(digest(directory / name) == fingerprint for name, fingerprint in integrity["report_hashes"].items())
        with (directory / "episodes.csv").open() as stream:
            raw.extend(dict(row, audit_domain=domain) for row in csv.DictReader(stream))
    assert len(raw) == 456
    permissions = {x["case"]: x for x in read(BASE / "formal-progress/registered-permissions.json")["per_scene"]}
    witnessed = {x["case"]: next(y for y in x["episodes"] if y["method"] == "conditional-a")
                 for x in read(BASE / "formal-progress/registered-early-results.json")["cases"]}
    rows = []
    for x in raw:
        if x["audit_domain"] != "custom" or x["method"] != "conditional-a":
            continue
        robot = Path(x["artifact_dir"]) / "private/robot"
        events, tail = read_trace_rows(robot / "timeline.jsonl", allow_truncated=True)
        submitted = {e["step"]: e["binding"]["submitted_at"] for e in events if e["event"] == "query_submitted"}
        grants = permissions[x["case"]]["explicit_early_allowed_fields"]
        refs, physical, legal_skills = [], False, 0
        for p in sorted(robot.glob("conditional_contract_*.json")):
            step = int(p.stem.rsplit("_", 1)[1])
            if step not in submitted:
                continue
            contract = read(p)
            t = contract["task"]
            conditional = {r["id"] for r in t["requirements"] if r.get("when")
                           and set(r["when"]["outcomes"]) != set(t["queries"][0]["outcomes"])}
            physical |= bool(conditional)
            legal_skills += sum(bool(conditional.intersection(s["covers"])) and s["permission"] is None for s in t["skills"])
            gain = one_view_reference(contract, submitted[step])
            if gain is not None:
                refs.append(gain)
        actions = witnessed.get(x["case"], {}).get("early_action_witnesses", [])
        approved = [a for a in actions if a["photo_material_verified"]]
        taken = [a for a in actions if (a["photo_seconds_before_reply"] or 0) > 0]
        rows.append(dict(case=x["case"], family=x["family"], stratum=x["stratum"], source_group=x["group"],
                         submitted_queries=len(submitted), bound_judgments=x["judgment_condition_verified_judgments"],
                         branch_dependent_material_on_logged_contract=physical if submitted else "unobserved",
                         explicit_early_fields="|".join(grants), mapped_legal_conditional_skills=legal_skills,
                         independent_one_view_gain_forecast_s=max(refs) if refs else "",
                         admissible_conditional_candidate_logged=any(e["event"] == "candidate_generated" and
                             e.get("permission") == "all_pending_outcomes" for e in events) if grants else False,
                         early_starts=len(actions), early_photos=len(taken),
                         verified_photos_from_early_starts=len(approved),
                         verified_photos_finished_before_reply=sum(a in approved for a in taken),
                         early_material_report_uses=x["conditional_early_materials_used_in_report"],
                         blocking_saved_s="unidentified", causal_whole_task_effect="unidentified",
                         actual_success=x["success"], actual_seconds=x["continuous_task_seconds"],
                         final_log_line_truncated=tail, timeline_sha256=digest(robot / "timeline.jsonl")))
    assert len(rows) == 72
    write_table(OUT / "opportunities.csv", rows)
    legacy = []
    for p in sorted(ORIGINAL.glob("*/result.json")):
        preflight = p.parent / "clean_preflight.json"
        if not preflight.is_file() or read(preflight)["mode"] != "ACTUAL_GPT_CLEAN_SERIAL":
            continue
        d = read(p)
        strict = audit_run(p.parent)
        legacy.append(dict(run=p.parent.name, source=str(p), family=d["family"], seed=d["seed"],
                           limit_s=d["episode_timeout_seconds"], old=d["score"]["outcome"],
                           strict=strict["complete"], objects=strict["audited_objects"],
                           missing=strict["missing_fields_by_object"], hash=digest(p)))
    write_table(OUT / "original_regraded.csv", legacy)
    counts = [len(rows), sum(bool(r["submitted_queries"]) for r in rows),
              sum(r["branch_dependent_material_on_logged_contract"] is True for r in rows),
              sum(bool(r["explicit_early_fields"]) and r["branch_dependent_material_on_logged_contract"] is True for r in rows),
              sum(bool(r["mapped_legal_conditional_skills"]) and bool(r["explicit_early_fields"]) for r in rows),
              sum(bool(r["early_starts"]) for r in rows), sum(bool(r["early_photos"]) for r in rows),
              sum(bool(r["verified_photos_finished_before_reply"]) for r in rows),
              sum(int(r["early_material_report_uses"] or 0)>0 for r in rows)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["All A scenes", "Query submitted", "Branch-specific material", "Explicit grant", "Mapped legal view",
              "Movement before reply", "Photo before reply", "Early photo verified", "Early material in report"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.barh(labels, counts, color="#327278")
    ax.bar_label(bars, padding=4)
    ax.invert_yaxis()
    ax.set(xlabel="Historical scenes (72 total)", xlim=(0, 80))
    fig.tight_layout()
    fig.savefig(OUT / "opportunities.png", dpi=160)
    plt.close(fig)
    planner = [r for r in raw if r["audit_domain"]=="custom" and r["method"]=="conditional-planner"]
    ceiling = sum(float(r["restricted_seconds"]) for r in planner if permissions[r["case"]]["explicit_early_allowed_fields"])
    denom = sum(float(r["restricted_seconds"]) for r in planner)
    summary = dict(formal=Counter(r["state"] for r in raw), sends=sum(int(r["actual_dispatches"]) for r in raw),
                   causes=Counter(r.get("failure_reason") or r.get("state") for r in raw), funnel=dict(zip(labels,counts)),
                   legacy=legacy, forecasts_on_granted={r["case"]:r["independent_one_view_gain_forecast_s"]
                       for r in rows if r["explicit_early_fields"]},
                   registered_conditional_only_planner_upper_bound=ceiling/denom,
                   numerator_s=ceiling, denominator_s=denom, new_requests=0)
    save(OUT / "history.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    audit()
