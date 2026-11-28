"""Independently check the saved, unaccepted final-round A draft. Never edit the episode."""
from experiments.task_attack.evaluate import evaluate
from experiments.flexible_execution.revision_audit import OUT, digest, read, save
from experiments.flexible_execution.timeline import read_trace_rows


def main():
    ep=OUT/"round3/dev-cracks-prim/conditional-a"
    robot=ep/"private/robot"
    calls,_=read_trace_rows(robot/"model_calls.jsonl")
    checks=[c for c in calls if "proposed_report" in c.get("input",{})]
    assert len(checks)==2 and all(c["error"]["code"]=="timeout" for c in checks)
    assert checks[0]["input"]==checks[1]["input"] and checks[0]["image_sha256"]==checks[1]["image_sha256"]
    proposal=checks[0]["input"]["proposed_report"]
    verdict=evaluate(read(ep/"private/truth.json"),proposal,merge_fragments=True)
    materials=read(robot/"conditional_result.json")["accepted_materials"]
    identities=[r for r in materials.values() if r["field"]=="identity"]
    for identity in identities:
        assert digest(robot/(identity["image_id"]+".jpg"))==identity["frame_hash"]
    handoff=[e for e in read_trace_rows(robot/"timeline.jsonl")[0] if e["event"]=="full_inspection_started"]
    diag=dict(role="after-the-fact draft audit only; unchanged terminal result",expected=verdict["expected_ids"],
              claimed=sorted({r["object_id"] for r in proposal["records"]}),recognized_report_ids=verdict["reported_ids"],
              draft_correct=verdict["correct"],missing=verdict["missing_ids"],
              unexpected=verdict["unexpected_ids"],accepted_material_values=[r["value"] for r in identities],
              accepted_photo_digests=[r["frame_hash"] for r in identities],seeded_records=handoff[0]["seeded_records"],
              verifier_durations=[c["wall_seconds"] for c in checks],verifier_errors=[c["error"]["code"] for c in checks],
              actual_raw_result_sha256=digest(ep/"result.json"),source_log_sha256=digest(robot/"model_calls.jsonl"))
    # Inspected offline alongside the simulator's recorded identity. The texture
    # was never supplied to the agent as a separate view.
    texture=ep/"assets/materials/textures/serial_01.png"
    diag["offline_asset_texture_sha256"]=digest(texture)
    diag["offline_asset_texture_path"]=str(texture.relative_to(OUT/"round3"))
    save(OUT/"round3/independent_draft_check.json",diag)
    print(diag)


if __name__=="__main__":
    main()
