"""Verify cross-camera batching on two exposed view pairs with fixed inputs and full receipts."""
import argparse
from hashlib import sha256
from pathlib import Path
from statistics import median

from experiments.flexible_execution.public_audit import _grade_segmentation
from experiments.flexible_execution.revision_audit import digest, read, save
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_crossworld_views import assert_snapshot
from scripts.continuing_paired_rgb import PROMPT, predict, slots

CFG = Path("config/cross_source_assay_dev02.json")
ROOT = Path("outputs/continuing/cross-source-assay-dev02")


def register():
    config = read(CFG)
    rows = []
    for case in config["cases"]:
        folder = Path(case["source"])
        timeline = folder / "timeline.jsonl"
        frames = {e["step"]: e for e in read_trace_rows(timeline)[0] if e["event"] == "camera_observation"}
        hashes = {v: digest(folder / (v + ".jpg")) for v in ("front", "side")}
        if (any(frames[v]["image_sha256"] != h for v, h in hashes.items())
                or frames["front"]["monotonic_seconds"] >= frames["side"]["monotonic_seconds"]
                or len(set(hashes.values())) != 2):
            raise ValueError("Recorded camera bind failed")
        rows.append({**case, "image_sha256": hashes, "timeline_sha256": digest(timeline)})
    ROOT.mkdir(exist_ok=True)
    reg = dict(config_sha256=digest(CFG), prompt_sha256=sha256(PROMPT.encode()).hexdigest(),
               runner_sha256=digest(Path(__file__)), shared_transport_sha256=digest(Path("scripts/continuing_paired_rgb.py")),
               cases=rows)
    assert_snapshot(ROOT / "inputs.json", reg)
    return config, reg


def grade(reg):
    ledger = read_trace_rows(ROOT / "sends.jsonl")[0]
    rows, hashes = {}, {}
    for case, mode, slot in slots(reg):
        folder = ROOT / slot
        log, receipt = folder / "model_calls.jsonl", folder / "receipt.json"
        c, = read_trace_rows(log)[0]
        rec = read(receipt)
        names = ["front", "side"] if mode == "joint" else [mode]
        images = [case["image_sha256"][v] for v in names]
        expected = dict(items=[dict(id=v, category=case["targets"][v]) for v in names])
        reserve = ledger[c["logical_job_attempt"] - 1]
        if not (rec["sent"] and c["input"] == expected and c["image_sha256"] == images == reserve["image_sha256"]
                and reserve["reservation"] == c["logical_job_attempt"] == len(hashes) + 1
                and c["model"] == "gpt-6.1-terra" and c["reasoning_effort"] == "max"
                and c["instructions_sha256"] == reg["prompt_sha256"] and rec["raw_sha256"] == digest(log)):
            raise ValueError("Request no longer matches its registered result: " + slot)
        rows.setdefault(case["house"], {})[mode] = rec
        hashes[slot] = dict(log=digest(log), receipt=digest(receipt))
    if len(hashes) != len(ledger):
        raise ValueError("Unexpected real dispatch")
    assert_snapshot(ROOT / "sealed.json", hashes)
    cases = []
    for case in reg["cases"]:
        scoring = Path(case["source"]) / "scoring"
        for source, key in case["score_fields"].items():
            if read(scoring / (key + ".json"))["category"] != case["targets"][source]:
                raise ValueError("Referee and public category differ")
        expected = {v: _grade_segmentation(scoring, k) for v, k in case["score_fields"].items()}
        outputs = rows[case["house"]]
        correct = {mode: {key: (rec["answer"] or {}).get(key) == expected[key]
                          for key in (expected if mode == "joint" else [mode])} for mode, rec in outputs.items()}
        j, f, s = [outputs[m] for m in ("joint", "front", "side")]
        complete = all(row["valid"] for row in outputs.values())
        delivered = complete and all(v in ("positive", "negative") for row in outputs.values() for v in row["answer"].values())
        cases.append(dict(house=case["house"], expected=expected, observations=outputs, correct=correct,
            image_sha256=case["image_sha256"], both_correct_joint=all(correct["joint"].values()),
            both_correct_separate=correct["front"]["front"] and correct["side"]["side"],
            paired_answers_delivered=delivered, service_delta_s=f["service_s"] + s["service_s"] - j["service_s"] if delivered else None))
    deltas = [c["service_delta_s"] for c in cases if c["service_delta_s"] is not None]
    result = dict(actual_sends=len(ledger), seal_sha256=digest(ROOT/"sealed.json"), cases=cases,
        timeouts_joint=sum(c["observations"]["joint"]["error"]=="timeout" for c in cases),
        timeouts_separate=sum(c["observations"][v]["error"]=="timeout" for c in cases for v in ("front","side")),
        both_correct_joint=sum(c["both_correct_joint"] for c in cases),
        both_correct_separate=sum(c["both_correct_separate"] for c in cases),
        matched_delivery_pairs=len(deltas), paired_service_delta_median_s=median(deltas) if deltas else None)
    save(ROOT / "summary.json", result)
    print({key: value for key, value in result.items() if key != "cases"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("register", "run", "grade", "status"))
    mode = parser.parse_args().mode
    config, reg = register()
    if mode in ("register", "status"):
        print("Preserved receipts", len(list(ROOT.glob("*/receipt.json"))), "of", config["max_total_sends"])
    else:
        if mode == "run":
            predict(config, ROOT, reg)
        grade(reg)

