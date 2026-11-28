"""Paired semantic service assay on exposed camera frames. No simulator or task success score."""
import argparse
import fcntl
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import monotonic

from experiments.flexible_execution.revision_audit import digest, read, save, write_table
from experiments.flexible_execution.request_budget import RequestBudget
from experiments.flexible_execution.timeline import read_trace_rows
from experiments.task_attack.model import ModelClient, EXTERNAL_RESPONSE_CODES


CONFIG = Path("config/paired_rgb_assay01.json")
PROMPT = """Visually inspect each attached RGB photograph independently. For each item, decide whether
at least one instance of its requested class is visibly present in the matching photograph.
Answer positive for visible presence, negative if none is identifiable, uncertain if you cannot judge.
Use only the corresponding pixels. Return exactly one JSON object mapping each supplied item id to
its outcome. Report no other properties. Reading both images when both are supplied is authorized."""
LABELS = {"positive","negative","uncertain"}


def register(config,root):
    frozen = root/"inputs.json"
    if frozen.exists():
        doc = read(frozen)
        if doc["config_sha256"] != digest(CONFIG) or doc["prompt_sha256"] != sha256(PROMPT.encode()).hexdigest():
            raise ValueError("Configuration changed after registration")
        return doc
    pool = Path(config["input_pool"])
    houses = sorted(pool.glob("procthor-test-house-*-view-*"),
                    key=lambda p:sha256((config["salt"]+p.name).encode()).digest())
    chosen, seen = [], set()
    for house in houses:
        candidates = sorted(p.parent for p in house.glob("*/attempt-*/episode/side.jpg")
                            if (p.parent/"front.jpg").is_file())
        if not candidates:
            continue
        ep = candidates[0]
        public = read(ep/"source.json")
        if public["house_index"] in seen:
            continue
        seen.add(public["house_index"])
        case = read(ep.parents[1]/"job.json")["case"]
        chosen.append(dict(house=house.name,source=str(ep),
            targets={"front":case["expected_language_contract"]["primary"],"side":case["expected_language_contract"]["secondary"]},
            image_sha256={v:digest(ep/(v+".jpg")) for v in ("front","side")},order=config["order"][len(chosen)]))
        if len(chosen) == config["houses"]:
            break
    if len(chosen) != config["houses"]:
        raise ValueError("Insufficient previously acquired matched photographs")
    doc = dict(config_sha256=digest(CONFIG),prompt_sha256=sha256(PROMPT.encode()).hexdigest(),cases=chosen)
    save(frozen,doc)
    return doc


def slots(reg):
    for i,case in enumerate(reg["cases"]):
        for mode in case["order"]:
            yield case,mode,("%02d-"%i)+mode


def predict(config,root,reg):
    from uav_bridge.chatgpt_auth import CredentialStore
    parent_ledger = root/"sends.jsonl"
    budget = RequestBudget(parent_ledger,config["max_total_sends"])
    auth = CredentialStore(directory=Path("/home/tdszy/uav/Diff-VLA/.runtime/chatgpt"))
    with (root.parent/"online.lock").open("a+") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for case,mode,slot in slots(reg):
            ep = root/slot
            flags = ["front","side"] if mode=="joint" else [mode]
            if ep.exists():
                if not (ep/"receipt.json").exists():
                    raise SystemExit("Unsettled slot; inspect its original dispatch and retain failure: "+slot)
                receipt = read(ep/"receipt.json")
                if receipt.get("external_block"):
                    raise SystemExit("Endpoint block in preserved receipt: "+slot)
                continue
            ep.mkdir()
            payload,images,expected_hashes = [],[],[]
            for f in flags:
                path = Path(case["source"])/(f+".jpg")
                rgb = path.read_bytes()
                h = sha256(rgb).hexdigest()
                if h!=case["image_sha256"][f]:
                    raise ValueError("Camera image changed")
                payload.append(dict(id=f,category=case["targets"][f]))
                images.append(dict(label=f+" camera RGB",data=rgb))
                expected_hashes.append(h)
            client = ModelClient(ep/"model_calls.jsonl",timeout=config["per_call_seconds"],effort="max",
                                 credential_store=auth,request_budget=budget)
            begin = monotonic()
            error = None
            answer = None
            external = False
            try:
                answer = client.request(PROMPT,{"items":payload},images,attempt_limit=1)
            except Exception as ex:
                code = getattr(ex,"code",type(ex).__name__)
                error = str(code)
                external = code in EXTERNAL_RESPONSE_CODES or getattr(ex,"status",None) in (401,403,429)
            seconds = monotonic()-begin
            log = ep/"model_calls.jsonl"
            calls = read_trace_rows(log)[0] if log.exists() else []
            sent = bool(calls and calls[0].get("dispatched_monotonic_seconds") is not None)
            if not sent and error:
                external = True
            if sent and (len(calls)!=1 or calls[0]["image_sha256"]!=expected_hashes):
                raise ValueError("Input/transport ledger mismatch: "+slot)
            valid = isinstance(answer,dict) and set(answer)==set(flags) and all(isinstance(v,str) and v in LABELS for v in answer.values())
            save(ep/"receipt.json",dict(mode=mode,house=case["house"],sent=sent,valid=valid,answer=answer if valid else None,
                 error=error,external_block=external,wall_s=seconds,service_s=(calls[0]["return_monotonic_seconds"]-
                 calls[0]["dispatched_monotonic_seconds"]) if sent else None,raw_sha256=digest(log) if calls else None))
            if external:
                raise SystemExit("Endpoint or pre-dispatch failure: "+slot+" / "+str(error))
            print(slot, "response" if valid else "invalid",round(seconds,2),flush=True)


def grade(config,root,reg):
    from experiments.flexible_execution.public_audit import _grade_segmentation
    receipts = [root/slot/"receipt.json" for _,_,slot in slots(reg)]
    if not all(p.is_file() for p in receipts):
        raise ValueError("All prediction slots must be closed before opening scoring data")
    lock = root/"prediction_fingerprint.json"
    hashes = {str(p.relative_to(root)):digest(p) for p in receipts}
    if lock.exists() and read(lock)!=hashes:
        raise ValueError("Sealed prediction receipts changed")
    save(lock,hashes)
    by_case, records = {}, []
    ledger = read_trace_rows(root/"sends.jsonl")[0]
    sends = []
    for case,mode,slot in slots(reg):
        ep = root/slot
        row = read(ep/"receipt.json")
        if row["sent"]:
            path = ep/"model_calls.jsonl"
            if digest(path)!=row["raw_sha256"]:
                raise ValueError("Sealed response log changed")
            sends.extend(read_trace_rows(path)[0])
        fields = ("front","side") if mode=="joint" else (mode,)
        truths = {f:_grade_segmentation(Path(case["source"])/"scoring",f) for f in fields}
        correct = {f:(row["answer"] or {}).get(f)==truths[f] for f in fields}
        records.extend(dict(house=case["house"],mode=mode,field=f,image_sha256=case["image_sha256"][f],
                            category=case["targets"][f],expected=truths[f],outcome=(row["answer"] or {}).get(f),
                            correct=correct[f],service_s=row["service_s"],error=row["error"]) for f in fields)
        by_case.setdefault(case["house"],{})[mode]=(all(correct.values()),row["service_s"],row["valid"])
    assert [(c["logical_job_attempt"],c["image_sha256"]) for c in sends]==[(e["reservation"],e["image_sha256"]) for e in ledger]
    totals=[]
    for house,items in by_case.items():
        j,f,s=(items[k] for k in ("joint","front","side"))
        delta=f[1]+s[1]-j[1] if all(m[2] and m[1] is not None for m in (j,f,s)) else None
        totals.append(dict(house=house,joint_both_correct=j[0],separate_both_correct=f[0] and s[0],
                           joint_s=j[1],front_s=f[1],side_s=s[1],separate_minus_joint_service_s=delta))
    write_table(root/"per_frame.csv",records)
    write_table(root/"paired.csv",totals)
    times=[r["separate_minus_joint_service_s"] for r in totals if r["separate_minus_joint_service_s"] is not None]
    summary=dict(cases=len(totals),sends=len(ledger),joint_both_correct=sum(r["joint_both_correct"] for r in totals),
                 separate_both_correct=sum(r["separate_both_correct"] for r in totals),timed_valid_pairs=len(times),
                 median_separate_minus_joint_service_s=median(times) if times else None,tokens_recorded=None)
    save(root/"summary.json",summary)
    print(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=("register","predict","score"))
    args = parser.parse_args()
    config = read(CONFIG)
    root = Path("outputs/continuing")/config["id"]
    root.mkdir(parents=True,exist_ok=True)
    reg = register(config,root)
    if args.action=="predict":
        predict(config,root,reg)
    elif args.action=="score":
        grade(config,root,reg)
    else:
        print([(c["house"],c["targets"]) for c in reg["cases"]])


if __name__=="__main__":
    main()
