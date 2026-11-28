"""Fresh, resumable CPU development blocks. Status is available while a worker runs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
from hashlib import sha256

from experiments.task_attack.model import EXTERNAL_RESPONSE_CODES
from .revision_audit import digest, read, save
from .revision_jobs import observed, source_id, worker_active
from .timeline import read_trace_rows


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime_fingerprint(config):
    base=source_id()
    extra=config.get("runtime_files",[])
    if not extra:
        return base
    h=sha256(bytes.fromhex(base))
    for name in sorted(extra):
        path=Path(name)
        h.update(name.encode()+b"\0"+path.read_bytes())
    return h.hexdigest()


def planned(config, block):
    for case in config["cases"]:
        for arm in case[block]:
            yield case,arm


def command(config,case,arm,ep):
    args=["bash","scripts/run_flexible_clean.sh","--method",arm,"--backend","cpu","--family",case["family"],
          "--seed",str(case["seed"]),"--layout-seed",str(case["layout_seed"]),"--source-profile",case["source_profile"],
          "--equipment-count",str(case["count"]),"--episode-timeout",str(config["episode_s"]),
          "--max-request-attempts",str(config["max_requests_per_episode"]),"--onboard-evidence-only",
          "--ros-port","11774","--gazebo-port","11775","--chatgpt-project","/home/tdszy/uav/Diff-VLA","--output",str(ep)]
    if case.get("early_fields"):
        args+=["--early-fields",case["early_fields"]]
    if case.get("collection_rule"):
        args+=["--collection-rule",case["collection_rule"]]
    if arm!="original-gpt":
        args+=["--cost-config",config["cost_config"],"--execution-revision",config["id"]]
    return args


def call_errors(ep):
    f=ep/"private/robot/model_calls.jsonl"
    if not f.exists():
        return []
    return [(r.get("error") or {}).get("code") for r in read_trace_rows(f,allow_truncated=True)[0]
            if r.get("error")]


def receipt_path(root,case,arm):
    return root/case["id"]/(arm+".receipt.json")


def unresolved_service_limits(parent):
    limits=[]
    for p in parent.glob("*/*/*.receipt.json"):
        d=read(p)
        if not d.get("endpoint_block"):
            continue
        key=digest(p)
        markers=(p.parents[1]/"recovery").glob("restoration-*.json")
        if not any(key in read(m).get("interrupted_receipt_sha256",[]) for m in markers):
            limits.append((str(p),d["endpoint_block"]))
    return limits


def snapshot(root,config):
    rows=[]
    for group in ("primary","comparators"):
        for case,arm in planned(config,group):
            receipt=receipt_path(root,case,arm)
            ep=root/case["id"]/arm
            record=read(receipt) if receipt.exists() else {}
            try:
                actual=observed(ep)
            except (OSError,json.JSONDecodeError):
                actual=dict(state="unfinished",sends=None)
            if record.get("state")=="scored" and (actual.get("result_sha256")!=record.get("result_sha256")
                      or actual.get("strict")!=record.get("strict") or actual.get("sends")!=record.get("sends")):
                raise ValueError("Settled outcome differs from its source: "+str(receipt))
            state=record.get("state",actual["state"] if ep.exists() else "queued")
            rows.append(dict(block=group,case=case["id"],method=arm,state=state,sends=actual["sends"],
                             strict=actual.get("strict"),seconds=actual.get("seconds")))
    return rows


def settle(ep,receipt,meta,exit_code):
    data=observed(ep)
    errors=call_errors(ep)
    if data["state"]!="scored":
        data["state"]="interrupted-after-dispatch" if data["sends"] else "startup-failure"
    ext=next((e for e in errors if e in EXTERNAL_RESPONSE_CODES),None)
    out={**meta,**data,"errors":errors,"endpoint_block":ext,"end_utc":now(),"exit_code":exit_code}
    save(receipt,out)
    return out


def run(config_path,block,limit=None):
    config=read(config_path)
    root=Path("outputs/continuing")/config["id"]
    root.mkdir(parents=True,exist_ok=True)
    guard=root.parent/"online.lock"
    with guard.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        limits=unresolved_service_limits(root.parent)
        if limits:
            raise SystemExit("Existing plan limit; use the verified recovery entry after service restoration: "+str(limits))
        launched=0
        for case,arm in planned(config,block):
            ep=root/case["id"]/arm
            receipt=receipt_path(root,case,arm)
            previous=read(receipt) if receipt.exists() else {}
            if previous.get("endpoint_block"):
                raise SystemExit("Existing endpoint block: "+previous["endpoint_block"])
            if previous.get("state") in ("scored","interrupted-after-dispatch","closed-failure"):
                continue
            if previous.get("state")=="startup-failure":
                raise SystemExit("Inspect preserved startup failure before retry: "+str(ep))
            if ep.exists():
                while worker_active(ep):
                    save(root/"heartbeat.json",dict(utc=now(),state="adopting-worker",case=case["id"],method=arm))
                    time.sleep(5)
                result=settle(ep,receipt,previous,None)
                if result["endpoint_block"] or result["state"]=="startup-failure":
                    raise SystemExit("Recovered a blocked attempt: "+str(ep))
                continue
            if limit is not None and launched>=limit:
                break
            if block=="comparators":
                gate=root/"comparator_gate.json"
                if not gate.exists() or read(gate).get("admit") is not True:
                    raise SystemExit("Comparator gate has not been established from observed material use")
            receipt.parent.mkdir(parents=True,exist_ok=True)
            fingerprint=runtime_fingerprint(config)
            if config.get("runtime_sha256",fingerprint)!=fingerprint:
                raise SystemExit("Registered runtime differs from the files on disk; no new request made.")
            argv=command(config,case,arm,ep)
            meta=dict(case=case["id"],method=arm,block=block,config_sha256=digest(config_path),
                      git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                      source_sha256=fingerprint,controller_pid=os.getpid(),start_utc=now(),command=argv)
            save(receipt,{**meta,"state":"starting"})
            with receipt.with_name(arm+".console.log").open("a") as log:
                proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
                save(receipt,{**meta,"state":"running","worker_pid":proc.pid})
                while proc.poll() is None:
                    ledger=ep/"request_attempts.jsonl"
                    sends=len(ledger.read_bytes().splitlines()) if ledger.exists() else 0
                    save(root/"heartbeat.json",dict(utc=now(),state="running",controller_pid=os.getpid(),
                         worker_pid=proc.pid,case=case["id"],method=arm,sends=sends))
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            result=settle(ep,receipt,meta,proc.returncode)
            launched+=1
            print(case["id"],arm,result["state"],result.get("strict"),result["sends"],flush=True)
            save(root/"heartbeat.json",dict(utc=now(),state="between-jobs",last_receipt=str(receipt)))
            if result["endpoint_block"] or result["state"]!="scored":
                raise SystemExit("Study dispatch paused after an unscored or externally blocked attempt: "+str(receipt))
        save(root/"heartbeat.json",dict(utc=now(),state="idle",workers=[],rows=snapshot(root,config)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=Path("config/native_views_dev01.json"))
    parser.add_argument("--block",choices=("primary","comparators"),default="primary")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    cfg=read(args.config)
    if args.run:
        run(args.config,args.block,args.limit)
    else:
        for row in snapshot(Path("outputs/continuing")/cfg["id"],cfg):
            print(json.dumps(row))


if __name__=="__main__":
    main()
