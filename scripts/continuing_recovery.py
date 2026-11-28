"""Reconcile terminal service interruptions and resume only previously unsent registered jobs."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.flexible_execution.continuing_jobs import command, planned, receipt_path, unresolved_service_limits
from experiments.flexible_execution.revision_audit import read,save,digest
from experiments.flexible_execution.timeline import read_trace_rows


def interruptions(root,config):
    failures=[]
    for group in ("primary","comparators"):
        for case,arm in planned(config,group):
            receipt=receipt_path(root,case,arm)
            if not receipt.exists():
                continue
            outcome=read(receipt)
            if not outcome.get("endpoint_block"):
                continue
            ep=root/case["id"]/arm
            calls=read_trace_rows(ep/"private/robot/model_calls.jsonl")[0]
            ledger=read_trace_rows(ep/"request_attempts.jsonl")[0]
            result=read(ep/"result.json")
            code=outcome["endpoint_block"]
            if (len(ledger)!=len(calls) or outcome["sends"]!=len(ledger) or
                    (result.get("error") or {}).get("code")!=code or
                    (calls[-1].get("error") or {}).get("code")!=code or result.get("report") is not None):
                raise ValueError("Endpoint failure differs across the raw receipt, service and result")
            for n,(call,entry) in enumerate(zip(calls,ledger),1):
                names=[x["image_id"] for x in call["input"]["recent_images"]]+[call["input"]["current_image_id"]]
                hashes=[digest(ep/"private/robot"/(name+".jpg")) for name in names]
                if not (entry["reservation"]==call["logical_job_attempt"]==n and hashes==entry["image_sha256"]==call["image_sha256"]):
                    raise ValueError("Interrupted attempt has mismatched camera/request provenance")
            failures.append(dict(cohort=config["id"],case=case["id"],method=arm,block=group,code=code,
                   actual_sends=len(ledger),valid_replies=sum(isinstance(c.get("output"),dict) for c in calls),
                   task_seconds=read(ep/"job_clock.json")["elapsed_seconds"],restricted_if_counted_s=config["episode_s"],
                   report_submitted=False,retry_after=None,receipt_sha256=digest(receipt),
                   model_log_sha256=digest(ep/"private/robot/model_calls.jsonl"),result_sha256=digest(ep/"result.json"),
                   end_utc=outcome["end_utc"]))
    return failures


def acknowledged(root,failures):
    covered=set()
    for p in sorted((root/"recovery").glob("restoration-*.json")):
        doc=read(p)
        if isinstance(doc.get("evidence"),str) and doc["evidence"].strip():
            covered.update(doc["interrupted_receipt_sha256"])
    return {f["receipt_sha256"] for f in failures}<=covered


def remaining(config,root,block):
    pending=[]
    for case,arm in planned(config,block):
        ep=root/case["id"]/arm
        receipt=receipt_path(root,case,arm)
        if receipt.exists() and read(receipt)["state"] in ("scored","interrupted-after-dispatch","closed-failure"):
            continue
        if ep.exists() or receipt.exists():
            raise ValueError("Inspect the still-running, orphan or startup-failed slot before a new online dispatch: "+str(ep))
        pending.append((case["id"],arm))
    return pending


def continuation(config,root,block,pending,config_hash):
    # The frozen runner can consume this suffix without rewriting its source or earlier receipts.
    # Assert equality of both worker commands in addition to method, seed, deadline and budget.
    new={**config,"parent_config_sha256":config_hash,"recovery_role":"unsent-registered-slots-only","cases":[]}
    for case in config["cases"]:
        reduced={**case}
        for b in ("primary","comparators"):
            reduced[b]=[m for m in case[b] if b==block and (case["id"],m) in pending]
        new["cases"].append(reduced)
        for method in reduced[block]:
            ep=root/case["id"]/method
            if command(config,case,method,ep)!=command(new,reduced,method,ep):
                raise ValueError("Worker command changed in the suffix manifest")
    return new


def frozen_runtime(config,config_path,root):
    shared=root.parent.parent.resolve()
    listed=subprocess.check_output(["git","worktree","list","--porcelain"],text=True)
    candidates=[Path.cwd(),*(Path(line[9:]) for line in listed.splitlines() if line.startswith("worktree "))]
    registered={read(p).get("git") for p in root.glob("*/*.receipt.json") if read(p).get("source_sha256")==config["runtime_sha256"]}
    # Prefer the actual known launch commit when a maintained copy exists.
    def pinned(p):
        rev=subprocess.run(["git","rev-parse","HEAD"],cwd=p,capture_output=True,text=True)
        return rev.returncode==0 and rev.stdout.strip() in registered
    possible=[p for p in dict.fromkeys(candidates) if p.is_dir() and (p/"outputs").resolve()==shared]
    for cwd in sorted(possible,key=lambda p:not pinned(p)):
        env={**os.environ,"PYTHONPATH":os.pathsep.join(map(str,(cwd/"bridge",cwd)))}
        source=subprocess.run([sys.executable,"-c","from pathlib import Path; from experiments.flexible_execution.continuing_jobs import runtime_fingerprint; from experiments.flexible_execution.revision_audit import read; import sys; print(runtime_fingerprint(read(Path(sys.argv[1]))))",str(config_path.resolve())],cwd=cwd,env=env,capture_output=True,text=True)
        if source.returncode==0 and source.stdout.strip()==config["runtime_sha256"]:
            return cwd,env
    raise ValueError("The actual registered worker source is unavailable in the worktrees sharing these receipts")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=Path("config/native_evidence_dev02.json"))
    parser.add_argument("--block",choices=("primary","comparators"),default="primary")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--restoration-evidence",help="A verified reset or the user's confirmation that this same authorized plan is available again")
    args=parser.parse_args()
    config=read(args.config)
    root=Path("outputs/continuing")/config["id"]
    failures=interruptions(root,config)
    pending=remaining(config,root,args.block)
    worker_dir,worker_env=frozen_runtime(config,args.config,root)
    info=dict(cohort=config["id"],block=args.block,interruptions=failures,pending=pending,
              availability_confirmed=acknowledged(root,failures),worker_tree_commit=subprocess.check_output(
                  ["git","rev-parse","HEAD"],cwd=worker_dir,text=True).strip(),requests_made_by_inspection=0)
    print(json.dumps(info,ensure_ascii=False))
    target=root/"recovery"
    target.mkdir(parents=True,exist_ok=True)
    save(target/("inspection-"+args.block+".json"),info)
    if not args.run or not pending:
        return
    if failures and not info["availability_confirmed"] and not (args.restoration_evidence or "").strip():
        raise SystemExit("The authorized endpoint still has a recorded limit. Wait for verified service restoration.")
    if failures and not info["availability_confirmed"]:
        refs=sorted(f["receipt_sha256"] for f in failures)
        name=sha256(json.dumps(refs).encode()).hexdigest()[:16]
        save(target/("restoration-"+name+".json"),dict(evidence=args.restoration_evidence.strip(),interrupted_receipt_sha256=refs))
    limits=unresolved_service_limits(root.parent)
    if limits:
        raise SystemExit("A recorded limit remains on this plan: "+str(limits))
    suffix=continuation(config,root,args.block,pending,digest(args.config))
    token=sha256(json.dumps(suffix,sort_keys=True).encode()).hexdigest()[:16]
    path=target/("registered-"+args.block+"-"+token+".json")
    if path.exists() and read(path)!=suffix:
        raise ValueError("Suffix manifest conflict")
    save(path,suffix)
    subprocess.run([sys.executable,"-m","experiments.flexible_execution.continuing_jobs","--config",
                   str(path.resolve()),"--run","--block",args.block],cwd=worker_dir,env=worker_env,check=True)


if __name__=="__main__":
    main()
