"""Source manifest used when designing the post-quota development protocols."""
from pathlib import Path

from experiments.flexible_execution.revision_audit import read,digest


def before_restoration(include_interrupted=False):
    result=[]
    config=read(Path("config/pre_recovery_native_exposures.json"))
    for row in config["records"]:
        if not include_interrupted and row["state"]!="scored":
            continue
        f=Path("outputs/continuing")/row["receipt"]
        receipt=read(f)
        robot=f.with_name(receipt["method"])/"private/robot"
        if digest(f)!=row["receipt_sha256"] or digest(robot/"model_calls.jsonl")!=row["model_log_sha256"]:
            raise ValueError("Frozen development source has changed: "+str(f))
        result.append(f)
    return result
