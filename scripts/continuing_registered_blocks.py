"""Run or inspect the remaining registered development work, reconciling actual receipts between blocks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep

from experiments.flexible_execution.continuing_jobs import now,snapshot,unresolved_service_limits
from experiments.flexible_execution.revision_audit import read,save,digest
from scripts.continuing_native_report import comparator_witnesses
from scripts.continuing_recovery import remaining,interruptions


OUT=Path("outputs/continuing")
COHORTS=[Path("config/native_evidence_dev02.json"),Path("config/coordinated_handoff_dev03.json")]


def cli(*words):
    return [sys.executable,*map(str,words)]


def controller_alive(pid):
    if not isinstance(pid,int):
        return False
    try:
        cmd=Path("/proc",str(pid),"cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError,ProcessLookupError):
        return False
    return b"experiments.flexible_execution.continuing_jobs" in cmd


def active_controller(root):
    beat=root/"heartbeat.json"
    if not beat.exists():
        return None
    data=read(beat)
    pid=data.get("controller_pid")
    return pid if controller_alive(pid) else None


def verify_source(config,root):
    for c in config["cases"]:
        for m in c["primary"]+c["comparators"]:
            p=root/c["id"]/(m+".receipt.json")
            if p.exists() and read(p).get("source_sha256")!=config["runtime_sha256"]:
                raise ValueError("Mixed worker versions in cohort: "+str(p))


def mark(stage,**extra):
    save(OUT/"registered-blocks.json",dict(utc=now(),coordinator_pid=os.getpid(),stage=stage,**extra))


def report(paths):
    for path in paths:
        subprocess.run(cli("scripts/continuing_native_report.py","--config",path),check=True,stdout=subprocess.DEVNULL)
    subprocess.run(cli("scripts/continuing_checkpoint.py"),check=True,stdout=subprocess.DEVNULL)
    subprocess.run(cli("scripts/continuing_interface_audit.py"),check=True,stdout=subprocess.DEVNULL)
    if (OUT/"view-demand-dev01/freeze_receipt.json").is_file():
        subprocess.run(cli("scripts/continuing_view_demand.py","campus"),check=True,stdout=subprocess.DEVNULL)
    subprocess.run(cli("scripts/continuing_results.py"),check=True,stdout=subprocess.DEVNULL)


def monitor(root,proc=None):
    samplers={}
    adopted=active_controller(root)
    def still_running():
        return proc.poll() is None if proc is not None else controller_alive(adopted)
    while still_running():
        pid=active_controller(root)
        if pid and pid not in samplers:
            # An already-running block may already have an external resource sampler.
            matches=False
            for f in Path("/proc").glob("[0-9]*/cmdline"):
                try:
                    args=f.read_bytes().split(b"\0")
                    if b"scripts/revision_resources.py" in args and str(pid).encode() in args:
                        matches=True;break
                except (FileNotFoundError,PermissionError,ProcessLookupError):
                    pass
            samplers[pid]=None if matches else subprocess.Popen(cli("scripts/revision_resources.py","--pid",pid,"--root",root))
        sleep(2)
    for child in samplers.values():
        if child is not None:
            child.wait(timeout=10)
    if proc is not None and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode,proc.args)


def gate(path,config,root):
    proof=read(root/"audit/photo_chains.json")
    witness=comparator_witnesses(config,proof)
    if config.get("admission")=="coordinated-motion-material":
        from scripts.continuing_sensor_visibility import nominated_positive_visible
        witness=[p for p in witness if nominated_positive_visible(root/p["case"]/p["method"],p)]
    decision=root/"comparator_gate.json"
    primaries=[root/c["id"]/(arm+".receipt.json") for c in config["cases"] for arm in c["primary"]]
    if not all(p.exists() for p in primaries):
        raise ValueError("A primary registered slot remains unaccounted")
    immutable=[digest(p) for p in primaries]
    violations=False
    table=root/"audit/episodes.csv"
    if table.exists():
        import csv
        with table.open() as fp:
            violations=any(int(r["field_grant_violation_count"]) for r in csv.DictReader(fp))
    if not witness or violations:
        if decision.exists():
            raise ValueError("A prior positive gate no longer has its evidence")
        skip=root/"comparator_gate_closed.json"
        record=dict(admit=False,config_sha256=digest(path),primary_receipts=immutable,
                    reason="Documented permission violation" if violations else "No photo met the registered mechanism witness")
        if skip.exists() and read(skip)!=record:
            raise ValueError("An unchanged registration cannot reverse its gate")
        save(skip,record)
        return False
    if (root/"comparator_gate_closed.json").exists():
        raise ValueError("A previously closed gate cannot be reopened on the same exposures")
    subprocess.run(cli("scripts/continuing_native_report.py","--config",path,"--admit-comparators"),check=True,stdout=subprocess.DEVNULL)
    return read(decision)["admit"] is True


def restriction():
    limits=unresolved_service_limits(OUT)
    if limits:
        mark("service-limit",receipts=[p for p,_ in limits])
        raise SystemExit("Preserved external service restriction: "+str(limits))


def drain():
    with (OUT/"registered-blocks.lock").open("a+") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path in COHORTS:
            config=read(path);root=OUT/config["id"]
            if active_controller(root):
                mark("monitor-existing",cohort=config["id"])
                monitor(root)
            verify_source(config,root)
            restriction()
            for block in ("primary","comparators"):
                report(COHORTS)
                if block=="comparators" and not gate(path,config,root):
                    break
                pending=remaining(config,root,block)
                while pending:
                    restriction()
                    mark("native",cohort=config["id"],block=block,unsent=pending)
                    proc=subprocess.Popen(cli("scripts/continuing_recovery.py","--config",path,"--block",block,"--run"))
                    try:
                        monitor(root,proc)
                    except subprocess.CalledProcessError:
                        restriction()
                        ended=interruptions(root,config,include_other=True)
                        if not any((f["case"],f["method"]) in pending for f in ended):
                            raise
                    verify_source(config,root)
                    rest=remaining(config,root,block)
                    if not set(rest)<set(pending):
                        raise ValueError("No registered slot settled; preserve the worker before another dispatch")
                    pending=rest
            report(COHORTS)
        restriction()
        mark("one-photo-regions")
        subprocess.run(cli("scripts/continuing_region_assay.py","predict"),check=True)
        subprocess.run(cli("scripts/continuing_region_assay.py","grade"),check=True)
        region=read(OUT/"same-frame-region-assay01/summary.json")
        passing=(len(region["cases"])==2 and all(c["group_all_correct"] is True and c["single_all_correct"] is True for c in region["cases"]))
        if passing:
            mark("scoped-physical-mechanism")
            for _ in read(Path("config/scoped_mission_dev01.json"))["online_order"]:
                restriction()
                subprocess.run(cli("scripts/continuing_scoped_mission.py","run-next"),check=True)
            subprocess.run(cli("scripts/continuing_scoped_mission.py","grade"),check=True)
        report(COHORTS)
        mark("registered-decision-point",scoped_mechanism_admitted=passing,region_summary_sha256=digest(OUT/"same-frame-region-assay01/summary.json"))
        print("Bounded registered blocks settled; inspect outcomes for the next scientific decision.")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",action="store_true")
    if parser.parse_args().run:
        drain()
        return
    for path in COHORTS:
        config=read(path);root=OUT/config["id"]
        verify_source(config,root)
        rows=snapshot(root,config)
        counts={s:sum(r["state"]==s for r in rows) for s in sorted({r["state"] for r in rows})}
        print(json.dumps(dict(cohort=config["id"],actual_controller=active_controller(root),states=counts)))
    print("unresolved service limits",unresolved_service_limits(OUT),"inspection sends",0)


if __name__=="__main__":
    main()
