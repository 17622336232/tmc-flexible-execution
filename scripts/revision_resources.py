"""Passive local process-tree samples for an already-authorized simulator block."""
import argparse
import json
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    parent = psutil.Process(args.pid)
    if "experiments.flexible_execution.revision_jobs" not in parent.cmdline():
        raise SystemExit("PID does not identify a revision study controller")
    birth = parent.create_time()
    path = args.root / ("resources-%d-%d.jsonl" % (args.pid, time.time()))
    with path.open("x",encoding="utf8",buffering=1) as stream:
        while parent.is_running() and parent.status()!=psutil.STATUS_ZOMBIE and parent.create_time()==birth:
            active=[]
            for p in args.root.glob("*/*.receipt.json"):
                try:
                    d=json.loads(p.read_text())
                except (FileNotFoundError,json.JSONDecodeError):
                    continue
                if d["state"]=="starting" and d["controller_pid"]==args.pid:
                    active.append((p.parent.name,d["method"]))
            if len(active)==1:
                rss=0
                cpu=0.
                processes=0
                for child in parent.children(recursive=True):
                    try:
                        rss+=child.memory_info().rss
                        t=child.cpu_times()
                        cpu+=t.user+t.system
                        processes+=1
                    except (psutil.NoSuchProcess,psutil.AccessDenied):
                        continue
                row=dict(case=active[0][0],method=active[0][1],monotonic=time.monotonic(),
                         rss_sum_bytes=rss,alive_cpu_seconds_sum=cpu,processes=processes)
                stream.write(json.dumps(row)+"\n")
            time.sleep(2)


if __name__=="__main__":
    main()
