"""Verify original prompts, paired simulator inputs and local resource sample coverage."""
import ast
import csv
import json
import subprocess
import os
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from experiments.flexible_execution.adapters.original_controller import AGENT_BLOB, AGENT_SHA256
from experiments.flexible_execution.revision_audit import OUT, digest, read, save, write_table
from experiments.flexible_execution.timeline import read_trace_rows


def main():
    upstream = Path(os.environ.get("VLA_LEGACY_ROOT", Path.home()/"uav/Embodied-intelligent-behavior"))
    found=subprocess.run(["git","cat-file","blob",AGENT_BLOB],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    code=found.stdout if found.returncode==0 else (upstream/"experiments/task_attack/agent.py").read_bytes()
    assert sha256(code).hexdigest()==AGENT_SHA256
    syntax=ast.parse(code)
    system=next(ast.literal_eval(s.value) for s in syntax.body if isinstance(s,ast.Assign)
                and any(isinstance(name,ast.Name) and name.id=="SYSTEM" for name in s.targets))
    prompt=sha256(system.encode()).hexdigest()
    helpers={}
    base=Path("experiments/task_attack")
    for filename,names in (("records.py",("object_id","validate_record","facts")),("runtime.py",("append_json","write_json"))):
        versions=[]
        for origin in (upstream, Path.cwd()):
            syntax=ast.parse((origin/base/filename).read_text())
            versions.append({f.name:sha256(ast.dump(f,include_attributes=False).encode()).hexdigest()
                             for f in syntax.body if isinstance(f,ast.FunctionDef) and f.name in names})
        assert versions[0]==versions[1] and set(versions[0])==set(names)
        helpers[filename]=versions[0]
    originals=[]
    for log in sorted(OUT.glob("dev-*/original-gpt/private/robot/model_calls.jsonl")):
        calls,_=read_trace_rows(log,allow_truncated=True)
        assert all(c["instructions_sha256"]==prompt and c["model"]=="gpt-6.1-terra"
                   and c["reasoning_effort"]=="max" for c in calls)
        originals.append(dict(case=log.parents[3].name,calls=len(calls),raw_sha256=digest(log)))
    rows=[]
    for p in [OUT/"development_results.csv", OUT/"round3/development_results.csv"]:
        if p.exists():
            with p.open() as stream:
                rows.extend(csv.DictReader(stream))
    groups=defaultdict(list)
    for r in rows:
        groups[r["case"]].append(r)
    equality={}
    for case, batch in groups.items():
        equality[case]={k:sorted({r[k] for r in batch if r[k]}) for k in
                        ("canonical_world_sha256","texture_set_sha256","task_sha256")}
        if any(len(s)!=1 for s in equality[case].values()):
            raise ValueError("Scene sources differ between compared arms")
    save(OUT/"run_provenance.json",dict(original_source_sha256=AGENT_SHA256,upstream_system_sha256=prompt,
         original_calls=originals,original_record_and_output_helpers=helpers,paired_input_digests=equality))
    samples=defaultdict(list)
    for p in OUT.rglob("resources-*.jsonl"):
        for row in read_trace_rows(p,allow_truncated=True)[0]:
            samples[(str(p.parent.relative_to(OUT)),row["case"],row["method"])].append(row)
    resources=[]
    for (block,case,arm),rs in sorted(samples.items()):
        times=[r["monotonic"] for r in rs]
        resources.append(dict(block=block,case=case,method=arm,samples=len(rs),window_s=max(times)-min(times),
             peak_sampled_rss_mib=max(r["rss_sum_bytes"] for r in rs)/2**20,
             peak_alive_cpu_s=max(r["alive_cpu_seconds_sum"] for r in rs),
             note="2 s local process-tree samples; summed RSS may count shared pages more than once; GPU and energy unmeasured"))
    if resources:
        write_table(OUT/"sampled_resources.csv",resources)
    print("Original controller prompt verified:",originals,"matched scene groups:",len(equality),"resource windows:",len(resources))


if __name__=="__main__":
    main()
