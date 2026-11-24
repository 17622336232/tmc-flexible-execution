"""Extend only the observer subprocess of the unchanged native experiment.

Never monkeypatch stdlib subprocess, native ROS, sensors, actor tools or score.
Use a separate extension audit because the old native hash excludes artifacts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.task_attack_integrity import run as native
from experiments.task_attack.runtime import write_json
from experiments.task_attack_sensor_proxy.audit import implementation_hash

ROOT=Path(__file__).resolve().parents[2]
H=Path(__file__).resolve().parent


class ObserverOnlyProxy:
    def __init__(self,policies):self.policies=policies;self.changed=[]
    def __getattr__(self,name):return getattr(subprocess,name)
    def Popen(self,command,*args,**kwargs):
        cmd=list(command)
        if '-m' in cmd and '--policy' in cmd:
            module=cmd[cmd.index('-m')+1];policy=cmd[cmd.index('--policy')+1]
            if module in ('experiments.task_attack_integrity.belief','experiments.task_attack_integrity.direct','experiments.task_attack_integrity.redvla') and policy in self.policies:
                cmd[cmd.index('-m')+1]='conditional_critic_observer'
                env=dict(kwargs.get('env',os.environ));env['PYTHONPATH']=os.pathsep.join(str(q) for q in (ROOT/'.runtime/task_attack_learning_deps',ROOT/'.runtime/task_attack_cpu_deps',H,ROOT/'bridge',ROOT))
                kwargs['env']=env;self.changed.append(dict(original_module=module,policy=policy))
        return subprocess.Popen(cmd,*args,**kwargs)


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--candidate-policies',default='belief');known,argv=p.parse_known_args()
    policies=set(known.candidate_policies.split(','))
    if not policies<={'belief','belief_blind','belief_known_task','direct_effect','redvla_external'}:raise ValueError('Only explicit observer policies can be extended')
    output=Path(argv[argv.index('--output')+1]);source=implementation_hash();proxy=ObserverOnlyProxy(policies)
    extension={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),H/'conditional_critic_observer.py',H/'conditional_critic_policy.py',H/'common_resource_red_anchor.py',H/'crossfit_dependency_stopping_data.py',H/'learn_read_dependency.py']}
    native.subprocess=proxy
    try:native.main(argv)
    finally:
        extension_after={name:hashlib.sha256((H/name).read_bytes()).hexdigest() for name in extension}
        audit=dict(status='Observer-only DEVELOPMENT extension; not an unmodified-controller claim or formal superiority evidence',native_sha256_before=source,native_sha256_after=implementation_hash(),observer_overrides=proxy.changed,extension_source_sha256=extension,extension_source_sha256_after=extension_after,extension_unchanged=extension_after==extension,
            original_actor_safety_sensors_scoring_preserved=source==implementation_hash(),actual_observer_module='conditional_critic_observer' if proxy.changed else None)
        if output.exists():
            write_json(output/'candidate_execution_audit.json',audit)
            path=output/'manifest.json'
            if path.exists():
                manifest=json.loads(path.read_text());manifest['observer_extension']=audit;manifest['full_paper_KDE_reproduction']=False;write_json(path,manifest)


if __name__=='__main__':main()
