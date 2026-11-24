"""OFFLINE proxy-only controlled interventions with explicit teacher provenance.

The teacher can instrument its OWN training robot. No teacher action, true
task or evaluator answer is provided to any held-out attack policy.
"""
import argparse
import json
import random
import time
import urllib.request
from pathlib import Path
from experiments.task_attack.observer import _get
from experiments.task_attack.runtime import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--gateway',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',choices=('randomized',),required=True);p.add_argument('--calibration-seed',type=int,required=True)
    p.add_argument('--teacher-action');p.add_argument('--teacher-propensity',type=float)
    p.add_argument('--call-budget',type=int,default=1);p.add_argument('--effort',default='high');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);catalog=_get(a.gateway,'catalog')
    menu=[item for item in catalog['editable_actions'] if item['kind'] in ('cosmetic_shell_decal','printed_display_overlay')]
    rng=random.Random(a.calibration_seed);slots=[15.,40.,65.]
    if a.teacher_action:
        action=next((item for item in menu if item['action_id']==a.teacher_action),None)
        if action is None or not isinstance(a.teacher_propensity,float) or not 0<a.teacher_propensity<=1:raise ValueError('Invalid training-only teacher action/propensity')
        propensity=a.teacher_propensity
    else:action=rng.choice(menu);propensity=1./(len(menu)*len(slots))
    slot=rng.choice(slots)
    result=dict(policy='randomized',training_only=True,teacher_selected=bool(a.teacher_action),
        randomization_seed=a.calibration_seed,action_id=action['action_id'],sampled_dispatch_seconds=slot,
        propensity=propensity,clock_slots=slots,calls=0,decision='waiting_public_clock',
        meaning='Controlled proxy training intervention. This is NOT a method evaluated under hidden-task conditions.')
    file=a.output/'result.json';write_json(file,result)
    while True:
        state=_get(a.gateway,'state')
        if state['finished'] or state['edit_used']:
            result['decision']='episode_ended_before_training_slot';write_json(file,result);return
        if state['elapsed_seconds']>=slot:break
        time.sleep(.2)
    request=urllib.request.Request(a.gateway+'/edit',data=json.dumps(dict(action_id=action['action_id'])).encode(),headers={'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(request,timeout=8) as response:result['acknowledgment']=json.load(response)
        result['decision']='edit'
    except Exception as error:result.update(decision='edit_failed_or_late',error_type=type(error).__name__)
    write_json(file,result)


if __name__=='__main__':main()
