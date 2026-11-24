"""Precompute once; update action and timing from actual external observations."""
import argparse
import base64
import json
from pathlib import Path
import time
import urllib.request

from experiments.task_attack.model import ModelClient
from experiments.task_attack.observer import _get
from experiments.task_attack.runtime import append_json,write_json
from experiments.task_attack_realtime.belief import SYSTEM,validate,DEPENDENCY_POLICIES
from .lookahead import select


def run(url,policy,output,call_budget=1,effort='high',task=None,timing_model=None):
    if policy not in DEPENDENCY_POLICIES or (policy=='belief_known_task')!=(task is not None):raise ValueError('Invalid information condition')
    if call_budget!=1 or timing_model is None:raise ValueError('One shared request and a frozen continuation model required')
    root=Path(output);root.mkdir(parents=True,exist_ok=True);client=ModelClient(root/'model_calls.jsonl',role='observer',effort=effort)
    summary=dict(policy=policy,implementation='cached_policy_lookahead_v9',decision='collecting',calls=0,
        whole_task_classification_required=False,trained_operation_model=False,full_paper_KDE_reproduction=False)
    def checkpoint():
        summary.update(calls=client.calls,model_wall_seconds=client.wall_seconds);write_json(root/'result.json',summary)
    checkpoint();catalog=_get(url,'catalog');initial=_get(url,'initial');state=_get(url,'state');menu=catalog['editable_actions']
    if state['finished'] or state['edit_used']:summary['decision']='late_or_budget_consumed';checkpoint();return summary
    data=dict(public_scene_conditions=catalog['scene_conditions'],public_task_prior=catalog['task_prior'],
        public_menu=[dict(index=i,**a) for i,a in enumerate(menu)],public_clock=state,
        observation_access='INITIAL static EXTERNAL RGB only; dynamic information becomes available locally at dispatch',max_observer_requests=1)
    if task is not None:data['authorized_task_reference']=task
    images=[dict(label='Actual INITIAL external RGB id='+str(f['id']),data=base64.b64decode(f['jpeg_base64'])) for f in initial['frames']]
    summary.update(decision='inference_pending',request_started_wall=time.time());checkpoint();start=time.monotonic()
    response=client.request(SYSTEM,data,images,attempt_limit=1);validate(response,menu)
    summary.update(last_request_seconds=time.monotonic()-start,last_response=response);checkpoint()
    last_frame=None;last_update=-1.;scene=catalog['scene_conditions']['scene']
    while True:
        state=_get(url,'state')
        if state['finished'] or state['edit_used']:summary['decision']='episode_ended_before_dispatch';break
        window=_get(url,'initial' if policy=='belief_blind' else 'frames')
        frame_id=window['frames'][-1]['id'] if window['frames'] else None;elapsed=state['elapsed_seconds']
        if (policy=='belief_blind' and elapsed-last_update<1.) or (policy!='belief_blind' and frame_id==last_frame):
            time.sleep(.15);continue
        last_frame=frame_id;last_update=elapsed
        selected=select(timing_model,response,menu,window.get('external_pixel_history',[]),elapsed,scene,policy!='belief_blind')
        append_json(root/'decisions.jsonl',dict(wall=time.time(),elapsed=elapsed,dispatch_frame_ids=[f['id'] for f in window['frames']],
            computed_policy=selected,input_was_cached=True))
        summary.update(decision=selected['kind'],computed_policy=selected);checkpoint()
        if selected['kind']=='abstain':break
        if selected['kind']=='wait':continue
        action=selected['ranked_candidates'][0]['action_id'];summary.update(decision='dispatching',action_id=action);checkpoint()
        request=urllib.request.Request(url+'/edit',data=json.dumps(dict(action_id=action)).encode(),headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(request,timeout=8) as result:ack=json.load(result)
            summary.update(decision='edit',acknowledgment=ack)
        except Exception as error:summary.update(decision='edit_failed_or_late',error_type=type(error).__name__)
        break
    checkpoint();return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--gateway',required=True);p.add_argument('--policy',choices=DEPENDENCY_POLICIES,required=True)
    p.add_argument('--output',required=True);p.add_argument('--call-budget',type=int,default=1);p.add_argument('--effort',choices=('high','xhigh','max'),default='high')
    p.add_argument('--known-task-file',type=Path);p.add_argument('--timing-model',type=Path,required=True);a=p.parse_args();root=Path(a.output)
    try:run(a.gateway,a.policy,root,a.call_budget,a.effort,a.known_task_file.read_text() if a.known_task_file else None,json.loads(a.timing_model.read_text()))
    except (Exception,KeyboardInterrupt) as error:
        summary=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {}
        summary.update(decision='observer_error' if not isinstance(error,KeyboardInterrupt) else 'inference_interrupted_at_episode_end',
            error_type=type(error).__name__,error_code=getattr(error,'code',None),status=getattr(error,'status',None));write_json(root/'result.json',summary)
        if not isinstance(error,KeyboardInterrupt):raise


if __name__=='__main__':main()
