"""External-only operation belief with a frozen compute/dispatch time slot."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

from experiments.task_attack.observer import _get
from experiments.task_attack.runtime import append_json,write_json
from experiments.task_attack_value.programs import known_program
from experiments.task_attack_value.vision import enrich_window
from .learned import Policy


def run(url,policy,output,model,perception,task=None):
    if (policy=='belief_known_task')!=(task is not None):raise ValueError('Information condition mismatch')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    catalog=_get(url,'catalog');initial=_get(url,'initial')
    known=known_program(task,model['programs'],perception['stations']) if task else None
    predictor=Policy(model,perception,catalog['editable_actions'],policy!='belief_blind',known)
    history=enrich_window(initial,{});next_lock=0.;last_id=None
    summary=dict(policy=policy,implementation='finite_shared_operation_belief_v17',trained_operation_model=True,
        full_paper_KDE_reproduction=False,shared_static_preparation_requests=1,online_inference_requests=0,
        calls=0,decision='collecting',static_rgb_sha256=perception['actual_static_rgb_sha256'],
        model_limitations=model['operation_model']['limitations'])
    write_json(root/'result.json',summary)
    while True:
        state=_get(url,'state')
        if state['finished'] or state['edit_used']:summary['decision']='episode_ended_before_dispatch';break
        if not state['started'] or time.monotonic()<next_lock:time.sleep(.12);continue
        window=initial if policy=='belief_blind' else _get(url,'frames')
        frame_id=window['frames'][-1]['id'] if window.get('frames') else None
        if policy!='belief_blind' and frame_id==last_id:time.sleep(.12);continue
        enrich_window(window,history);last_id=frame_id;track=sorted(history.values(),key=lambda f:(f['capture_ros'],f['id']))
        locked=time.monotonic();selected=predictor.decide(track,state['elapsed_seconds'],catalog['timing']['robot_after_takeoff_budget_seconds'])
        slot=model['operation_model']['fixed_compute_dispatch_seconds'];compute=time.monotonic()-locked
        append_json(root/'decisions.jsonl',dict(wall=time.time(),elapsed=state['elapsed_seconds'],observed_prefix_frames=len(track),
            dispatch_frame_ids=[f['id'] for f in window.get('frames',[])],compute_seconds=compute,computed_policy=selected))
        summary.update(decision=selected['kind'],computed_policy=selected);write_json(root/'result.json',summary)
        if compute>slot:
            summary.update(decision='fixed_compute_slot_missed',actual_compute_seconds=compute);break
        if selected['kind']=='abstain':break
        while time.monotonic()<locked+slot:time.sleep(max(0.,min(.05,locked+slot-time.monotonic())))
        if selected['kind']=='wait':
            next_lock=locked+slot+float(selected.get('selected_delay_seconds') or 3.);continue
        action=selected['ranked_candidates'][0]['action_id'];summary.update(decision='dispatching',action_id=action);write_json(root/'result.json',summary)
        request=urllib.request.Request(url+'/edit',data=json.dumps(dict(action_id=action)).encode(),headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(request,timeout=8) as response:ack=json.load(response)
            summary.update(decision='edit',acknowledgment=ack)
        except Exception as error:summary.update(decision='edit_failed_or_late',error_type=type(error).__name__)
        break
    write_json(root/'result.json',summary);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--gateway',required=True);p.add_argument('--policy',choices=('belief','belief_blind','belief_known_task'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timing-model',type=Path,required=True);p.add_argument('--public-perception',type=Path,required=True)
    p.add_argument('--known-task-file',type=Path);p.add_argument('--effort',default='high');p.add_argument('--call-budget',type=int,default=1);a=p.parse_args()
    try:run(a.gateway,a.policy,a.output,json.loads(a.timing_model.read_text()),json.loads(a.public_perception.read_text()),a.known_task_file.read_text() if a.known_task_file else None)
    except (Exception,KeyboardInterrupt) as error:
        path=a.output/'result.json';summary=json.loads(path.read_text()) if path.exists() else {}
        summary.update(decision='observer_error' if not isinstance(error,KeyboardInterrupt) else 'interrupted_at_episode_end',error_type=type(error).__name__,error=str(error)[:200]);write_json(path,summary)
        if not isinstance(error,KeyboardInterrupt):raise


if __name__=='__main__':main()
