"""Local learned task-relevant evidence dependency; no live model latency."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

from experiments.task_attack.observer import _get
from experiments.task_attack.runtime import append_json,write_json
from .learned import Policy
from .programs import known_program

POLICIES=('belief','belief_blind','belief_known_task')


def run(url,policy,output,model,perception,task=None):
    if (policy=='belief_known_task')!=(task is not None):raise ValueError('Information condition mismatch')
    if perception.get('current_task_used') or not perception.get('pre_task_static'):raise ValueError('Not a shared pre-task scene interpretation')
    root=Path(output);root.mkdir(parents=True,exist_ok=True);catalog=_get(url,'catalog');initial=_get(url,'initial')
    known=known_program(task,model['programs'],perception['stations']) if task else None
    predictor=Policy(model,perception,catalog['editable_actions'],policy!='belief_blind',known)
    summary=dict(policy=policy,decision='collecting',implementation='learned_local_dependency_v'+str(model['version']),calls=0,
        trained_operation_model=True,whole_task_classification_required=False,full_paper_KDE_reproduction=False,
        shared_static_preparation_requests=1,online_inference_requests=0,
        task_dependent_evidence_calibration=model.get('informative_prefix',{}).get('calibration'),
        model_calibration=model['dependency']['groups'],static_rgb_sha256=perception['actual_static_rgb_sha256'])
    # Keep compact calibration provenance in the summary; weights are frozen
    # in the separately hashed model bundle, not repeated per decision.
    summary['model_calibration']={k:dict(use_dynamic=v['use_dynamic'],dynamic=v['dynamic']['calibration'],clock=v['clock_only']['calibration']) for k,v in model['dependency']['groups'].items()}
    initial_track=initial.get('external_pixel_history') or initial.get('frames',[])
    history={f['id']:{k:f[k] for k in ('id','capture_ros','arrival_wall','robot_pixels')} for f in initial_track};last_frame=None;last_update=-1.
    write_json(root/'result.json',summary)
    while True:
        state=_get(url,'state')
        if state['finished'] or state['edit_used']:summary['decision']='episode_ended_before_dispatch';break
        if not state['started']:time.sleep(.12);continue
        window=initial if policy=='belief_blind' else _get(url,'frames')
        elapsed=state['elapsed_seconds'];frame_id=window['frames'][-1]['id'] if window.get('frames') else None
        if (policy=='belief_blind' and elapsed-last_update<1.) or (policy!='belief_blind' and frame_id==last_frame):time.sleep(.12);continue
        last_frame=frame_id;last_update=elapsed
        for frame in window.get('external_pixel_history',[]):history[frame['id']]=frame
        track=sorted(history.values(),key=lambda f:(f['capture_ros'],f['id']))
        selected=predictor.decide(track,elapsed,catalog['timing']['robot_after_takeoff_budget_seconds'])
        append_json(root/'decisions.jsonl',dict(wall=time.time(),elapsed=elapsed,dispatch_frame_ids=[f['id'] for f in window.get('frames',[])],
            observed_prefix_frames=len(track),computed_policy=selected))
        summary.update(decision=selected['kind'],computed_policy=selected);write_json(root/'result.json',summary)
        if selected['kind']=='abstain':break
        if selected['kind']=='wait':continue
        action=selected['ranked_candidates'][0]['action_id'];summary.update(decision='dispatching',action_id=action);write_json(root/'result.json',summary)
        request=urllib.request.Request(url+'/edit',data=json.dumps(dict(action_id=action)).encode(),headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(request,timeout=8) as response:ack=json.load(response)
            summary.update(decision='edit',acknowledgment=ack)
        except Exception as error:summary.update(decision='edit_failed_or_late',error_type=type(error).__name__)
        break
    write_json(root/'result.json',summary);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--gateway',required=True);p.add_argument('--policy',choices=POLICIES,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--call-budget',type=int,default=1);p.add_argument('--effort',default='high')
    p.add_argument('--timing-model',type=Path,required=True);p.add_argument('--public-perception',type=Path,required=True);p.add_argument('--known-task-file',type=Path);a=p.parse_args()
    try:run(a.gateway,a.policy,a.output,json.loads(a.timing_model.read_text()),json.loads(a.public_perception.read_text()),a.known_task_file.read_text() if a.known_task_file else None)
    except (Exception,KeyboardInterrupt) as error:
        path=a.output/'result.json';summary=json.loads(path.read_text()) if path.exists() else {}
        summary.update(decision='observer_error' if not isinstance(error,KeyboardInterrupt) else 'interrupted_at_episode_end',error_type=type(error).__name__,error=str(error)[:200]);write_json(path,summary)
        if not isinstance(error,KeyboardInterrupt):raise


if __name__=='__main__':main()
