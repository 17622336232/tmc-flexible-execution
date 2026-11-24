"""Strengthened external trajectory-anchor adaptation, not original RedVLA.

Use the same trained STATIC payload values as blind; motion selects the closest
trajectory source. It never predicts the current task's local READ dependency.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request

import numpy as np

from experiments.task_attack.baselines import SOURCES,closest_anchor,directional_update
from experiments.task_attack.event_timing import features
from experiments.task_attack_cached.lookahead import onset_probability
from experiments.task_attack.observer import _get
from experiments.task_attack.runtime import append_json,write_json
from .learned import Policy
from .programs import payload
from .vision import candidate_features,enrich_window


def run(url,output,model,perception):
    root=Path(output);root.mkdir(parents=True,exist_ok=True);catalog=_get(url,'catalog');initial=_get(url,'initial')
    predictor=Policy(model,perception,catalog['editable_actions'],False)
    summary=dict(policy='redvla_external',paper_adaptation=True,source=SOURCES['redvla_external'],
        implementation='trajectory_anchor_with_shared_static_payload_v13',calls=0,
        shared_static_preparation_requests=1,not_full_paper_reproduction=True,
        stronger_than_old='Shared optimized static payload objective; takeoff separated from task motion')
    history=enrich_window(initial,{});first=dict(history[initial['frames'][0]['id']]);last_id=None;write_json(root/'result.json',summary)
    while True:
        state=_get(url,'state')
        if state['finished'] or state['edit_used']:summary['decision']='episode_ended_before_behavior';break
        window=_get(url,'frames')
        enrich_window(window,history)
        track=sorted(history.values(),key=lambda item:item['id'])
        x=features(track,state['elapsed_seconds'])
        if x is None or onset_probability(model['legacy_onset'],x)<.5:time.sleep(.15);continue
        current=candidate_features([first],state['elapsed_seconds'],perception['anchors'])
        ranks=predictor.score(current);by_source={}
        for rank in ranks:
            source=payload(predictor.action_by_id[rank['action_id']])['source']
            by_source.setdefault(source,rank)
        points=[]
        for i,frame in enumerate(track):
            if not frame.get('robot_pixels'):continue
            value=features(track[max(0,i-35):i+1],max(0.,frame['capture_ros']-track[0]['capture_ros']))
            if value is not None and onset_probability(model['legacy_onset'],value)>=.5:points.append(frame['robot_pixels']['centroid'])
        if not points:time.sleep(.15);continue
        scored=[]
        for source,rank in by_source.items():
            center=np.array([perception['anchors'][source]['u'],perception['anchors'][source]['v']]);anchor=closest_anchor(points,center)
            adjusted=directional_update(center,anchor,.05,valid=lambda p:bool(np.all(p>=0)&np.all(p<=1)))
            scored.append(dict(**rank,source=source,anchor=anchor.tolist(),proposed_anchor_position=adjusted.tolist(),distance=float(np.linalg.norm(anchor-center))))
        scored.sort(key=lambda item:(item['distance'],-item['expected_wrong_accepted'],item['action_id']))
        action=scored[0]['action_id'];summary.update(decision='dispatching',action_id=action,anchor_candidates=scored,actual_external_onset_probability=onset_probability(model['legacy_onset'],x))
        write_json(root/'result.json',summary)
        request=urllib.request.Request(url+'/edit',data=json.dumps(dict(action_id=action)).encode(),headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(request,timeout=8) as result:summary['acknowledgment']=json.load(result)
            summary['decision']='edit'
        except Exception as error:summary.update(decision='edit_failed_or_late',error_type=type(error).__name__)
        append_json(root/'decisions.jsonl',dict(wall=time.time(),elapsed=state['elapsed_seconds'],observed_prefix_frames=len(track),response=summary));break
    write_json(root/'result.json',summary);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--gateway',required=True);p.add_argument('--policy',choices=('redvla_external',),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--effort',default='high');p.add_argument('--call-budget',type=int,default=1)
    p.add_argument('--timing-model',type=Path,required=True);p.add_argument('--public-perception',type=Path,required=True);a=p.parse_args()
    try:run(a.gateway,a.output,json.loads(a.timing_model.read_text()),json.loads(a.public_perception.read_text()))
    except (Exception,KeyboardInterrupt) as error:
        path=a.output/'result.json';summary=json.loads(path.read_text()) if path.exists() else {}
        summary.update(decision='observer_error' if not isinstance(error,KeyboardInterrupt) else 'interrupted_at_episode_end',error_type=type(error).__name__);write_json(path,summary)
        if not isinstance(error,KeyboardInterrupt):raise


if __name__=='__main__':main()
