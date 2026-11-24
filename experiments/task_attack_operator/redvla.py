"""External discrete RedVLA adaptation with learned takeoff separation.

The old marker-displacement trigger counted the common takeoff as task motion.
This keeps localization/anchor/direction selection and replaces that trigger
with the shared external-only learned onset predictor, never private phase.
"""
import argparse
import base64
import json
import math
from pathlib import Path
import time
import urllib.request

import numpy as np

from experiments.task_attack.baselines import LOCALIZE_SYSTEM,SOURCES,closest_anchor,directional_update
from experiments.task_attack.event_timing import features
from experiments.task_attack.model import ModelClient
from experiments.task_attack.observer import _get
from experiments.task_attack.runtime import append_json,write_json
from .lookahead import onset_probability


def run(url,output,effort,model,call_budget=1):
    if call_budget!=1:raise ValueError('One shared observer request required')
    root=Path(output);root.mkdir(parents=True,exist_ok=True);catalog=_get(url,'catalog');window=_get(url,'initial')
    actions=catalog['editable_actions'];allowed={a['action_id'] for a in actions}
    client=ModelClient(root/'model_calls.jsonl',role='observer',effort=effort)
    summary=dict(policy='redvla_external',paper_adaptation=True,source=SOURCES['redvla_external'],
        implementation='redvla_external_learned_onset_v9',decision='localization_pending',calls=0,
        corrected_takeoff_trigger=True,not_full_paper_reproduction=True)
    write_json(root/'result.json',summary)
    response=client.request(LOCALIZE_SYSTEM,dict(policy='redvla_external',allowed_actions=actions,
        scene_conditions=catalog.get('scene_conditions'),public_task_prior=catalog.get('task_prior')),
        [dict(label='Actual INITIAL external RGB',data=base64.b64decode(window['frames'][0]['jpeg_base64']))],attempt_limit=1)
    placements=[]
    for p in response.get('placements',[]):
        center=p.get('center')
        if p.get('action_id') in allowed and isinstance(center,list) and len(center)==2 and all(
            isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=1 for v in center):placements.append(p)
    if not placements:raise ValueError('No actual-image localized carrier')
    summary.update(calls=client.calls,model_wall_seconds=client.wall_seconds,localization=response,decision='collecting_external_motion')
    write_json(root/'result.json',summary)
    while True:
        state=_get(url,'state')
        if state['finished'] or state['edit_used']:
            summary['decision']='episode_ended_before_behavior';write_json(root/'result.json',summary);return summary
        window=_get(url,'frames');history=window.get('external_pixel_history',[]);x=features(history,state['elapsed_seconds'])
        if x is not None and onset_probability(model,x)>=.5:break
        time.sleep(.25)
    # Limit the anchor trajectory to samples whose learned external evidence
    # supports autonomous task movement. Common ascent samples no longer select
    # the carrier just because they are the initial closest image pixels.
    points=[]
    for i,frame in enumerate(history):
        px=frame.get('robot_pixels')
        if not px:continue
        value=features(history[max(0,i-35):i+1],max(0.,frame['capture_ros']-history[0]['capture_ros']))
        if value is not None and onset_probability(model,value)>=.5:points.append(px['centroid'])
    if not points:raise ValueError('No image-track support for selected task motion')
    scored=[];trajectory=np.array(points)
    for item in placements:
        anchor=closest_anchor(trajectory,item['center']);adjusted=directional_update(item['center'],anchor,.05,
            valid=lambda p:bool(np.all(p>=0)&np.all(p<=1)))
        scored.append(dict(action_id=item['action_id'],anchor=anchor.tolist(),proposed_anchor_position=adjusted.tolist(),
            distance=float(np.linalg.norm(anchor-item['center']))))
    scored.sort(key=lambda r:(r['distance'],r['action_id']));action=scored[0]['action_id']
    summary.update(action_id=action,anchor_candidates=scored,observed_external_frame_ids=[f['id'] for f in window['frames']],
        actual_external_onset_probability=onset_probability(model,x))
    if _get(url,'state')['finished']:summary['decision']='episode_ended_before_dispatch'
    else:
        request=urllib.request.Request(url+'/edit',data=json.dumps(dict(action_id=action)).encode(),headers={'Content-Type':'application/json'},method='POST')
        with urllib.request.urlopen(request,timeout=8) as result:summary['acknowledgment']=json.load(result)
        summary['decision']='edit'
    append_json(root/'decisions.jsonl',dict(wall=time.time(),response=summary));write_json(root/'result.json',summary);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--gateway',required=True);p.add_argument('--policy',choices=('redvla_external',),required=True)
    p.add_argument('--output',required=True);p.add_argument('--effort',choices=('high','xhigh','max'),default='high')
    p.add_argument('--call-budget',type=int,default=1);p.add_argument('--timing-model',type=Path,required=True);a=p.parse_args();root=Path(a.output)
    try:run(a.gateway,root,a.effort,json.loads(a.timing_model.read_text()),a.call_budget)
    except (Exception,KeyboardInterrupt) as error:
        summary=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {}
        summary.update(decision='observer_error' if not isinstance(error,KeyboardInterrupt) else 'inference_interrupted_at_episode_end',
            error_type=type(error).__name__,error_code=getattr(error,'code',None),status=getattr(error,'status',None));write_json(root/'result.json',summary)
        if not isinstance(error,KeyboardInterrupt):raise


if __name__=='__main__':main()
