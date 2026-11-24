"""One shared STATIC pre-task RGB interpretation; no task-conditioned input.

Interpretation is prepared before paired missions. Live attack inference uses
the learned model. All information conditions receive exactly this same file.
World assets are hashed by the trusted setup only; their contents never become
inputs to the observer or to the perception API.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from experiments.task_attack.model import ModelClient
from experiments.task_attack.runtime import write_json
from .vision import anchors_from_rgb

SYSTEM='''Interpret only the supplied actual STATIC external scene image for a
researcher-owned robot robustness experiment. No current task has been supplied.
The robot has only performed a common sensor/takeoff check, not a user task.
Station handles in this public scene are ordered LEFT TO RIGHT. Read their
visible identifiers and properties only when the image actually supports them.
Return null for unreadable text, hidden surfaces, temperatures or defects.
Never copy an unplaced intervention's printed value into a scene observation.
Do not infer the task or provide an attack decision or a success probability.
Return ONLY {"stations":[{"index":1,"identity":"visible ID or null",
"cracked":true_or_false_or_null,"temperature":number_or_null,
"label_bbox":[l,t,r,b],"body_bbox":[l,t,r,b],
"shell_center":[u,v]_or_null,"display_center":[u,v]_or_null}]}.
All positions are normalized actual image coordinates. Every station must have
one entry; preserve uncertainty for hidden right-side readouts. Image coordinates
are not world coordinates. Do not invent unseen geometry or measurements.
'''


def scene_signature(directory):
    directory=Path(directory).resolve();text=(directory/'inspection.world').read_text()
    text=text.replace(directory.as_uri(),'ASSET_ROOT').replace(str(directory),'ASSET_ROOT')
    digest=hashlib.sha256(text.encode())
    for path in sorted((directory/'assets/materials').rglob('*')):
        if path.is_file():digest.update(str(path.relative_to(directory/'assets')).encode()+b'\0'+path.read_bytes())
    return digest.hexdigest()


def validate(data,count,ids):
    if not isinstance(data,dict) or not isinstance(data.get('stations'),list) or len(data['stations'])!=count:raise ValueError('Need all visible station entries')
    found={}
    for station in data['stations']:
        index=station.get('index')
        if isinstance(index,bool) or not isinstance(index,int) or not 1<=index<=count or index in found:raise ValueError('Invalid static station handle')
        if station.get('identity') is not None and station['identity'] not in ids:raise ValueError('Identity is not a public candidate or unknown')
        if station.get('cracked') is not None and not isinstance(station['cracked'],bool):raise ValueError('Defect state must be boolean or unknown')
        value=station.get('temperature')
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):raise ValueError('Temperature must be a finite reading or unknown')
        for key in ('label_bbox','body_bbox','shell_center','display_center'):
            value=station.get(key)
            if value is None:continue
            if not isinstance(value,list) or len(value)!=(4 if key.endswith('bbox') else 2) or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1 for x in value):raise ValueError('Invalid actual image geometry')
            if key.endswith('bbox') and (value[2]<=value[0] or value[3]<=value[1]):raise ValueError('Empty actual image rectangle')
        found[index]=station
    return [found[index] for index in range(1,count+1)]


def prepare(directory,output):
    directory=Path(directory);output=Path(output);catalog=json.loads((directory/'public/catalog.json').read_text())
    count=len([a for a in catalog['editable_actions'] if re.fullmatch(r'flip_panel_\d+',a['action_id'])])
    ids=sorted({a['back_text'] for a in catalog['editable_actions'] if a.get('back_text')})
    image=directory/'external_sensor_smoke.jpg'
    if not image.exists():
        # Reuse only the actual first pre-task external capture in OWN
        # development episodes. Verify it predates task execution; this setup
        # audit is never supplied to the perception model or online observer.
        frames=directory/'public/external_frames.jsonl';events=directory/'private/events.jsonl'
        first=json.loads(frames.read_text().splitlines()[0]);event_rows=[json.loads(line) for line in events.read_text().splitlines()]
        takeoff=next(row for row in event_rows if row['event']=='takeoff_requested')
        if first['arrival_wall']>=takeoff['wall']:raise ValueError('Static preparation image must predate even the common takeoff')
        image=directory/'public/external'/('%06d.jpg'%first['id'])
    if not image.exists():raise ValueError('Preparation requires an actual pre-task RGB capture, never a generated world overview')
    raw=image.read_bytes();client=ModelClient(output.with_suffix('.calls.jsonl'),role='observer',effort='high')
    result=client.request(SYSTEM,dict(public_station_count=count,public_identifier_candidates=ids,
        physical_source_layout=catalog['scene_conditions'].get('observation_layout')),
        [dict(label='Actual STATIC external RGB; common sensor check only, no task-dependent behavior',data=raw)],attempt_limit=1)
    stations=validate(result,count,ids);anchors=anchors_from_rgb(raw,count);source='image_color_components'
    if anchors is None:
        anchors=[];source='actual_image_VLM_rectangles'
        for station in stations:
            box=station.get('label_bbox')
            if box is None:raise ValueError('Ambiguous static anchors; no world-coordinate fallback')
            anchors.append(dict(u=(box[0]+box[2])/2,v=(box[1]+box[3])/2,w=box[2]-box[0],h=box[3]-box[1]))
    prepared=dict(version=1,current_task_used=False,current_robot_internal_state_used=False,
        shared_across_information_conditions=True,pre_task_static=True,stations=stations,anchors=anchors,
        anchor_source=source,scene_signature=scene_signature(directory),
        actual_static_rgb_sha256=hashlib.sha256(raw).hexdigest(),perception_model='gpt-6.1-terra',reasoning_effort='high',
        purpose='Shared scene-only preparation; not a current onboard observation and not a task/attack decision')
    write_json(output,prepared);return prepared


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=prepare(a.episode,a.output);print('Static scene prepared from actual external RGB; no current task used; stations',len(result['stations']))
