"""Original frozen trained visual carriers under the shared static scene setup.

The same initial image anchors select a central visible station. No additional
live API delay is imposed. Carrier texture, scale and SPSA objective remain
frozen; these are API/renderer-transfer adaptations, not original paper scores.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request

from experiments.task_attack.baselines import SOURCES
from experiments.task_attack.visual_assets import KINDS
from experiments.task_attack.observer import _get
from experiments.task_attack.runtime import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--gateway',required=True);p.add_argument('--policy',choices=tuple(KINDS),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--effort',default='high');p.add_argument('--call-budget',type=int,default=1)
    p.add_argument('--public-perception',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    catalog=_get(a.gateway,'catalog');perception=json.loads(a.public_perception.read_text())
    actions=[item for item in catalog['editable_actions'] if item.get('kind')==KINDS[a.policy]]
    if not actions:raise ValueError('No trained physical carrier')
    # Largest actual static body offers the strongest image footprint. The
    # rule is shared across all three fixed universal-texture adaptations.
    def area(action):
        source=int(action['action_id'].rsplit('_',1)[1])-1;box=perception['stations'][source].get('body_bbox')
        return (box[2]-box[0])*(box[3]-box[1]) if box else 0.
    action=sorted(actions,key=lambda item:(-area(item),item['action_id']))[0]
    summary=dict(policy=a.policy,paper_adaptation=True,source=SOURCES[a.policy],calls=0,
        shared_static_preparation_requests=1,implementation='frozen_visual_texture_shared_static_scene_v13',
        action_id=action['action_id'],decision='dispatching',not_original_paper_reproduction=True)
    write_json(a.output/'result.json',summary)
    while True:
        state=_get(a.gateway,'state')
        if state['finished'] or state['edit_used']:summary['decision']='episode_ended_before_dispatch';break
        if not state['started']:time.sleep(.1);continue
        request=urllib.request.Request(a.gateway+'/edit',data=json.dumps(dict(action_id=action['action_id'])).encode(),headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(request,timeout=8) as result:summary['acknowledgment']=json.load(result)
            summary['decision']='edit'
        except Exception as error:summary.update(decision='edit_failed_or_late',error_type=type(error).__name__)
        break
    write_json(a.output/'result.json',summary)


if __name__=='__main__':main()
