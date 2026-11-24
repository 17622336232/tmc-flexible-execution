"""Original local trajectory-anchor adaptation with the common static head.

Still an adaptation of RedVLA, not its original implementation. The behavior
criterion, onset threshold and nearest-anchor ranking are preserved. Only
the common-resource payload ranker and shared public RGB collector are new.
"""
import numpy as np
from experiments.task_attack.baselines import closest_anchor,directional_update
from experiments.task_attack.event_timing import features
from experiments.task_attack_cached.lookahead import onset_probability
from experiments.task_attack_value.programs import payload
from experiments.task_attack_value.value_features import action_features
from experiments.task_attack_value.vision import candidate_features
from experiments.task_attack_value.forest import predict


class Policy:
    def __init__(self,model,legacy,perception,actions):
        self.model=model;self.legacy=legacy;self.perception=perception;self.actions=[a for a in actions if payload(a)];self.first=None
    def decide(self,history,elapsed,budget):
        if not history:return dict(kind='wait',reason='No actual public RGB prefix')
        if self.first is None:self.first=dict(history[0])
        x=features(history,elapsed)
        if x is None or onset_probability(self.legacy,x)<.5:return dict(kind='wait',reason='Original learned task-motion onset criterion')
        static=candidate_features([self.first],elapsed,self.perception['anchors'])
        if static is None:return dict(kind='wait',reason='No actual initial body descriptor')
        xs,_=action_features(static,self.perception,self.actions);values=predict(self.model['heads']['static'],xs);ranks=[]
        for action,value in zip(self.actions,values):ranks.append(dict(action_id=action['action_id'],expected_wrong_accepted=float(value)))
        ranks.sort(key=lambda r:(-r['expected_wrong_accepted'],r['action_id']));by_source={};by_id={a['action_id']:a for a in self.actions}
        for rank in ranks:by_source.setdefault(payload(by_id[rank['action_id']])['source'],rank)
        points=[]
        for i,frame in enumerate(history):
            if not frame.get('robot_pixels'):continue
            value=features(history[max(0,i-35):i+1],max(0.,frame['capture_ros']-history[0]['capture_ros']))
            if value is not None and onset_probability(self.legacy,value)>=.5:points.append(frame['robot_pixels']['centroid'])
        if not points:return dict(kind='wait',reason='Original trajectory anchor not yet observed')
        scored=[]
        for source,rank in by_source.items():
            center=np.array([self.perception['anchors'][source]['u'],self.perception['anchors'][source]['v']]);anchor=closest_anchor(points,center)
            adjusted=directional_update(center,anchor,.05,valid=lambda p:bool(np.all(p>=0)&np.all(p<=1)))
            scored.append(dict(rank,source=source,anchor=anchor.tolist(),proposed_anchor_position=adjusted.tolist(),distance=float(np.linalg.norm(anchor-center))))
        scored.sort(key=lambda r:(r['distance'],-r['expected_wrong_accepted'],r['action_id']))
        return dict(kind='edit',reason='Original external trajectory-anchor ranking with common-resource static payload head',ranked_candidates=scored,paper_adaptation=True,full_paper_reproduction=False)
