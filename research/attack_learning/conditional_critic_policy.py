"""Public-input online policy for the explicitly approximate critic candidate.

This module never opens a scene, true-task file, robot log, onboard frame or
private pose. Model loading is the observer's responsibility. All goal
arguments below are candidate hypotheses except the authorized known control.
"""
import numpy as np

from experiments.task_attack_value.forest import predict
from experiments.task_attack_value.value_features import action_features
from experiments.task_attack_value.vision import candidate_features
from experiments.task_attack_value.programs import payload
from learn_read_dependency import quality_history,context,goal_inputs
from crossfit_dependency_stopping_data import stopping_features

MODES={'belief':'ours_critic','belief_blind':'blind','direct_effect':'direct','belief_known_task':'known_critic'}


def public_events(history,pose_heads,origin):
    result=[];queries=[];indices=[]
    for frame in history:
        pixels=frame.get('robot_pixels')
        if not pixels:continue
        u,v=pixels['centroid'];box=pixels['bbox'];w=box[2]-box[0];h=box[3]-box[1]
        raw=frame.get('rgb_orientation_features');value=[u,v,w,h,0.,0.];mask=[True]*4+[False]*2
        if raw is not None:
            queries.append(np.r_[raw,u,v,w,h]);indices.append(len(result));mask[4:]=[True,True]
        result.append(dict(t=max(0.,frame['capture_ros']-origin),value=value,mask=mask,id=frame['id']))
    if queries:
        headings=np.column_stack([2*predict(head,np.array(queries))-1 for head in pose_heads])
        for i,heading in zip(indices,headings):result[i]['value'][4:]=heading.tolist()
    return result


class Policy:
    def __init__(self,model,stopping,pose_heads,perception,actions,policy,known=None):
        self.model=model;self.stopping=stopping;self.pose_heads=pose_heads;self.perception=perception;self.actions=[a for a in actions if payload(a)];self.arm=MODES[policy];self.known=known
        if (self.arm=='known_critic')!=(known is not None):raise ValueError('Information condition mismatch')
        self.first=None;self.origin=None
        self.model['intent'].n_jobs=1;self.model['quality'].forest.n_jobs=1;self.stopping.n_jobs=1

    def decide(self,history,elapsed,budget):
        actual=sorted(history,key=lambda f:(f['capture_ros'],f['id']))
        if not actual or not actual[0].get('robot_pixels'):return dict(kind='wait',reason='No actual initial RGB robot observation')
        if self.first is None:self.first=dict(actual[0]);self.origin=self.first['capture_ros']
        used=[self.first] if self.arm=='blind' else actual
        events=public_events(used,self.pose_heads,self.origin)
        if not events:return dict(kind='wait',reason='No actual public RGB body descriptor')
        # Public elapsed has already been expressed relative to the first
        # actually received external image by the caller, matching training.
        qs=quality_history(events,self.perception,self.model['quality']);ports,clock=context(events,qs,0. if self.arm=='blind' else elapsed,self.perception)
        hypotheses=self.model['hypotheses'];prior=self.model['prior']
        if self.arm=='ours_critic':
            raw=np.clip(self.model['intent'].predict(goal_inputs(ports,clock,hypotheses)),1e-9,1);posterior=raw/raw.sum()
        else:posterior=prior
        x=candidate_features(used,elapsed,self.perception['anchors']);static=candidate_features([self.first],elapsed,self.perception['anchors'])
        if x is None or static is None:return dict(kind='wait',reason='No actual public motion descriptor')
        public_features,_=action_features(static if self.arm=='blind' else x,self.perception,self.actions)
        if self.arm=='blind':values=predict(self.model['heads']['static'],public_features)
        elif self.arm=='direct':values=predict(self.model['heads']['dynamic'],public_features)
        else:
            goals=[self.known] if self.arm=='known_critic' else hypotheses
            inputs=np.vstack([action_features(x,self.perception,self.actions,g)[0] for g in goals])
            conditional=predict(self.model['heads']['known_dynamic'],inputs).reshape(len(goals),len(self.actions))
            values=conditional[0] if self.arm=='known_critic' else posterior.dot(conditional)
        order=sorted(range(len(self.actions)),key=lambda i:(-values[i],self.actions[i]['action_id']));index=order[0];action=self.actions[index]
        visible_clock=np.r_[np.log1p(elapsed),0.,clock[2:]] if self.arm=='blind' else clock
        vector=stopping_features(public_features[index],ports,visible_clock,values,index,action,posterior,hypotheses,self.known)
        planned=self.stopping.predict(np.array(vector)[None])[0];now=float(planned[0]);wait=float(max(planned[1:]))
        kind='wait' if wait>now else 'edit' if now>0 else 'abstain'
        ranked=[dict(action_id=self.actions[i]['action_id'],counterfactual_proxy_value=float(values[i])) for i in order]
        return dict(kind=kind,reason='Learned stop/wait values from OWN world-cross-fitted development trajectories',ranked_candidates=ranked,
            immediate_proxy=float(now),wait_proxy=float(wait),information_condition=self.arm,approximate_conditional_critic=True,full_paper_KDE_reproduction=False,
            hypothesis_weights=posterior.tolist() if self.arm=='ours_critic' else None,
            observed_body_events=len(events),actual_rgb_heading_events=sum(e['mask'][4] for e in events))
