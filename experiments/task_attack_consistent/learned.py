"""Compare prospective gains with NOW scored by the SAME crossfit weights.

The frozen parent compares a full-data immediate model to smaller fold-model
forecasts. Their different calibration can look like a reason to act early.
Here each branch subtracts that fold's current estimate before averaging its
gain. Actual candidate selection still uses the frozen full-data model.
"""
import numpy as np

from experiments.task_attack_value.learned import Policy as ParentPolicy


class Policy(ParentPolicy):
    def decide_features(self,x,static,elapsed,budget,information):
        self.information_weight=information if self.dynamic else 0.
        ranked=self.score(x,static_x=static);now=ranked[0]['expected_wrong_accepted'] if ranked else 0.
        neighbors=self.neighbors(x if self.dynamic else static);requests=[];current_request={};future_request={};branches=[]
        for weight,index,episode in neighbors:
            fold=str(episode['fold'])
            if fold not in current_request:
                current_request[fold]=len(requests)
                requests.append(dict(groups=self.model['crossfit_value_models'][fold],x=x,static=static,information=self.information_weight))
        delays=[d for d in (3.,6.,12.,24.,48.) if elapsed+d<budget]
        for delay in delays:
            for e,(weight,index,episode) in enumerate(neighbors):
                target=episode['samples'][index]['t']+delay
                future=next((s for s in episode['samples'][index:] if s['t']>=target),None)
                if future is None:continue
                future_request[delay,e]=len(requests)
                requests.append(dict(groups=self.model['crossfit_value_models'][str(episode['fold'])],
                    x=self.retime(future['x' if self.dynamic else 'static'],elapsed+delay),
                    static=self.retime(future['static'],elapsed+delay),
                    information=future['informative_probability'] if self.dynamic else 0.))
        values=self.score_requests(requests) if requests else []
        for delay in delays:
            gain=0.;old_absolute=0.;fold_now=0.
            for e,(weight,index,episode) in enumerate(neighbors):
                current=values[current_request[str(episode['fold'])]]
                future=values[future_request[delay,e]] if (delay,e) in future_request else 0.
                gain+=weight*(future-current);old_absolute+=weight*future;fold_now+=weight*current
            if neighbors:branches.append(dict(delay_seconds=delay,expected_wrong_accepted=float(np.clip(now+gain,0.,1.)),
                comparable_crossfit_gain=float(gain),crossfit_now=float(fold_now),crossfit_future=float(old_absolute),branches=len(neighbors)))
        best=max(branches,key=lambda item:item['comparable_crossfit_gain']) if branches else None
        if best and best['comparable_crossfit_gain']>1e-9:
            kind='wait';reason='Same-model future gain is positive after independent branch calibration'
        elif now>0:kind='edit';reason='Immediate learned value is at least as high as comparable waiting value'
        else:kind='abstain';reason='Allowed edits have no learned factual value'
        return dict(kind=kind,reason=reason,ranked_candidates=ranked,immediate_value=now,wait_branches=branches,
                    learned_task_dependent_evidence_probability=self.information_weight,
                    selected_delay_seconds=best['delay_seconds'] if best and kind=='wait' else None,
                    information='actual external JPEG and motion prefix' if self.dynamic else 'initial external JPEG and public clock',
                    future_comparison='Cross-fitted future minus SAME fold current; full-model immediate used only as common reference')
