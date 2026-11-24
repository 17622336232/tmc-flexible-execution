import copy
import unittest

from experiments.task_attack_realtime.belief import rank,validate
from experiments.task_attack_realtime.continuation import forecast,nearest,summarize


class ContinuationTests(unittest.TestCase):
    def model(self):
        sample=[0.,.3,.5,.02,.02,0.,0.,0.,0.,0.,1.,0.]
        return dict(mean=sample[1:],scale=[1.]*11,neighbors=1,pixel_std=[.01,.01],episodes=[
            dict(seed=1,scene='equipment',samples=[sample],reads=[[1.,.3,.5]],end=2.),
            dict(seed=2,scene='equipment',samples=[[3.,*sample[1:]]],reads=[[1.,.3,.5]],end=4.)])

    def prediction(self,point,future=True):
        return dict(rollouts=[dict(weight=1.,future=[[2.,*point]] if future else [],past=[])],
            pixel_std=[.01,.01],probability_any_future_read=float(future))

    def test_terminal_prefix_has_zero_future_read_mass(self):
        model=self.model();model['episodes']=model['episodes'][1:]
        pred=forecast(model,[],0.,'equipment',False)
        self.assertEqual(pred['probability_any_future_read'],0.)
        self.assertEqual(summarize(pred,None)['any_future_region_probability'],0.)

    def test_blind_matching_cannot_use_current_motion(self):
        model=self.model();a=model['mean'][:];b=a[:];b[0]=.99;b[6]=.8
        left=nearest(model,a,'equipment',False);right=nearest(model,b,'equipment',False)
        self.assertEqual([(x['episode']['seed'],x['weight']) for x in left],
            [(x['episode']['seed'],x['weight']) for x in right])

    def test_no_observation_means_no_unsupported_positive_attack(self):
        response={'b':[dict(p=1.,r=None,cached=0.)],'q':[[0,[.9],[0.]]]},
        # A properly formed positive model forecast still loses its value
        # when the empirical continuation has no remaining read opportunity.
        response=response[0];pred=self.prediction([.3,.5],False)
        ranks,audit=rank(response,[dict(action_id='flip_panel_01')],pred,pred,True)
        self.assertEqual(ranks[0]['expected_incremental_wrong_acceptance'],0.)

    def test_motion_can_change_selected_carrier_without_new_api(self):
        response={'b':[dict(p=.5,r=[.2,.4,.4,.6],cached=0.),dict(p=.5,r=[.6,.4,.8,.6],cached=0.)],
            'q':[[0,[.8,0.],[0.,0.]],[1,[0.,.8],[0.,0.]]]}
        menu=[dict(action_id='left'),dict(action_id='right')]
        prior=dict(rollouts=[dict(weight=.5,future=[[2.,.3,.5]],past=[]),
            dict(weight=.5,future=[[2.,.7,.5]],past=[])],pixel_std=[.01,.01],probability_any_future_read=1.)
        left,_=rank(response,menu,self.prediction([.3,.5]),prior,True)
        right,_=rank(response,menu,self.prediction([.7,.5]),prior,True)
        self.assertEqual(left[0]['action_id'],'left');self.assertEqual(right[0]['action_id'],'right')

    def test_identical_dynamic_and_static_do_not_add_area_bias(self):
        response={'b':[dict(p=.5,r=[0.,0.,1.,1.],cached=0.),dict(p=.5,r=[.29,.49,.31,.51],cached=0.)],
            'q':[[0,[.1,.1],[.1,.1]]]}
        pred=self.prediction([.3,.5]);_,audit=rank(response,[dict(action_id='a')],pred,pred,True)
        self.assertEqual(audit['posterior'],[.5,.5])

    def test_malformed_or_outside_menu_forecast_is_rejected(self):
        response={'b':[dict(p=1.,r=None,cached=0.)],'q':[[2,[1.],[0.]]]}
        with self.assertRaises(ValueError):validate(response,[dict(action_id='only')])
        response['q']=[[0,[float('nan')],[0.]]]
        with self.assertRaises(ValueError):validate(response,[dict(action_id='only')])


if __name__=='__main__':unittest.main()
