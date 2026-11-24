import unittest

from experiments.task_attack_cached.lookahead import rank


class SourceOpportunityTests(unittest.TestCase):
    def prediction(self,source_map):
        return dict(rollouts=[dict(weight=1.,future=[[3.,.4,.6,source_map]],past=[])],
            pixel_std=[.01,.01],probability_any_future_read=1.)

    def test_new_camera_without_edited_carrier_does_not_have_positive_value(self):
        response={'b':[dict(p=1.,r=None,cached=0.)],'q':[[0,[.8],[0.]]]}
        pred=self.prediction({'flip_panel_01':0.})
        ranks,audit=rank(response,[dict(action_id='flip_panel_01')],pred,pred,False)
        self.assertEqual(ranks[0]['expected_incremental_wrong_acceptance'],0.)

    def test_unknown_carrier_model_is_not_fabricated_as_invisible(self):
        response={'b':[dict(p=1.,r=None,cached=0.)],'q':[[0,[.8],[0.]]]}
        pred=self.prediction({})
        ranks,audit=rank(response,[dict(action_id='unsupported_texture')],pred,pred,False)
        self.assertEqual(ranks[0]['expected_incremental_wrong_acceptance'],.8)
        self.assertIsNone(ranks[0]['predicted_source_frustum_opportunity'])


if __name__=='__main__':unittest.main()
