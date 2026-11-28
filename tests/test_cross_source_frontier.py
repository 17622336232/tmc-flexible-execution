import unittest
from unittest.mock import Mock,patch

from research.flexible_scheduling.cross_source_workflow import CausalMeasurements, CrossSourcePlanner, CrossSourceWorkflow, PermissionCache
from research.flexible_scheduling.event_workflow import Dispatch, Mission, Query, State, UNKNOWN
from research.flexible_scheduling.natural_frontier import discover, read_ordinary_records, released, source_map
from research.flexible_scheduling.native_frontier_adapter import run_shadow


class CrossSourceFrontier(unittest.TestCase):
    def test_requires_both_camera_sources_and_previous_gate(self):
        queries = (Query("front root", (0,)), Query("side root", (1,)),
                   Query("front followup", (0,), (((1, True),),)))
        m = Mission(("front","side"), queries, ((0,10),(10,0)), (1,1), (True,True), 0, 1, (0,8,15), 0, 1)
        graph = CrossSourceWorkflow(m, 10)
        initial = graph.initial()
        after_capture = State(initial.values,3,1)
        merged = Dispatch((), ((2,3),))
        self.assertNotIn(merged,graph.dispatches(initial))
        self.assertIn(merged,graph.dispatches(after_capture))
        self.assertNotIn(Dispatch((), ((2,6),)),graph.dispatches(after_capture))
        elapsed, outcomes = graph.successors(after_capture,merged)
        self.assertEqual(elapsed,10)
        self.assertTrue(all(t.values[2] == UNKNOWN for _,t in outcomes))
        generic = CrossSourcePlanner(graph,quotient=False).solve(initial)
        specialized = CrossSourcePlanner(graph).solve(initial)
        self.assertEqual(generic["value_ticks"],specialized["value_ticks"])

    def test_requests_do_not_train_estimates_before_settlement(self):
        c = CausalMeasurements()
        c.record("semantic",available_at=3,seconds=7,delivered=True)
        c.record("semantic",available_at=5,seconds=120,delivered=False)
        self.assertEqual(c.describe("semantic",2,20)["successful_median_s"],20)
        self.assertEqual(c.describe("semantic",4,20)["capped_median_s"],7)
        self.assertEqual(c.describe("semantic",5,20)["undelivered"],1)
        self.assertEqual(c.describe("semantic",5,20)["successful_median_s"],7)
        self.assertEqual(c.conditional_remaining("semantic",5,8,120)["deliveries"],0)
        self.assertIsNone(c.conditional_remaining("semantic",5,8,120)["median_until_reply_s"])

    def test_permission_cache_reuses_a_branch_without_reusing_old_prices(self):
        queries = (Query("a",(0,)), Query("b",(1,)), Query("c",(0,),(((0,True),),((1,True),))),
                   Query("d",(1,)))
        m = Mission(("front","side"),queries,((0,1),(1,0)),(1,1),(True,True),0,3,(0,5,8),1,1)
        cache = PermissionCache(m,True,residual=True)
        e, revised = [CrossSourceWorkflow(m,n,permission_cache=cache) for n in (6,11)]
        first = State((0,1,UNKNOWN,UNKNOWN),3,1)
        second = State((1,0,UNKNOWN,UNKNOWN),3,1)
        self.assertEqual(set(e.dispatches(first)),set(revised.dispatches(second)))
        self.assertEqual(cache.hits,1)
        proposal = Dispatch((),((2,12),))
        self.assertEqual(e.successors(first,proposal)[0],6)
        self.assertEqual(revised.successors(second,proposal)[0],11)

    def test_unnumbered_natural_task_and_native_records(self):
        task = ("正面是起始朝向。原地向左转40度为侧面。看看正面照片里有没有红色货箱。也检查侧面照片中是否有蓝色货箱。"
                "如果侧面有蓝色货箱，才判断正面照片里有无白色货箱。允许在等待时提前拍摄侧面照片。最后停在侧面。")
        plan = discover(task)
        self.assertTrue(plan.early_capture)
        self.assertEqual(released(plan,{}),{0,1})
        self.assertEqual(released(plan,{0:False,1:True}),{2})
        self.assertEqual(released(plan,{0:True,1:False}),set())
        self.assertIsNone(discover(task+"不要提前拍摄任何照片。"))
        pose = dict(position=dict(x=1,y=2,z=3),yaw=.4)
        front = dict(image_id="camera_01",**pose)
        side = dict(image_id="camera_02",position=pose["position"],yaw=.4+.6981317)
        sources = source_map(plan,pose,[front,side])
        rows = [dict(object_id=name,evidence_image_id=src,facts=dict(visible=True))
                for name,src in (("红色货箱","camera_01"),("蓝色货箱","camera_02"),("白色货箱","camera_01"))]
        self.assertEqual(read_ordinary_records(plan,dict(records=rows),{},sources),{0:True,1:True})
        self.assertEqual(read_ordinary_records(plan,dict(records=rows),{},{"camera_01":"side"}),{})

    def test_native_integration_uses_one_unmodified_request(self):
        task = ("正面是起始朝向。原地向左转40度为侧面。看看正面照片里有没有红色货箱。也检查侧面照片里有没有蓝色货箱。"
                "允许提前拍摄侧面照片。最后停在侧面。")
        context = dict(authorized_task=task,current_image_id="front",
                       geometry=dict(position=dict(x=0,y=0,z=1),yaw=0))
        images = [dict(data=b"measured-camera")]
        reply = dict(records=[],action=dict(type="move"))
        model = Mock()
        model.request.__name__ = "request"
        model.request.return_value = reply
        original = model.request
        trace = Mock()

        def native(*_,model,trace):
            return model.request("original instructions",context,images)

        with patch("research.flexible_scheduling.native_frontier_adapter.original_controller.run",side_effect=native):
            self.assertIs(run_shadow(None,task,None,None,240,model=model,trace=trace),reply)
        original.assert_called_once_with("original instructions",context,images)
        self.assertIs(model.request,original)
        self.assertEqual(trace.event.call_args.args[:2],("natural_frontier_observed","front"))


if __name__ == "__main__":
    unittest.main()
