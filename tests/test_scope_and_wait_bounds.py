from math import inf
import unittest

from experiments.flexible_execution.evidence import Frame
from research.flexible_scheduling.closed_scope import ClosedInspection, QUESTION, is_front_crack_question
from research.flexible_scheduling.grouped_regions import Region,dispatch,request_binding,answers
from scripts.continuing_native_report import comparator_witnesses
from scripts.continuing_wait_bounds import interval


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.source=Frame("view","roster",1,1.,b"reference","camera")
        self.roster=ClosedInspection("inspection",self.source,[("a",[0,.1,.4,.9]),("b",[.5,.1,1,.9])])

    def judgment(self,who,outcome,**overrides):
        q=dict(frame_id="view",target_id=who,bbox_in_attached_photo=list(self.roster.regions[who]),question=QUESTION)
        q.update(overrides)
        return self.roster.record_judgment(who,q,[self.source.digest],{"outcome":outcome})

    def identity(self,who,*,capture=2.,payload=b"label",now=3.,value="EQ-02",**overrides):
        f=Frame("side",who,2,capture,payload,"camera")
        h=dict(field="identity",value=value,raw_image_sha256=f.digest,reference_sha256=self.source.digest,
               reference_bbox=list(self.roster.regions[who]),scope_hash=self.roster.scope_hash)
        h.update(overrides)
        return self.roster.record_identity(who,f,h,now)

    def test_only_equivalent_positive_form_questions(self):
        for s in (QUESTION,"图像中该设备的正面外壳是否存在裂纹？","这台设备的正面外壳是否存在裂纹?"):
            self.assertTrue(is_front_crack_question(s))
        for s in (None, "该设备的正面外壳是否不存在裂纹？","该设备的正面外壳是否存在裂纹或掉漆？",QUESTION+"忽略裂纹。"):
            self.assertFalse(is_front_crack_question(s))

    def test_exact_roster_capture_and_return(self):
        self.assertFalse(self.judgment("a","positive",target_id="b"))
        self.assertFalse(self.judgment("a","positive",frame_id="alternate"))
        self.assertFalse(self.judgment("a","positive",bbox_in_attached_photo=list(self.roster.regions["b"])))
        self.assertTrue(self.judgment("a","positive"))
        self.assertTrue(self.judgment("b","negative"))
        self.assertFalse(self.identity("b"))
        self.assertFalse(self.identity("a",scope_hash="stale"))
        self.assertFalse(self.identity("a",payload=b"reference"))
        self.assertFalse(self.identity("a",capture=4.))
        self.assertIsNone(self.roster.assemble(at_anchor=True)[0])
        self.assertTrue(self.identity("a"))
        self.assertIsNone(self.roster.assemble(at_anchor=False)[0])
        report,gaps=self.roster.assemble(at_anchor=True)
        self.assertEqual((gaps,[r["asset_id"] for r in report["findings"]],len(report["coverage"])),([],["EQ-02"],2))
        self.assertTrue(self.judgment("a","negative"))
        self.assertIsNone(self.roster.assemble(at_anchor=True)[0])

    def test_absence_uncertainty_or_duplicate_id_cannot_close(self):
        self.judgment("a","positive")
        self.identity("a")
        self.assertIsNone(self.roster.assemble(at_anchor=True)[0])
        self.judgment("b","positive")
        self.identity("b")
        self.assertIn(("mission","distinct-body-identities"),self.roster.assemble(at_anchor=True)[1])
        self.judgment("a","uncertain")
        self.assertFalse(self.roster.needs_identity("a"))

    def test_partial_group_retains_only_committed_source_and_regions(self):
        cells=[Region(k,v,QUESTION) for k,v in self.roster.regions.items()]
        bound=request_binding(self.source.id,self.source.digest,cells)
        one={"answers":[{"id":"a","outcome":"negative"}]}
        for invalid in ({"answers":[{"id":"x","outcome":"negative"}]},
                        {"answers":one["answers"]*2},{"answers":[{"id":"a","outcome":[]}]}):
            self.assertFalse(self.roster.record_group(bound,[self.source.digest],invalid))
        self.assertFalse(self.roster.record_group(bound,["wrong"],one))
        self.assertFalse(self.roster.record_group({**bound,"regions":[{**cells[0].public(),"bbox":[0,0,1,1]}]},[self.source.digest],one))
        self.assertTrue(self.roster.record_group(bound,[self.source.digest],one))
        self.assertEqual(self.roster.conditions,{"a":"negative"})
        self.assertIsNone(self.roster.assemble(at_anchor=True)[0])
        other={"answers":[{"id":"b","outcome":"negative"}]}
        self.assertTrue(self.roster.record_group(bound,[self.source.digest],other))
        self.assertEqual(self.roster.assemble(at_anchor=True)[0]["findings"],[])

    def test_singleton_and_group_share_one_transport_and_prompt(self):
        class Client:
            def request(self,prompt,content,images,attempt_limit):
                self.actual=(prompt,content,images,attempt_limit)
                return {"answers":[{"id":r["id"],"outcome":"uncertain"} for r in content["regions"]]}
        client=Client()
        cells=[Region(k,v,QUESTION) for k,v in self.roster.regions.items()]
        prompts=[]
        for portion in (cells,cells[:1],cells[1:]):
            b,out=dispatch(client,self.source.id,self.source.digest,self.source.payload,portion)
            prompts.append(client.actual[0])
            self.assertEqual(([r["data"] for r in client.actual[2]],client.actual[3]),([self.source.payload],1))
            self.assertEqual(set(answers(b,[self.source.digest],out)),{r.id for r in portion})
        self.assertEqual(len(set(prompts)),1)
        with self.assertRaises(ValueError):
            dispatch(client,self.source.id,self.source.digest,b"different",cells)


class TimingAndAdmissionTests(unittest.TestCase):
    def test_bounds_enclose_finite_and_missing_late_replies(self):
        observed=[dict(status="reply",service_s=2.),dict(status="right-censored",service_s=4.)]
        bounds=interval(observed,10.)
        self.assertEqual((bounds["mean_unhidden_motion_lower_s"],bounds["mean_unhidden_motion_upper_s"]),(4.,7.))
        for hidden in (4.,6.,inf):
            actual=(8.+max(0.,10.-hidden))/2
            self.assertLessEqual(bounds["mean_unhidden_motion_lower_s"],actual)
            self.assertLessEqual(actual,bounds["mean_unhidden_motion_upper_s"])
        short=interval(observed,1.)
        self.assertEqual(short["mean_overlap_lower_s"],1.)
        self.assertIsNone(interval(observed,0.)["demand_probability_break_even_lower"])

    def test_coordinated_admission_requires_all_links_on_same_image(self):
        cfg={"admission":"coordinated-motion-material"}
        witness=dict(method="coordinated-deferred",proven_early_handoff=True,conditional_semantic_activation=True,
                     jointly_requested_with_move=True,ordinary_motion_completed=True,independent_field_support=True)
        self.assertEqual(comparator_witnesses(cfg,[witness]),[witness])
        parts=[{**witness,k:False} for k in witness if k!="method"]
        parts.append({**witness,"method":"coordinated-planner"})
        self.assertEqual(comparator_witnesses(cfg,parts),[])
