"""Regression checks for demonstrated failure modes and branch-preserving progress."""
import unittest
from types import SimpleNamespace

from experiments.flexible_execution.adapters.v2_execution import COMPILER, compile_requirements
from experiments.flexible_execution.executor import Snapshot
from experiments.flexible_execution.frontend_evidence import atomic_facts
from experiments.flexible_execution.live_planning import permission_for, select
from experiments.flexible_execution.task_ir import Condition, QuerySpec, Requirement, Skill, TaskContract


class ResearchRevisionTest(unittest.TestCase):
    def test_optional_branch_preserves_mandatory_typed_task(self):
        p = {"scope":{"mode":"all","description":"设备"}, "fields":{"temperature":"number"},
             "filter":{"op":"keep"}, "extreme":{"op":"min","field":"temperature"}, "report":["temperature"]}
        saved = {"program":p, "conditional":{"status":"unsupported","reason":"requires comparing objects"}}
        model = SimpleNamespace(request=lambda *args: saved)
        program, intent = compile_requirements(model, "检查所有设备，报告最低温度")
        self.assertIsNone(intent)
        self.assertNotIn('return {"status":"unsupported"', COMPILER.lower())
        matches = program.evaluate({"X":{"temperature":20}, "Y":{"temperature":20},
                                    "Z":{"temperature":21}}, scope_confirmed=True)
        self.assertEqual([x["object_id"] for x in matches["findings"]], ["X","Y"])
        saved["conditional"]["arbitrary_permission"] = True
        with self.assertRaises(ValueError):
            compile_requirements(model, "检查设备")
        saved["conditional"] = None
        saved["program"] = {}
        with self.assertRaises(ValueError):
            compile_requirements(model, "检查设备")

    def test_atomic_visual_question_keeps_polarity_and_scope(self):
        for lead in ("图像中", "照片中的", "画面中", ""):
            for outcome, value in (("positive", True), ("negative", False)):
                self.assertEqual(atomic_facts(lead+"该设备的正面外壳是否存在裂纹？", outcome,
                                              {"cracked":"flag"}), {"cracked": value})
        for q in ("图像中该设备的正面外壳是否不存在裂纹？", "该设备是否有裂纹或发热？", "邻近设备是否有裂纹？"):
            self.assertEqual(atomic_facts(q, "positive", {"cracked":"flag"}), {})
        self.assertEqual(atomic_facts("该设备是否有裂纹？", "uncertain", {"cracked":"flag"}), {})

    def test_task_authority_and_timing(self):
        m = SimpleNamespace(field="identity", needed_when=("positive",))
        report = "只报告异常设备的编号。"
        for grant in ("允许对指定设备进行非接触式巡检和拍摄。", "授权机器人对本次任务范围内的全部设备开展非接触巡检并拍照。"):
            self.assertTrue(permission_for(grant + report, m).early)
            self.assertFalse(permission_for(grant + report + "只有确认异常之后才允许拍摄编号。", m).early)
            self.assertEqual(permission_for(grant + report + "禁止拍摄编号。", m).after, ())
            self.assertFalse(permission_for(grant + report, SimpleNamespace(field="temperature", needed_when=("positive",))).early)
        for s in (report, "仅允许对A区指定设备进行非接触式巡检和拍摄。" + report,
                  "如果管理员同意，方可拍摄编号。" + report):
            self.assertFalse(permission_for(s,m).early)

    def test_branch_specific_continuation(self):
        outcomes = ("positive","negative")
        pos, neg = (Condition("q",(x,)) for x in outcomes)
        task = TaskContract("Each branch requires its own evidence.", "entry", "exit", 50., 1.,
            (Requirement("p","X","mark",pos),Requirement("n","X","label",neg)), (QuerySpec("q","X", outcomes,10,12),),
            (Skill("read-mark","mid",("p",),1.), Skill("confirmed-label","mid",("n",),1.,neg)),
            (("entry","mid",3.),("mid","exit",2.),("entry","exit",5.)))
        state = Snapshot(0.,"entry",{},{},{})
        quiet = lambda *args, **kwargs: None
        self.assertIsNone(select(task,state,"conditional-a",quiet))
        for method in ("adaptive-progress","conditional-planner"):
            choice = select(task,state,method,quiet)
            self.assertEqual([a.skill.id for a in choice.prepare],["read-mark"])
        self.assertEqual(state.evidence, {})
