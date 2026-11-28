import unittest

from experiments.flexible_execution.text_requirements import compile_text


class TextRequirementsTest(unittest.TestCase):
    def test_schema_is_not_an_instruction(self):
        task = ("找到编号为BAY-23的仓储货位，报告该货位的实际箱数和货箱颜色，保留图像证据。 "
                "报告中以 object_id 表示对象编号，facts 使用字段：box_count（箱数）、color（英文颜色名）。")
        program, intent = compile_text(task)
        self.assertEqual(program.scope["identity"],"BAY-23")
        self.assertEqual(set(program.report),{"color","box_count"})
        self.assertIsNone(intent)
        for altered in (task.replace("实际箱数", "最多可能的箱数"), task.replace("该货位的","邻近货位的"),
                        task.replace("color（英文颜色名）","color（温度）"), task+"待获得许可后才能拍摄。"):
            self.assertIsNone(compile_text(altered))

    def test_predicate_and_permission_are_separate(self):
        task=("检查这一排设备的正面外壳，报告存在裂纹的设备编号，并保留图像证据。编号指设备本体身份。 "
              "报告中以 object_id 表示对象编号，facts 使用字段：cracked（是否有裂纹）。")
        program, intent = compile_text(task)
        self.assertEqual(intent.materials[0].needed_when,("positive",))
        self.assertIsNone(intent.materials[0].permission_quote)
        self.assertEqual(program.filter,{"op":"eq","field":"cracked","right":{"literal":True}})
        self.assertIsNone(compile_text(task.replace("存在裂纹","有裂纹且温度超过70度")))
        self.assertIsNone(compile_text(task.replace("设备编号","通道编号")))
        self.assertIsNone(compile_text(task.replace("检查这一排","只检查左侧这一排")))
