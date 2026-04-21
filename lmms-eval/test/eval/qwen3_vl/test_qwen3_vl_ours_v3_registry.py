import ast
from pathlib import Path
from unittest import TestCase


class TestQwen3VLOursV3Registry(TestCase):
    def test_qwen3_vl_ours_v3_registry_and_module_exist(self):
        repo_root = Path(__file__).resolve().parents[3]
        registry_path = repo_root / "lmms_eval" / "models" / "__init__.py"
        module_path = repo_root / "lmms_eval" / "models" / "chat" / "qwen3_vl_ours_v3.py"

        registry_text = registry_path.read_text(encoding="utf-8")
        self.assertIn('"qwen3_vl_ours_v3": "Qwen3_VL_Ours_V3"', registry_text)
        self.assertTrue(module_path.exists(), "qwen3_vl_ours_v3.py should exist")

        module_ast = ast.parse(module_path.read_text(encoding="utf-8"))
        class_names = {
            node.name
            for node in module_ast.body
            if isinstance(node, ast.ClassDef)
        }
        self.assertIn("Qwen3_VL_Ours_V3", class_names)

        register_names = []
        for node in module_ast.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                if getattr(decorator.func, "id", None) != "register_model":
                    continue
                if decorator.args and isinstance(decorator.args[0], ast.Constant):
                    register_names.append(decorator.args[0].value)

        self.assertIn("qwen3_vl_ours_v3", register_names)
