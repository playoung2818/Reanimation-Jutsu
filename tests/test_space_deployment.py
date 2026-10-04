import ast
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from scripts.train_qian_hf_job import build_prompt


ROOT = Path(__file__).resolve().parents[1]


def app_namespace():
    """Test application functions without GPUs, downloads, or launching Gradio."""
    tree = ast.parse((ROOT / "huggingface_space/app.py").read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))]
    namespace = {
        "os": SimpleNamespace(getenv=lambda key: "test-token"),
        "gr": SimpleNamespace(Error=RuntimeError),
        "spaces": SimpleNamespace(GPU=lambda **kwargs: lambda fn: fn),
        "torch": MagicMock(), "snapshot_download": MagicMock(),
        "AutoTokenizer": MagicMock(), "AutoModelForCausalLM": MagicMock(),
        "BitsAndBytesConfig": MagicMock(), "PeftModel": MagicMock(),
    }
    namespace["snapshot_download"].side_effect = ["/cached/base", "/cached/qian", "/cached/lincoln"]
    namespace["torch"].cuda.is_bf16_supported.return_value = False
    exec(compile(tree, "huggingface_space/app.py", "exec"), namespace)
    return namespace


class SpaceDeploymentTests(unittest.TestCase):
    def test_rewrite_tab_is_first_and_lincoln_tab_is_hidden(self):
        tree = ast.parse((ROOT / "huggingface_space/app.py").read_text(encoding="utf-8"))
        tabs = [
            node.items[0].context_expr for node in ast.walk(tree)
            if isinstance(node, ast.With)
            and isinstance(node.items[0].context_expr, ast.Call)
            and isinstance(node.items[0].context_expr.func, ast.Attribute)
            and node.items[0].context_expr.func.attr == "Tab"
        ]
        self.assertEqual([tab.args[0].value for tab in tabs], ["钱钟书 Qian Zhongshu", "Abraham Lincoln"])
        self.assertFalse(any(kw.arg == "visible" and kw.value.value is False for kw in tabs[0].keywords))
        self.assertTrue(any(kw.arg == "visible" and kw.value.value is False for kw in tabs[1].keywords))

    def test_rewrite_prompt_and_revision_match_completed_training(self):
        ns = app_namespace()
        run = json.loads((ROOT / "huggingface_space/deployment.json").read_text())
        rows = json.loads((ROOT / "example_data/qianzhongshu_rewrite_curated.json").read_text(encoding="utf-8"))
        self.assertEqual(ns["QIAN_ADAPTER"], run["qian_adapter"])
        self.assertEqual(ns["QIAN_REVISION"], run["qian_revision"])
        self.assertEqual(ns["BASE_REVISION"], run["base_revision"])
        self.assertEqual({row["instruction"] for row in rows}, {ns["QIAN_INSTRUCTION"]})
        ns["load_model"] = MagicMock()
        ns["generate"] = MagicMock(return_value="改写结果")
        self.assertEqual(ns["respond_qian"](" 原始输入。 ", []), "改写结果")
        ns["generate"].assert_called_once_with(build_prompt(ns["QIAN_INSTRUCTION"], "原始输入。"), "qian")

    def test_private_adapters_are_downloaded_by_revision_and_cached(self):
        ns = app_namespace()
        ns["prepare_assets"]()
        calls = ns["snapshot_download"].call_args_list
        self.assertEqual(len(calls), 3)
        for call, repo, revision in zip(calls,
                [ns["BASE_MODEL"], ns["QIAN_ADAPTER"], ns["LINCOLN_ADAPTER"]],
                [ns["BASE_REVISION"], ns["QIAN_REVISION"], ns["LINCOLN_REVISION"]]):
            self.assertEqual(call.args, (repo,))
            self.assertEqual(call.kwargs["revision"], revision)
            self.assertEqual(call.kwargs["token"], "test-token")
        self.assertEqual(calls[1].kwargs["allow_patterns"], ["adapter_config.json", "adapter_model.safetensors"])
        ns["prepare_assets"]()
        self.assertEqual(ns["snapshot_download"].call_count, 3)
        self.assertEqual(ns["qian_path"], "/cached/qian")

    def test_missing_secret_fails_before_download(self):
        ns = app_namespace()
        ns["HF_TOKEN"] = None
        with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
            ns["prepare_assets"]()
        ns["snapshot_download"].assert_not_called()

    def test_both_adapters_load_from_local_snapshots(self):
        ns = app_namespace()
        ns["load_model"]()
        self.assertEqual(ns["AutoTokenizer"].from_pretrained.call_args.args, ("/cached/base",))
        self.assertEqual(ns["AutoModelForCausalLM"].from_pretrained.call_args.args, ("/cached/base",))
        self.assertTrue(ns["AutoModelForCausalLM"].from_pretrained.call_args.kwargs["local_files_only"])
        self.assertEqual(ns["PeftModel"].from_pretrained.call_args.args[1], "/cached/qian")
        ns["model"].load_adapter.assert_called_once_with("/cached/lincoln", adapter_name="lincoln", is_trainable=False)
        ns["load_model"]()
        self.assertEqual(ns["PeftModel"].from_pretrained.call_count, 1)

    def test_failed_adapter_load_cannot_leave_a_partial_global_model(self):
        ns = app_namespace()
        ns["PeftModel"].from_pretrained.return_value.load_adapter.side_effect = RuntimeError("test failure")
        with self.assertRaises(RuntimeError):
            ns["load_model"]()
        self.assertIsNone(ns["model"])
        self.assertIsNone(ns["tokenizer"])

    def test_long_input_is_rejected_before_generation(self):
        ns = app_namespace()
        ns["model"] = MagicMock()
        ns["tokenizer"] = MagicMock(return_value={"input_ids": SimpleNamespace(shape=(1, 1025))})
        with self.assertRaisesRegex(RuntimeError, "shorter"):
            ns["generate"]("too long", "qian")
        ns["model"].generate.assert_not_called()

    def test_lincoln_history_still_uses_chat_template(self):
        ns = app_namespace()
        ns["load_model"] = MagicMock()
        ns["tokenizer"] = MagicMock()
        ns["tokenizer"].apply_chat_template.return_value = "chat prompt"
        ns["generate"] = MagicMock(return_value="answer")
        history = [{"role": "user", "content": "Earlier question"}, {"role": "assistant", "content": "Earlier reply"}]
        before = copy.deepcopy(history)
        self.assertEqual(ns["respond_lincoln"]("New question", history), "answer")
        self.assertEqual(history, before)
        messages = ns["tokenizer"].apply_chat_template.call_args.args[0]
        self.assertEqual(messages[1:3], history)
        self.assertEqual(messages[-1], {"role": "user", "content": "New question"})
        ns["generate"].assert_called_once_with("chat prompt", "lincoln")


if __name__ == "__main__":
    unittest.main()
