import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_weicheng_dataset import (
    build_rewrite_examples,
    is_near_identical,
    main,
    neutralize_input,
    normalize_for_similarity,
    read_extra_pairs,
)


class WeichengRewriteTests(unittest.TestCase):
    def test_rewrite_examples_use_rewrite_schema(self):
        blocks = [
            "他的笑容非常勉强，像贴歪了的邮票，大家都看得出来，他自己却以为遮掩得很好。",
            "会议上的人争着发言，仿佛嘴巴可以代替脑子工作，没有人听别人说什么，问题也一直没有解决。",
        ]
        rows = build_rewrite_examples(blocks, min_chars=8, max_chars=80)
        self.assertGreaterEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(set(row), {"instruction", "input", "output"})
            self.assertIn("改写", row["instruction"])
            self.assertTrue(row["input"])
            self.assertTrue(row["output"])
            self.assertFalse(is_near_identical(row["input"], row["output"]))

    def test_identical_examples_are_removed(self):
        blocks = ["现代人离不开手机。一个人为了显得有学问，总爱引用自己不懂的书。"]
        self.assertEqual(build_rewrite_examples(blocks, min_chars=8, max_chars=80), [])

    def test_cosmetic_changes_are_ignored(self):
        pairs = [
            ("相同的句子。", "相同的句子。"),
            ("现代人 离不开手机。", "现代人\n离不开手机！"),
            ('他后来问: "你为什么不走?"', "他後来问道：「你为什麽不走？」"),
            ("鸿渐说：你好。", "鸿渐道：你好。"),
            ("他说：回家吧。", "他说道：回家罢。"),
            ("ABC123", "ＡＢＣ１２３"),
        ]
        for source, target in pairs:
            with self.subTest(source=source, target=target):
                self.assertEqual(normalize_for_similarity(source), normalize_for_similarity(target))
                self.assertTrue(is_near_identical(source, target))

    def test_similarity_threshold_is_inclusive_and_configurable(self):
        source = "甲乙丙丁戊己庚辛壬癸"
        target = "甲乙丙丁戊己庚辛壬子"
        self.assertTrue(is_near_identical(source, target, 0.90))
        self.assertFalse(is_near_identical(source, target, 0.91))
        self.assertEqual(is_near_identical(source, target), is_near_identical(target, source))

    def test_substantial_rewrites_are_kept(self):
        self.assertFalse(is_near_identical(
            "现代人离不开手机。",
            "现代人把手机带在身边，像带着一位小皇帝；时时低头朝见，还自以为是在治理天下。",
        ))

    def test_empty_pairs_are_removed(self):
        for source, target in [("", "一句话"), ("一句话", ""), ("？！", "。"), (" ", "\n")]:
            with self.subTest(source=source, target=target):
                self.assertTrue(is_near_identical(source, target))

    def test_invalid_thresholds_are_rejected(self):
        for value in [0, -0.1, 1.1, float("nan")]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    is_near_identical("输入", "输出", value)

    def test_extra_pairs_are_also_filtered(self):
        good = {"input": "大家等答复。", "output": "大家等一个明确答复，仿佛等一封迟到的信。"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [
                {"input": "原句。", "output": "原句。"},
                {"input": "他后来走了。", "output": "他後来走了！"},
                good,
            ]), encoding="utf-8")
            rows = read_extra_pairs(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input"], good["input"])
        self.assertEqual(rows[0]["output"], good["output"])

    def test_cli_filters_before_splitting_and_filters_extra_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text(
                "他的笑容显得非常勉强，像贴歪了的邮票，在场的大家都看得出来，他自己却以为遮掩得很好。\n\n"
                "会议上的人争着发言，仿佛嘴巴可以代替脑子工作，没有人听别人说什么，问题也一直没有解决。",
                encoding="utf-8",
            )
            extra = root / "extra.jsonl"
            extra.write_text(json.dumps({"input": "原句。", "output": "原句。"}), encoding="utf-8")
            train, evaluation = root / "train.jsonl", root / "eval.jsonl"
            with patch("sys.argv", [
                "prepare_weicheng_dataset.py", "--source", str(source),
                "--train-out", str(train), "--eval-out", str(evaluation),
                "--extra-pairs", str(extra),
            ]), patch("builtins.print"):
                self.assertEqual(main(), 0)
            for path in [train, evaluation]:
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                self.assertTrue(rows)
                self.assertTrue(all(not is_near_identical(row["input"], row["output"]) for row in rows))

    def test_neutralize_keeps_meaning_when_clause_removal_is_too_large(self):
        text = "他像一只被雨打湿的鸟。"
        self.assertEqual(neutralize_input(text), text)


if __name__ == "__main__":
    unittest.main()
