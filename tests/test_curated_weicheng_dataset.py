import copy
import json
import unittest
from pathlib import Path

from scripts.prepare_curated_weicheng_dataset import novel_body, split_rows, validate_rows


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    "第一章\n\n第二章\n\n序\n前言中的一句话。\n\n"
    "第一章\n\n夜色越来越深，远处的一点灯光却迟迟没有消失。\n\n"
    "第二章\n\n人坐满了屋子，能解决问题的办法却一个也没有。\n\n"
    "附录\n附录中的一句话。"
)
ROWS = [
    {
        "instruction": "保持意思，改写成文学中文。",
        "input": "到了深夜，远处仍然亮着一盏灯。",
        "output": "夜色越来越深，远处的一点灯光却迟迟没有消失。",
    },
    {
        "instruction": "保持意思，改写成文学中文。",
        "input": "屋里有很多人，但大家都没有解决问题的办法。",
        "output": "人坐满了屋子，能解决问题的办法却一个也没有。",
    },
]


class CuratedWeichengDatasetTests(unittest.TestCase):
    def test_body_excludes_contents_front_matter_and_appendix(self):
        body = novel_body(SOURCE)
        self.assertIn(ROWS[0]["output"], body)
        self.assertIn(ROWS[1]["output"], body)
        self.assertNotIn("前言中的一句话", body)
        self.assertNotIn("附录中的一句话", body)

    def test_valid_rows(self):
        validate_rows(ROWS, SOURCE, expected_count=2)

    def test_edited_original_is_rejected(self):
        rows = copy.deepcopy(ROWS)
        rows[0]["output"] = "夜色渐深，远处的灯光没有消失。"
        with self.assertRaisesRegex(ValueError, "exact novel excerpt"):
            validate_rows(rows, SOURCE, expected_count=2)

    def test_front_matter_is_not_a_novel_excerpt(self):
        rows = copy.deepcopy(ROWS)
        rows[0]["output"] = "前言中的一句话。"
        with self.assertRaisesRegex(ValueError, "exact novel excerpt"):
            validate_rows(rows, SOURCE, expected_count=2)

    def test_names_in_any_field_are_rejected(self):
        for field in ("instruction", "input", "output"):
            with self.subTest(field=field):
                rows = copy.deepcopy(ROWS)
                rows[0][field] += "鸿渐"
                source = SOURCE.replace(ROWS[0]["output"], rows[0]["output"])
                with self.assertRaisesRegex(ValueError, "forbidden name"):
                    validate_rows(rows, source, expected_count=2)

    def test_copy_like_input_is_rejected(self):
        rows = copy.deepcopy(ROWS)
        rows[0]["input"] = rows[0]["output"].replace("。", "！")
        with self.assertRaisesRegex(ValueError, "substantial paraphrase"):
            validate_rows(rows, SOURCE, expected_count=2)

    def test_duplicates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows([ROWS[0], ROWS[0]], SOURCE, expected_count=2)

    def test_split_is_reproducible_disjoint_and_does_not_mutate_input(self):
        before = copy.deepcopy(ROWS)
        train, evaluation = split_rows(ROWS)
        self.assertEqual((train, evaluation), split_rows(ROWS))
        self.assertEqual(ROWS, before)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(evaluation), 1)
        self.assertNotEqual(train[0]["output"], evaluation[0]["output"])

    def test_invalid_split_settings_are_rejected(self):
        for ratio in (0, 1, -0.1, 1.1):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                split_rows(ROWS, eval_ratio=ratio)
        with self.assertRaises(ValueError):
            split_rows(ROWS[:1])

    def test_actual_dataset_and_provenance(self):
        rows = json.loads((ROOT / "example_data/qianzhongshu_rewrite_curated.json").read_text(encoding="utf-8"))
        source = (ROOT / "围城.txt").read_text(encoding="utf-8")
        validate_rows(rows, source)
        manifest = json.loads((ROOT / "source_data/weicheng_curated_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 310)
        self.assertEqual(manifest["additional_pairs"], 300)
        self.assertEqual(len(manifest["rows"]), len(rows))
        for row, entry in zip(rows, manifest["rows"]):
            start = entry["source_offset"]
            self.assertEqual(row["output"], source[start:start + entry["source_length"]])
        train, evaluation = split_rows(rows)
        self.assertEqual((len(train), len(evaluation)), (285, 25))


if __name__ == "__main__":
    unittest.main()
