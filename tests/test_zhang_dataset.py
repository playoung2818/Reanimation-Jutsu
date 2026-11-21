import copy
import io
import json
import sys
import tempfile
import types
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts import launch_zhang_hf_job

from scripts.extract_zhang_epub import CHAPTERS, body_paragraphs, extract_collection
from scripts.launch_zhang_hf_job import build_training_data
from scripts.prepare_zhang_dataset import (
    INSTRUCTION, WORKS, excerpt_candidates, load_sources, select_candidates, sha256, split_rows, validate_rows,
)


def fixture_rows(counts=(104, 103, 103)):
    sources, rows = {}, []
    for work, count in zip(WORKS, counts):
        outputs = [f"{work}的测试句子{i}：夜色越来越深，远处的一点灯光却迟迟没有消失。" for i in range(count)]
        sources[work] = "\n\n".join(outputs)
        start = 0
        for i, output in enumerate(outputs):
            rows.append({
                "work": work, "source_offset": start, "source_length": len(output),
                "source_sha256": sha256(sources[work]), "instruction": INSTRUCTION,
                "input": f"测试{i}，作品{work}：已经很晚了，但是远方还亮着一盏灯。", "output": output,
                "reviewed": True,
            })
            start += len(output) + 2
    return sources, rows


class ZhangDatasetTests(unittest.TestCase):
    def setUp(self):
        self.sources, self.rows = fixture_rows()

    def test_complete_reviewed_rows_pass_and_export_matches_qian_trainer(self):
        files = build_training_data(self.rows, self.sources)
        self.assertEqual(set(files), {"train.jsonl", "validation.jsonl"})
        for filename, count in (("train.jsonl", 285), ("validation.jsonl", 25)):
            rows = [json.loads(line) for line in files[filename].decode().splitlines()]
            self.assertEqual(len(rows), count)
            self.assertTrue(all(set(row) == {"instruction", "input", "output"} for row in rows))

    def test_count_is_fixed(self):
        with self.assertRaisesRegex(ValueError, "310"):
            validate_rows(self.rows[:-1], self.sources)

    def test_edited_original_is_rejected(self):
        self.rows[0]["output"] = self.rows[0]["output"].replace("夜色", "月色")
        with self.assertRaisesRegex(ValueError, "exact source excerpt"):
            validate_rows(self.rows, self.sources)

    def test_source_change_is_rejected(self):
        self.sources[WORKS[0]] += "新段落"
        with self.assertRaisesRegex(ValueError, "source hash changed"):
            validate_rows(self.rows, self.sources)

    def test_offset_and_length_are_checked(self):
        for field, value in (("source_offset", -1), ("source_offset", True), ("source_offset", 1), ("source_length", 0)):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_rows(rows, self.sources)

    def test_unreviewed_pairs_cannot_be_exported(self):
        self.rows[0]["reviewed"] = False
        with self.assertRaisesRegex(ValueError, "human review"):
            build_training_data(self.rows, self.sources)
        validate_rows(self.rows, self.sources, require_review=False)

    def test_explicit_draft_export_preserves_flags_and_other_checks(self):
        self.rows[0]["reviewed"] = False
        before = copy.deepcopy(self.rows)
        files = build_training_data(self.rows, self.sources, require_review=False)
        self.assertEqual(set(files), {"train.jsonl", "validation.jsonl"})
        self.assertEqual(self.rows, before)
        self.rows[0]["input"] = self.rows[0]["output"]
        with self.assertRaisesRegex(ValueError, "substantial"):
            build_training_data(self.rows, self.sources, require_review=False)

    def test_review_flag_must_be_boolean(self):
        self.rows[0]["reviewed"] = "true"
        with self.assertRaisesRegex(ValueError, "human review"):
            validate_rows(self.rows, self.sources, require_review=False)

    def test_blank_or_copy_inputs_are_rejected(self):
        for inp, message in (("", "nonempty"), (self.rows[0]["output"], "substantial")):
            self.rows[0]["input"] = inp
            with self.subTest(inp=inp), self.assertRaisesRegex(ValueError, message):
                validate_rows(self.rows, self.sources)

    def test_qian_instruction_cannot_enter_zhang_data(self):
        self.rows[0]["instruction"] = "请改写成钱钟书式中文。"
        with self.assertRaisesRegex(ValueError, "Zhang Ailing rewrite instruction"):
            validate_rows(self.rows, self.sources)

    def test_duplicate_input_is_rejected(self):
        self.rows[-1]["input"] = self.rows[0]["input"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows(self.rows, self.sources)

    def test_name_guard_covers_inputs(self):
        self.rows[0]["input"] += "九莉"
        with self.assertRaisesRegex(ValueError, "source name or place"):
            validate_rows(self.rows, self.sources)

    def test_split_is_deterministic_balanced_and_disjoint(self):
        before = copy.deepcopy(self.rows)
        train, evaluation = split_rows(self.rows)
        self.assertEqual((len(train), len(evaluation)), (285, 25))
        self.assertEqual((train, evaluation), split_rows(self.rows))
        self.assertEqual(self.rows, before)
        self.assertEqual({row["work"] for row in train}, set(WORKS))
        self.assertEqual({row["work"] for row in evaluation}, set(WORKS))
        self.assertFalse({row["output"] for row in train} & {row["output"] for row in evaluation})

    def test_invalid_ratio_and_small_work_are_rejected(self):
        for ratio in (0, 1, 0.001, 0.999):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                split_rows(self.rows, ratio)
        with self.assertRaises(ValueError):
            split_rows(self.rows[:104])

    def test_candidate_offsets_and_review_status(self):
        text = "标题\n\n深夜街头没有行人，只有路灯依旧照着湿漉漉的路面。\n\n远处的流苏不说话，仿佛整个人被月光淡淡地隔开了。"
        sources = {work: text + f"\n{work}" for work in WORKS}
        # Shared sentences cannot be selected twice across different works.
        with self.assertRaisesRegex(ValueError, "eligible sentences"):
            select_candidates(sources, counts=(1, 1, 1))
        candidates = excerpt_candidates(text)
        self.assertEqual(len(candidates), 1)
        offset, output = candidates[0]
        self.assertEqual(text[offset:offset + len(output)], output)
        sources = {work: text.replace("深夜", f"{work}里的深夜") for work in WORKS}
        rows = select_candidates(sources, counts=(1, 1, 1))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["input"] == "" and row["reviewed"] is False for row in rows))

    def test_local_preview_exactness_when_available(self):
        root = Path(__file__).resolve().parents[1]
        preview = root / "example_data/zhangailing_rewrite_preview.json"
        if not preview.exists():
            self.skipTest("Local copyrighted source data is not stored in Git")
        validate_rows(json.loads(preview.read_text(encoding="utf-8")), load_sources(root / "source_data/zhang_ailing"),
                      expected_count=10, require_review=False)


class ZhangLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources_dir = self.root / "sources"
        self.sources_dir.mkdir()
        sources, rows = fixture_rows()
        for work, text in sources.items():
            (self.sources_dir / f"{work}.txt").write_text(text, encoding="utf-8")
        self.dataset = self.root / "curated.json"
        self.dataset.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        (self.root / "scripts").mkdir()
        (self.root / "scripts/train_qian_hf_job.py").write_text("# test trainer\n", encoding="utf-8")
        self.argv = ["launch", "--dataset", str(self.dataset), "--source-dir", str(self.sources_dir)]
        self.api = MagicMock()
        self.api.dataset_info.return_value.private = True
        self.api.model_info.return_value = types.SimpleNamespace(private=True, sha="base-sha")
        self.api.create_commit.return_value.oid = "dataset-sha"
        self.api.run_uv_job.return_value.id = "test-job"
        self.hub = types.ModuleType("huggingface_hub")
        self.hub.HfApi = MagicMock(return_value=self.api)
        self.hub.CommitOperationAdd = lambda **kwargs: kwargs

    def run_launcher(self, submit=False, allow_unreviewed=False):
        flags = (["--submit"] if submit else []) + (["--allow-unreviewed"] if allow_unreviewed else [])
        with patch.object(sys, "argv", self.argv + flags), \
             patch.object(launch_zhang_hf_job, "ROOT", self.root), \
             patch.dict(sys.modules, {"huggingface_hub": self.hub}), \
             redirect_stdout(io.StringIO()):
            return launch_zhang_hf_job.main()

    def test_dry_run_never_initializes_hub_or_starts_job(self):
        self.assertEqual(self.run_launcher(), 0)
        self.hub.HfApi.assert_not_called()
        self.api.run_uv_job.assert_not_called()

    def test_submission_uses_zhang_ids_pins_secrets_and_l4(self):
        with patch.dict("os.environ", {"HF_TOKEN": "synthetic-test-token"}):
            self.assertEqual(self.run_launcher(submit=True), 0)
        args = self.api.run_uv_job.call_args.kwargs
        self.assertEqual(args["flavor"], "l4x1")
        self.assertEqual(args["timeout"], "2h")
        self.assertEqual(args["secrets"], {"HF_TOKEN": "synthetic-test-token"})
        self.assertIn(launch_zhang_hf_job.DATASET_ID, args["script_args"])
        self.assertIn(launch_zhang_hf_job.MODEL_ID, args["script_args"])
        self.assertIn("dataset-sha", args["script_args"])
        self.assertIn("base-sha", args["script_args"])
        record = (self.root / "saved_models/zhangailing_curated_hf_job.json").read_text()
        self.assertNotIn("synthetic-test-token", record)
        self.assertEqual(json.loads(record)["job_id"], "test-job")
        operations = self.api.create_commit.call_args.kwargs["operations"]
        self.assertEqual({op["path_in_repo"] for op in operations}, {"train.jsonl", "validation.jsonl"})

    def test_drafts_are_blocked_by_default_before_any_hub_request(self):
        rows = json.loads(self.dataset.read_text())
        for row in rows:
            row["reviewed"] = False
        self.dataset.write_text(json.dumps(rows), encoding="utf-8")
        with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
            self.run_launcher(submit=True)
        self.hub.HfApi.assert_not_called()

    def test_explicit_draft_job_records_unreviewed_status_without_changing_data(self):
        rows = json.loads(self.dataset.read_text())
        for row in rows:
            row["reviewed"] = False
        self.dataset.write_text(json.dumps(rows), encoding="utf-8")
        before = self.dataset.read_bytes()
        with patch.dict("os.environ", {"HF_TOKEN": "synthetic-test-token"}):
            self.assertEqual(self.run_launcher(submit=True, allow_unreviewed=True), 0)
        args = self.api.run_uv_job.call_args.kwargs["script_args"]
        self.assertEqual(args[args.index("--data-review-status") + 1], "unreviewed")
        record = json.loads((self.root / "saved_models/zhangailing_curated_hf_job.json").read_text())
        self.assertEqual(record["reviewed_rows"], 0)
        self.assertEqual(record["data_review_status"], "unreviewed")
        self.assertTrue(record["allow_unreviewed"])
        self.assertEqual(self.dataset.read_bytes(), before)
        self.assertIn("unreviewed", self.api.create_commit.call_args.kwargs["commit_message"])

    def test_public_dataset_blocks_upload_and_submission(self):
        self.api.dataset_info.return_value.private = False
        with patch.dict("os.environ", {"HF_TOKEN": "synthetic-test-token"}), \
             self.assertRaisesRegex(RuntimeError, "public dataset"):
            self.run_launcher(submit=True)
        self.api.create_commit.assert_not_called()
        self.api.run_uv_job.assert_not_called()


class ZhangExtractionTests(unittest.TestCase):
    def test_nested_paragraphs_keep_inline_text_and_exclude_front_matter(self):
        data = b'<html><body><h1>Title</h1><div><p class="bodytext">First <span>sentence</span>.</p><p class="bodytext1">Second.</p></div><p class="copyright">Not prose</p><script>Not prose</script></body></html>'
        self.assertEqual(body_paragraphs(data), ["First sentence.", "Second."])

    def test_full_collection_extracts_only_target_stories_and_twelve_chapters(self):
        def point(label, href, children=""):
            return f'<navPoint><navLabel><text>{label}</text></navLabel><content src="{href}"/>{children}</navPoint>'
        chapters = "".join(point(label, f"chapter{i}.html") for i, label in enumerate(CHAPTERS))
        toc = '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>' + point("倾城之恋", "cover.html", point("封面", "cover.html") + point("倾城之恋", "love.html") + point("金锁记", "gold.html")) + point("小团圆", "smallcover.html", point("版权信息", "smallcover.html") + chapters) + point("其他作品", "other.html") + '</navMap></ncx>'
        names = ["cover", "love", "gold", "smallcover"] + [f"chapter{i}" for i in range(12)] + ["other"]
        manifest = '<item id="toc" href="toc.ncx" media-type="application/x-dtbncx+xml"/>' + "".join(f'<item id="{name}" href="{name}.html"/>' for name in names)
        spine = "".join(f'<itemref idref="{name}"/>' for name in names)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collection.epub"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>')
                archive.writestr("OEBPS/content.opf", f'<package><manifest>{manifest}</manifest><spine toc="toc">{spine}</spine></package>')
                archive.writestr("OEBPS/toc.ncx", toc)
                for name in names:
                    archive.writestr(f"OEBPS/{name}.html", f'<html><body><h1>Heading</h1><p class="bodytext">{name} prose.</p></body></html>')
            texts, record = extract_collection(path)
        self.assertEqual(texts["倾城之恋"], "love prose.\n")
        self.assertEqual(texts["金锁记"], "gold prose.\n")
        self.assertEqual(record["works"]["小团圆"]["paragraphs"], 12)
        self.assertNotIn("cover", texts["小团圆"])
        self.assertNotIn("other", texts["小团圆"])
        self.assertEqual(record["works"]["金锁记"]["sha256"], sha256(texts["金锁记"]))


if __name__ == "__main__":
    unittest.main()
