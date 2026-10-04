import unittest

from scripts.train_qian_hf_job import build_prompt, check_splits, encode_row


class FakeTokenizer:
    eos_token_id = 7

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) for char in text]


class QianJobTests(unittest.TestCase):
    def setUp(self):
        self.row = {"instruction": "改写。", "input": "大家很忙。", "output": "忙碌成了门面。"}
        self.tokenizer = FakeTokenizer()

    def test_prompt_matches_local_and_space_headings(self):
        self.assertEqual(build_prompt("改写。", "大家很忙。"),
                         "### 指令:\n改写。\n\n### 输入:\n大家很忙。\n\n### 输出:\n")

    def test_prompt_is_masked_but_original_and_eos_are_trained(self):
        encoded = encode_row(self.row, self.tokenizer)
        prompt = self.tokenizer.encode(build_prompt(self.row["instruction"], self.row["input"]))
        output = self.tokenizer.encode(self.row["output"]) + [7]
        self.assertEqual(encoded["input_ids"], prompt + output)
        self.assertEqual(encoded["labels"], [-100] * len(prompt) + output)
        self.assertEqual(encoded["attention_mask"], [1] * len(encoded["input_ids"]))

    def test_original_cannot_be_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            encode_row(self.row, self.tokenizer, max_length=4)

    def test_eos_is_required(self):
        self.tokenizer.eos_token_id = None
        with self.assertRaisesRegex(ValueError, "end-of-sequence"):
            encode_row(self.row, self.tokenizer)

    def test_schema_fields_are_required(self):
        for field in ("instruction", "input", "output"):
            with self.subTest(field=field):
                row = self.row.copy()
                row[field] = ""
                with self.assertRaises(ValueError):
                    encode_row(row, self.tokenizer)

    def test_split_counts_are_checked(self):
        with self.assertRaisesRegex(ValueError, "285 training"):
            check_splits({"train": [self.row], "validation": [self.row]})

    def test_splits_must_not_share_inputs_or_outputs(self):
        rows = [dict(instruction="改写。", input=f"输入{i}", output=f"原文{i}") for i in range(310)]
        data = {"train": rows[:285], "validation": rows[285:]}
        check_splits(data)
        data["validation"][0] = data["train"][0].copy()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            check_splits(data)


if __name__ == "__main__":
    unittest.main()
