import unittest

from scripts.prepare_lincoln_dataset import split_rows


class DocumentSplitTests(unittest.TestCase):
    def test_documents_remain_in_one_split(self):
        rows = [
            {"title": title, "task": task}
            for title in ("Speech A", "Speech B", "Letter C", "Letter D")
            for task in ("topic_response", "continuation", "continuation")
        ]
        train, evaluation = split_rows(rows, 0.25, 1809)
        self.assertFalse({r["title"] for r in train} & {r["title"] for r in evaluation})
        self.assertEqual(len(evaluation), 3)
        self.assertEqual(len(train) + len(evaluation), len(rows))
        self.assertCountEqual(train + evaluation, rows)
        self.assertEqual((train, evaluation), split_rows(rows, 0.25, 1809))

    def test_small_dataset_keeps_both_splits(self):
        rows = [{"title": "A"}, {"title": "B"}]
        for ratio in (0.01, 0.99):
            train, evaluation = split_rows(rows, ratio, 1809)
            self.assertEqual((len(train), len(evaluation)), (1, 1))

    def test_invalid_inputs(self):
        for rows, ratio in (([], 0.1), ([{"title": "A"}], 0.1),
                            ([{"title": "A"}, {"title": "B"}], 0),
                            ([{"title": "A"}, {"title": "B"}], 1)):
            with self.assertRaises(ValueError):
                split_rows(rows, ratio, 1809)


if __name__ == "__main__":
    unittest.main()
