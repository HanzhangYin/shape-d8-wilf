import json
import tempfile
import unittest
from pathlib import Path

import generate_reverse_json as grj

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def class_sets(payload):
    return {
        frozenset(record["patterns"])
        for record in payload["record_classes"]
    }


class ReverseJsonGeneratorTests(unittest.TestCase):
    def test_format_pattern_is_unambiguous_for_k10(self):
        pattern = (9, 8, 7, 6, 5, 4, 3, 2, 1, 10)
        text = grj.format_pattern(pattern)
        self.assertEqual(text, "9 8 7 6 5 4 3 2 1 10")
        self.assertEqual(grj.parse_pattern(text), pattern)

    def test_k5_generation_matches_existing_reverse_file_semantically(self):
        existing = json.loads((DATA / "reverse_k5.json").read_text())
        generated = grj.build_reverse_data(
            5,
            singletons="omit-d8-seed",
        )

        self.assertEqual(generated["k"], existing["k"])
        self.assertNotIn("n_list", generated)
        self.assertNotIn("n_list", existing)
        self.assertEqual(class_sets(generated), class_sets(existing))
        self.assertEqual(len(generated["record_classes"]), 108)
        self.assertEqual(sum(r["size"] for r in generated["record_classes"]), 119)
        all_patterns = {p for r in generated["record_classes"] for p in r["patterns"]}
        self.assertNotIn("23451", all_patterns)

    def test_k7_sparse_mode_matches_existing_non_singleton_file(self):
        existing = json.loads((DATA / "reverse_k7.json").read_text())
        generated = grj.build_reverse_data(
            7,
            singletons="none",
        )

        self.assertEqual(class_sets(generated), class_sets(existing))
        self.assertEqual(len(generated["record_classes"]), 144)
        self.assertTrue(all(r["size"] > 1 for r in generated["record_classes"]))

    def test_exceptional_pair_family_has_factorial_count(self):
        pairs = grj.exceptional_pair_classes(8)
        self.assertEqual(len(pairs), 120)
        self.assertIn(frozenset({(2, 3, 1, 4, 5, 6, 7, 8), (3, 1, 2, 4, 5, 6, 7, 8)}), pairs)

    def test_cli_can_write_k5_and_compare_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "reverse_k5.generated.json"
            rc = grj.main_args([
                "5",
                "--output", str(out),
                "--compare-existing", str(DATA / "reverse_k5.json"),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())
            generated = json.loads(out.read_text())
            existing = json.loads((DATA / "reverse_k5.json").read_text())
            self.assertEqual(class_sets(generated), class_sets(existing))


if __name__ == "__main__":
    unittest.main()
