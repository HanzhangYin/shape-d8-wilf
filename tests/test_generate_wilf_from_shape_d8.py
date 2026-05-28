import json
import tempfile
import unittest
from pathlib import Path

import generate_wilf_from_shape_d8 as d8

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class D8OperationTests(unittest.TestCase):
    def test_basic_d8_orbit_for_123(self):
        orbit = {d8.format_pattern(p) for p in d8.d8_orbit((1, 2, 3))}
        self.assertEqual(orbit, {"123", "321"})

    def test_inverse(self):
        self.assertEqual(d8.inverse_pattern((2, 3, 1)), (3, 1, 2))


class GenerationTests(unittest.TestCase):
    def test_bundled_reverse_data_starts_at_k5(self):
        bundled = sorted(path.name for path in DATA.glob("reverse_k*.json"))
        self.assertEqual(
            bundled,
            ["reverse_k5.json", "reverse_k6.json", "reverse_k7.json", "reverse_k8.json"],
        )

    def test_k5_generation_matches_expected_summary(self):
        k, classes, _ = d8.load_shape_classes(DATA / "reverse_k5.json")
        self.assertEqual(k, 5)
        completed, added = d8.complete_d8_images(classes)
        self.assertEqual([d8.format_pattern(p) for p in added], ["23451"])
        components = d8.build_components(completed)
        report = d8.make_report(
            source=DATA / "reverse_k5.json",
            k=k,
            original_classes=classes,
            completed_classes=completed,
            added_singletons=added,
            components=components,
        )
        self.assertEqual(report["input_pattern_count"], 119)
        self.assertEqual(report["completed_pattern_count"], 120)
        self.assertEqual(report["generated_class_count"], 16)
        self.assertEqual(report["generated_size_distribution"], {2: 2, 4: 4, 8: 8, 16: 1, 20: 1})

    def test_k5_matches_bundled_wilf_table(self):
        k, classes, _ = d8.load_shape_classes(DATA / "reverse_k5.json")
        completed, added = d8.complete_d8_images(classes)
        report = d8.make_report(
            source=DATA / "reverse_k5.json",
            k=k,
            original_classes=classes,
            completed_classes=completed,
            added_singletons=added,
            components=d8.build_components(completed),
        )
        ok, lines = d8.compare_with_wilf_txt(report, DATA / "wilf.txt")
        self.assertTrue(ok, "\n".join(lines))

    def test_cli_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_out = tmp_path / "out.json"
            txt_out = tmp_path / "out.txt"
            rc = d8.main_args([
                "--input", str(DATA / "reverse_k5.json"),
                "--output-json", str(json_out),
                "--output-txt", str(txt_out),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(json_out.exists())
            self.assertTrue(txt_out.exists())
            report = json.loads(json_out.read_text())
            self.assertEqual(report["k"], 5)
            self.assertEqual(report["completed_pattern_count"], 120)
            self.assertEqual(report["generated_class_count"], 16)


if __name__ == "__main__":
    unittest.main()
