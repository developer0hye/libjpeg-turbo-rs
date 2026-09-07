"""Tests for the portable-vs-native encode A/B summariser (P4-133, #464).

The summary is what gets pasted into `experiments/encode.tsv` and README, so a
parser that silently dropped a case or paired the wrong numbers would put an
unmeasured claim back where the issue removed one. These pin the line shapes
the two benches actually print, the ratio arithmetic, and the failure mode for
a variant whose run produced nothing.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perf_ab_summary import parse_bench_output, render_table, summarize

RUST_OUTPUT = """\
Case                                                     Size         Time    Iters
-------------------------------------------------------------------------------------
encode_  64x64  _420                                  64x64         13.7 us  (20000 iters)
encode_ 320x240 _420                                 320x240        392.1 us  (5000 iters)
encode_1920x1080_444                                1920x1080    19057.4 us  (500 iters)
"""

C_OUTPUT = """\
Case                                                     Size         Time    Iters
-------------------------------------------------------------------------------------
C_encode_  64x64  _420                                64x64         12.0 us  (20000 iters)
C_encode_ 320x240 _420                               320x240        403.0 us  (5000 iters)
skip: tests/fixtures/photo_1920x1080_444.jpg (not found)
C_encode_1920x1080_444                              1920x1080    19873.0 us  (500 iters)
"""


class ParseBenchOutput(unittest.TestCase):
    def test_reads_rust_lines_with_padded_geometry(self) -> None:
        cases = parse_bench_output(RUST_OUTPUT)
        self.assertEqual(
            cases,
            {
                "64x64_420": 13.7,
                "320x240_420": 392.1,
                "1920x1080_444": 19057.4,
            },
        )

    def test_reads_c_lines_and_ignores_skips(self) -> None:
        cases = parse_bench_output(C_OUTPUT)
        self.assertEqual(
            cases,
            {"64x64_420": 12.0, "320x240_420": 403.0, "1920x1080_444": 19873.0},
        )

    def test_empty_output_is_empty_not_an_error(self) -> None:
        self.assertEqual(parse_bench_output("no bench lines here\n"), {})


class Summarize(unittest.TestCase):
    def test_ratios_are_relative_to_the_named_baseline(self) -> None:
        runs = {
            "stock": {"320x240_420": 400.0, "1920x1080_444": 20000.0},
            "native": {"320x240_420": 380.0, "1920x1080_444": 19000.0},
            "C": {"320x240_420": 403.0, "1920x1080_444": 19873.0},
        }
        rows = summarize(runs, baseline="stock", reference="C")
        by_case = {row.case: row for row in rows}
        native = by_case["1920x1080_444"].columns["native"]
        self.assertAlmostEqual(native.time_us, 19000.0)
        self.assertAlmostEqual(native.vs_baseline, 19000.0 / 20000.0)
        self.assertAlmostEqual(native.vs_reference, 19000.0 / 19873.0)
        stock = by_case["1920x1080_444"].columns["stock"]
        self.assertAlmostEqual(stock.vs_baseline, 1.0)

    def test_cases_are_ordered_by_pixel_count_then_subsampling(self) -> None:
        runs = {
            "stock": {
                "1920x1080_420": 1.0,
                "64x64_420": 1.0,
                "320x240_444": 1.0,
                "320x240_420": 1.0,
            }
        }
        rows = summarize(runs, baseline="stock", reference=None)
        self.assertEqual(
            [row.case for row in rows],
            ["64x64_420", "320x240_420", "320x240_444", "1920x1080_420"],
        )

    def test_a_variant_missing_a_case_is_reported_not_invented(self) -> None:
        runs = {
            "stock": {"320x240_420": 400.0, "1920x1080_444": 20000.0},
            "fma": {"320x240_420": 399.0},
        }
        rows = summarize(runs, baseline="stock", reference=None)
        by_case = {row.case: row for row in rows}
        self.assertIsNone(by_case["1920x1080_444"].columns["fma"])

    def test_unknown_baseline_is_an_error(self) -> None:
        with self.assertRaises(KeyError):
            summarize({"stock": {"64x64_420": 1.0}}, baseline="none", reference=None)


class RenderTable(unittest.TestCase):
    def test_markdown_has_one_row_per_case_and_ratio_columns(self) -> None:
        runs = {
            "stock": {"320x240_420": 400.0},
            "native": {"320x240_420": 380.0},
            "C": {"320x240_420": 403.0},
        }
        text = render_table(summarize(runs, baseline="stock", reference="C"))
        lines = text.splitlines()
        self.assertIn("| Case |", lines[0])
        self.assertIn("native", lines[0])
        self.assertEqual(len(lines), 3)  # header, separator, one case
        self.assertIn("320x240_420", lines[2])
        self.assertIn("0.950", lines[2])  # native / stock
        self.assertIn("0.943", lines[2])  # native / C

    def test_missing_cell_renders_as_dash(self) -> None:
        runs = {"stock": {"64x64_420": 10.0, "320x240_420": 400.0}, "fma": {"64x64_420": 9.0}}
        text = render_table(summarize(runs, baseline="stock", reference=None))
        row = [line for line in text.splitlines() if "320x240_420" in line][0]
        self.assertIn("| — |", row)


class MainEntry(unittest.TestCase):
    def test_reads_label_equals_path_arguments(self) -> None:
        from perf_ab_summary import main

        with tempfile.TemporaryDirectory() as tmp:
            stock = Path(tmp) / "stock.txt"
            stock.write_text(RUST_OUTPUT)
            c = Path(tmp) / "c.txt"
            c.write_text(C_OUTPUT)
            out: list[str] = []
            code = main(
                ["--baseline", "stock", "--reference", "C", f"stock={stock}", f"C={c}"],
                write=out.append,
            )
        self.assertEqual(code, 0)
        text = "".join(out)
        self.assertIn("1920x1080_444", text)
        self.assertIn("0.959", text)  # 19057.4 / 19873.0

    def test_an_empty_run_fails_loudly(self) -> None:
        from perf_ab_summary import main

        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.txt"
            empty.write_text("")
            out: list[str] = []
            code = main(["--baseline", "stock", f"stock={empty}"], write=out.append)
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
