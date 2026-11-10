"""Synthetic fixtures test the evaluator, not model capabilities."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
import sys
from datetime import datetime, time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nir_eval import spreadsheet as adapter
import openpyxl


def workbook(path, cells=None, sheet="Sheet1", extra_sheet=None):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    if extra_sheet:
        ws = wb.create_sheet(extra_sheet)
    for coordinate, value in (cells or {}).items():
        ws[coordinate] = value
    wb.save(path)
    wb.close()


def rewrite_zip(path, transform):
    with zipfile.ZipFile(path) as source:
        entries = [(entry.filename, source.read(entry)) for entry in source.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for name, data in entries:
            target.writestr(name, transform(name, data))


def cache_formula(path, value="2"):
    rewrite_zip(path, lambda name, data: data.replace(b"<v></v>", f"<v>{value}</v>".encode()).replace(b"<v />", f"<v>{value}</v>".encode()).replace(b"<v/>", f"<v>{value}</v>".encode())
                if name == "xl/worksheets/sheet1.xml" else data)


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ref, self.pred = self.root / "ref.xlsx", self.root / "pred.xlsx"
        self.task = {"id": "demo", "instruction": "Unit fixture", "spreadsheet_path": "spreadsheet/demo",
                     "instruction_type": "Cell-Level Manipulation", "answer_position": "A1"}

    def tearDown(self):
        self.temp.cleanup()

    def compare(self, **kwargs):
        return adapter.compare_case(self.ref, self.pred, self.task, **kwargs)

    def make_pair(self, expected=1, actual=1):
        workbook(self.ref, {"A1": expected})
        workbook(self.pred, {"A1": actual})

    def dataset(self, cases=(1,), nonstandard=False):
        data = self.root / "dataset"
        folder = data / "spreadsheet/demo"
        folder.mkdir(parents=True)
        (data / "dataset.json").write_text(json.dumps([self.task]))
        for index in cases:
            blank = " " if nonstandard else ""
            workbook(folder / f"{index}_demo_input{blank}.xlsx", {"A1": 1})
            workbook(folder / f"{index}_demo_answer.xlsx", {"A1": 2})
        return data

    def test_equal_values_pass(self):
        self.make_pair()
        result = self.compare()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["cells_checked"], 1)
        self.assertEqual(result["reference_artifact"]["sha256"], adapter.digest(self.ref))

    def test_mismatch_is_fail(self):
        self.make_pair(actual=2)
        result = self.compare()
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["mismatch_examples"][0]["cell"], "A1")

    def test_empty_reference_cell_mismatch(self):
        self.make_pair(expected=None, actual=3)
        self.assertEqual(self.compare()["status"], "fail")

    def test_missing_prediction(self):
        workbook(self.ref, {"A1": 2})
        self.assertEqual(self.compare()["reason"], "prediction_missing")

    def test_missing_predicted_sheet(self):
        workbook(self.ref, {"A1": 2})
        workbook(self.pred, {"A1": 2}, sheet="wrong")
        self.assertEqual(self.compare()["reason"], "prediction_worksheet_missing")

    def test_answer_sheet_is_respected(self):
        workbook(self.ref, {"A1": 2}, extra_sheet="Result")
        workbook(self.pred, {"A1": 3}, extra_sheet="Result")
        self.task["answer_sheet"] = "Result"
        self.assertEqual(self.compare()["status"], "fail")

    def test_ambiguous_answer_sheet_is_not_guessed(self):
        self.make_pair()
        self.task["answer_sheet"] = "Sheet1,Other"
        self.assertEqual(self.compare()["status"], "reference_error")

    def test_legacy_quoting_and_fullwidth_colon(self):
        self.make_pair()
        self.task["answer_position"] = "'Sheet1!'A1：A1'"
        result = self.compare()
        self.assertEqual(result["status"], "pass")
        self.assertIn("normalized_fullwidth_colon_in_answer_position", result["warnings"])

    def test_bad_range_is_not_guessed(self):
        self.make_pair()
        self.task["answer_position"] = "A1:30"
        self.assertEqual(self.compare()["status"], "reference_error")

    def test_whole_column_checks_extra_prediction_row(self):
        self.make_pair()
        workbook(self.pred, {"A1": 1, "A3": 99})
        self.task["answer_position"] = "A:A"
        result = self.compare()
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["cells_checked"], 3)

    def test_cell_limit(self):
        self.make_pair()
        self.task["answer_position"] = "A1:A100"
        result = self.compare(limits=adapter.Limits(max_cells=2))
        self.assertEqual(result["status"], "reference_error")

    def test_formula_without_prediction_cache(self):
        self.make_pair(expected=2, actual="=1+1")
        self.assertEqual(self.compare()["reason"], "prediction_formula_cache_missing")
        self.assertEqual(self.compare(formula_policy="cached")["status"], "inconclusive")

    def test_formula_cache_strict_vs_cached(self):
        self.make_pair(expected=2, actual="=1+1")
        cache_formula(self.pred)
        self.assertEqual(self.compare()["status"], "inconclusive")
        result = self.compare(formula_policy="cached")
        self.assertEqual(result["status"], "pass")
        self.assertIn("prediction_formula_cache_freshness_not_verified", result["warnings"])

    def test_both_empty_formula_caches_never_pass(self):
        self.make_pair(expected="=1+1", actual="=1+1")
        self.assertEqual(self.compare()["status"], "reference_error")

    def test_bad_reference_precedes_missing_prediction_sheet(self):
        workbook(self.ref, {"A1": "=1+1"})
        workbook(self.pred, {"A1": 2}, sheet="wrong")
        self.assertEqual(self.compare()["status"], "reference_error")

    def test_missing_output_has_zero_prediction_coverage(self):
        dataset = self.dataset()
        report = adapter.evaluate(dataset, self.root / "outputs", provenance="control-not-llm")
        self.assertEqual(report["summary"]["prediction_coverage"], 0)
        self.assertEqual(report["summary"]["comparison_coverage"], 0)
        self.assertEqual(report["summary"]["decision_coverage"], 1)

    def test_cached_reference_and_value_prediction(self):
        self.make_pair(expected="=1+1", actual=2)
        cache_formula(self.ref)
        self.assertEqual(self.compare()["status"], "pass")

    def test_invalid_prediction_zip(self):
        workbook(self.ref, {"A1": 2})
        self.pred.write_text("not zip")
        self.assertEqual(self.compare()["status"], "invalid_prediction")

    def test_macro_zip_rejected(self):
        self.make_pair()
        with zipfile.ZipFile(self.pred, "a") as archive:
            archive.writestr("xl/vbaProject.bin", b"fake macro")
        self.assertEqual(self.compare()["status"], "invalid_prediction")

    def test_external_links_rejected(self):
        self.make_pair()
        with zipfile.ZipFile(self.pred, "a") as archive:
            archive.writestr("xl/externalLinks/externalLink1.xml", b"<xml/>")
        self.assertEqual(self.compare()["status"], "invalid_prediction")

    def test_dtd_rejected(self):
        self.make_pair()
        rewrite_zip(self.pred, lambda name, data: b'<!DOCTYPE x [<!ENTITY x "danger">]>' + data
                    if name == "xl/workbook.xml" else data)
        self.assertEqual(self.compare()["status"], "invalid_prediction")

    def test_size_limit(self):
        self.make_pair()
        with self.assertRaises(adapter.AdapterError):
            adapter.inspect_xlsx(self.pred, adapter.Limits(max_archive_bytes=10))

    def test_unsafe_zip_path_rejected(self):
        self.make_pair()
        with zipfile.ZipFile(self.pred, "a") as archive:
            archive.writestr("../outside", b"x")
        self.assertEqual(self.compare()["status"], "invalid_prediction")

    def test_missing_output_no_input_fallback(self):
        dataset = self.dataset()
        report = adapter.evaluate(dataset, self.root / "outputs", provenance="control-not-llm")
        self.assertEqual(report["summary"]["cases_fail"], 1)
        self.assertEqual(report["tasks"][0]["cases"][0]["reason"], "prediction_missing")

    def test_irregular_counts_and_whitespace_input(self):
        dataset = self.dataset(cases=(1, 3), nonstandard=True)
        report = adapter.inventory(dataset)
        self.assertEqual(report["summary"]["cases_total"], 2)
        self.assertEqual(report["summary"]["complete_pairs"], 2)
        self.assertEqual(report["summary"]["nonstandard_input_names"], 2)

    def test_missing_pair_not_dropped(self):
        dataset = self.dataset(cases=(1, 2))
        (dataset / "spreadsheet/demo/2_demo_input.xlsx").unlink()
        report = adapter.evaluate(dataset, self.root / "outputs", provenance="control-not-llm")
        self.assertEqual(report["summary"]["cases_total"], 2)
        self.assertEqual(report["summary"]["cases_reference_error"], 1)
        self.assertEqual(report["summary"]["coverage"], 0.5)

    def test_prediction_path_cannot_escape(self):
        dataset = self.dataset()
        outputs = self.root / "outputs"
        outputs.mkdir()
        (outputs / "demo").symlink_to(dataset / "spreadsheet/demo", target_is_directory=True)
        with self.assertRaises(adapter.AdapterError):
            adapter.evaluate(dataset, outputs, provenance="control-not-llm")

    def test_predictions_not_dataset_directory(self):
        dataset = self.dataset()
        with self.assertRaises(adapter.AdapterError):
            adapter.evaluate(dataset, dataset / "outputs", provenance="replay")

    def test_metadata_path_cannot_escape(self):
        dataset = self.dataset()
        self.task["spreadsheet_path"] = "../outside"
        (dataset / "dataset.json").write_text(json.dumps([self.task]))
        with self.assertRaises(adapter.AdapterError):
            adapter.inventory(dataset)

    def test_provenance_is_required(self):
        dataset = self.dataset()
        with self.assertRaises(adapter.AdapterError):
            adapter.evaluate(dataset, self.root / "outputs", provenance="fake")
        with self.assertRaises(TypeError):
            adapter.evaluate(dataset, self.root / "outputs")

    def test_unscored_cases_stay_in_denominator(self):
        dataset = self.dataset(cases=(1, 2))
        outputs = self.root / "outputs/demo"
        outputs.mkdir(parents=True)
        workbook(outputs / "1_demo_output.xlsx", {"A1": 2})
        workbook(outputs / "2_demo_output.xlsx", {"A1": "=1+1"})
        report = adapter.evaluate(dataset, outputs.parent, provenance="control-not-llm")
        self.assertEqual(report["summary"]["case_pass_rate"], 0.5)
        self.assertEqual(report["summary"]["coverage"], 0.5)
        self.assertEqual(report["summary"]["task_hard_pass_rate"], 0.0)
        self.assertEqual(report["summary"]["rate_interpretation"], "lower_bound_unscored_as_not_passed")

    def test_reference_and_prediction_are_not_modified(self):
        self.make_pair()
        before = adapter.digest(self.ref), adapter.digest(self.pred)
        self.compare()
        self.assertEqual(before, (adapter.digest(self.ref), adapter.digest(self.pred)))

    def test_normalization_pinned_quirks(self):
        self.assertTrue(adapter.equal_values("1.234", 1.23))
        self.assertTrue(adapter.equal_values(None, ""))
        self.assertTrue(adapter.equal_values(True, 1))
        self.assertEqual(adapter.normalize_value(time(12, 30, 0)), "12:30")
        self.assertEqual(adapter.normalize_value(datetime(1900, 1, 1)), 2.0)
        self.assertFalse(adapter.equal_values("text", "Text"))


if __name__ == "__main__":
    unittest.main()
