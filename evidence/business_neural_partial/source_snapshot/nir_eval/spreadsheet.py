#!/usr/bin/env python3
"""Read-only, bounded XLSX evaluation for SpreadsheetBench 912 v0.1.

This is a documented local evaluation protocol, not the official leaderboard
runner. It never executes code, macros, formulas, or a spreadsheet application.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, time, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any
import zipfile

SCHEMA_VERSION = "1.0"
PROTOCOL = "nir-spreadsheetbench-values-v1"
DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "office_dataset/SpreadsheetBench/data/all_data_912_v0.1"
PROVENANCE = ("control-not-llm", "replay", "model-inference")
STATUSES = ("pass", "fail", "inconclusive", "reference_error", "invalid_prediction")


class AdapterError(ValueError):
    """A bounded, actionable adapter error."""


@dataclass(frozen=True)
class Limits:
    max_archive_bytes: int = 32 * 1024 * 1024
    max_uncompressed_bytes: int = 128 * 1024 * 1024
    max_entries: int = 4096
    max_compression_ratio: int = 1000
    max_cells: int = 250_000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contained(root: Path, path: Path) -> Path:
    root, resolved = root.resolve(), path.resolve()
    if not resolved.is_relative_to(root):
        raise AdapterError(f"Path escapes allowed root: {path}")
    return resolved


def load_tasks(dataset_root: Path, task_ids: list[str] | None = None) -> list[dict]:
    metadata = contained(dataset_root, dataset_root / "dataset.json")
    if metadata.stat().st_size > 16 * 1024 * 1024:
        raise AdapterError("Dataset metadata exceeds 16 MiB")
    tasks = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise AdapterError("dataset.json must contain an array; only the 912-v0.1 layout is supported")
    seen = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise AdapterError("Each dataset entry must be an object")
        for field in ("id", "instruction", "spreadsheet_path", "instruction_type", "answer_position"):
            if field not in task:
                raise AdapterError(f"Dataset entry is missing {field}")
        task["id"] = str(task["id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task["id"]):
            raise AdapterError(f"Unsafe task id: {task['id']!r}")
        if task["id"] in seen:
            raise AdapterError(f"Duplicate task id: {task['id']}")
        seen.add(task["id"])
        contained(dataset_root, dataset_root / task["spreadsheet_path"])
    requested = set(task_ids or [])
    if requested - seen:
        raise AdapterError(f"Unknown task IDs: {', '.join(sorted(requested - seen))}")
    return [task for task in tasks if not requested or task["id"] in requested]


def discover_cases(dataset_root: Path, task: dict) -> list[dict]:
    """Use the union of input and answer indices; never hide a missing pair."""
    folder = contained(dataset_root, dataset_root / task["spreadsheet_path"])
    if not folder.is_dir():
        return [{"case_index": 1, "input_path": None, "reference_path": None,
                 "warnings": ["missing_task_directory"]}]
    pattern = re.compile(rf"^(\d+)_{re.escape(task['id'])}_(input|answer)( *)\.xlsx$")
    pairs: dict[int, dict] = {}
    for path in sorted(folder.iterdir()):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        index, kind, whitespace = int(match[1]), match[2], match[3]
        case = pairs.setdefault(index, {"case_index": index, "input_path": None,
                                       "reference_path": None, "warnings": []})
        field = "input_path" if kind == "input" else "reference_path"
        if case[field] is not None:
            raise AdapterError(f"Duplicate {kind} files for {task['id']}/{index}")
        case[field] = str(contained(dataset_root, path))
        if whitespace:
            case["warnings"].append(f"nonstandard_filename:{path.name}")
    return [pairs[index] for index in sorted(pairs)] or [
        {"case_index": 1, "input_path": None, "reference_path": None,
         "warnings": ["no_xlsx_cases_found"]}]


def inventory(dataset_root: str | Path, task_ids: list[str] | None = None) -> dict:
    root = Path(dataset_root).resolve()
    tasks = load_tasks(root, task_ids)
    records = [{**task, "cases": discover_cases(root, task)} for task in tasks]
    cases = [case for record in records for case in record["cases"]]
    return {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "kind": "private_evaluator_inventory_not_model_input",
        "dataset": {"root": str(root), "metadata_sha256": digest(root / "dataset.json"),
                    "selected_task_ids": task_ids or None, "format": "912-v0.1"},
        "summary": {"tasks_total": len(records), "cases_total": len(cases),
                    "complete_pairs": sum(bool(c["input_path"] and c["reference_path"]) for c in cases),
                    "case_count_distribution": dict(sorted(Counter(len(t["cases"]) for t in records).items())),
                    "nonstandard_input_names": sum(any(w.startswith("nonstandard_filename:") for w in c["warnings"]) for c in cases)},
        "tasks": records,
    }


def inspect_xlsx(path: Path, limits: Limits = Limits()) -> dict:
    """Inspect ZIP/XML without extraction, external access or active execution."""
    if path.suffix.lower() != ".xlsx":
        raise AdapterError("Only .xlsx is supported; macro-enabled formats are rejected")
    if not path.is_file():
        raise AdapterError(f"Workbook does not exist: {path}")
    if path.stat().st_size > limits.max_archive_bytes:
        raise AdapterError("Workbook exceeds compressed size limit")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                raise AdapterError("Duplicate ZIP members")
            if len(entries) > limits.max_entries:
                raise AdapterError("Too many ZIP members")
            if sum(entry.file_size for entry in entries) > limits.max_uncompressed_bytes:
                raise AdapterError("Workbook exceeds uncompressed size limit")
            if not {"[Content_Types].xml", "xl/workbook.xml"}.issubset(names):
                raise AdapterError("Not an OOXML spreadsheet")
            for entry in entries:
                name = entry.filename
                parts = PurePosixPath(name).parts
                if name.startswith("/") or ".." in parts or "\\" in name:
                    raise AdapterError("Unsafe ZIP member name")
                if entry.flag_bits & 1:
                    raise AdapterError("Encrypted ZIP members are not supported")
                if entry.file_size > limits.max_compression_ratio * max(1, entry.compress_size):
                    raise AdapterError("Suspicious ZIP compression ratio")
                lowered = name.lower()
                if any(token in lowered for token in ("vbaproject", "externallinks/", "embeddings/", "connections.xml", "querytables/", "activex/")):
                    raise AdapterError(f"Active or externally linked content is not supported: {name}")
                if lowered.endswith((".xml", ".rels")):
                    # Bounded by archive-wide uncompressed limit. Explicitly reject
                    # entity/DTD declarations before openpyxl's XML parser sees them.
                    body = archive.read(entry)
                    probe = body.replace(b"\x00", b"").upper()
                    if b"<!DOCTYPE" in probe or b"<!ENTITY" in probe:
                        raise AdapterError("DTD/entity declarations are not supported")
                    if lowered == "[content_types].xml" and b"MACROENABLED" in probe:
                        raise AdapterError("Macro-enabled content disguised as XLSX")
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise AdapterError(f"Invalid XLSX archive: {error}") from error
    return {"sha256": digest(path), "bytes": path.stat().st_size,
            "archive_members": len(entries)}


def _clean_sheet(raw: str) -> str:
    # The released 912 metadata uses inconsistent outer quote placement.
    return raw.strip().strip("'").replace("''", "'")


def resolve_ranges(task: dict, reference, predicted, max_cells: int) -> tuple[list[tuple], list[str]]:
    from openpyxl.utils.cell import range_boundaries
    warnings: list[str] = []
    result = []
    total = 0
    position = task["answer_position"]
    if not isinstance(position, str) or not position.strip():
        raise AdapterError("Missing or empty answer_position")
    if "：" in position:
        warnings.append("normalized_fullwidth_colon_in_answer_position")
        position = position.replace("：", ":")
    # The pinned dataset has no comma in worksheet names. Such names need a
    # separately specified range schema and are intentionally not guessed.
    for part in position.split(","):
        if "!" in part:
            sheet, cell_range = part.rsplit("!", 1)
            sheet = _clean_sheet(sheet)
        else:
            cell_range = part
            answer_sheet = task.get("answer_sheet", "")
            if answer_sheet:
                sheets = [_clean_sheet(name) for name in answer_sheet.split(",")]
                if len(sheets) != 1:
                    raise AdapterError("Ambiguous answer_sheet with an unqualified answer range")
                sheet = sheets[0]
            else:
                sheet = reference.sheetnames[0]
        cell_range = cell_range.strip().strip("'").replace("$", "")
        if sheet not in reference.sheetnames:
            raise AdapterError(f"Reference worksheet does not exist: {sheet}")
        if not re.fullmatch(r"[A-Za-z]{1,3}[1-9][0-9]*(?::[A-Za-z]{1,3}[1-9][0-9]*)?|[A-Za-z]{1,3}:[A-Za-z]{1,3}", cell_range):
            raise AdapterError(f"Unsupported or malformed answer range: {cell_range}")
        c1, r1, c2, r2 = range_boundaries(cell_range)
        if c1 < 1 or c2 > 16384 or c1 > c2:
            raise AdapterError(f"Invalid column bounds: {cell_range}")
        if r1 is None and r2 is None:
            r1 = 1
            r2 = max(reference[sheet].max_row or 1,
                     (predicted[sheet].max_row or 1) if sheet in predicted.sheetnames else 1)
            warnings.append(f"whole_column_range_bounded_to_used_rows:{sheet}!{cell_range}:{r2}")
        if r1 is None or r2 is None or r1 > r2 or r2 > 1048576:
            raise AdapterError(f"Invalid row bounds: {cell_range}")
        total += (c2 - c1 + 1) * (r2 - r1 + 1)
        if total > max_cells:
            raise AdapterError(f"Answer scope exceeds {max_cells} cell limit")
        result.append((sheet, c1, r1, c2, r2))
    return result, warnings


def normalize_value(value: Any) -> Any:
    """Pinned upstream's value normalization; not a general Excel equivalence."""
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if isinstance(value, time):
        return str(value)[:-3]
    if isinstance(value, datetime):
        delta = value - datetime(1899, 12, 30)
        return round(delta.days + delta.seconds / 86400.0, 0)
    if isinstance(value, str):
        try:
            return round(float(value), 2)
        except ValueError:
            return value
    return value


def equal_values(left: Any, right: Any) -> bool:
    left, right = normalize_value(left), normalize_value(right)
    if (left is None or left == "") and (right is None or right == ""):
        return True
    return type(left) is type(right) and left == right


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value[:256]
    return str(value)[:256]


def compare_case(reference_path: str | Path, prediction_path: str | Path,
                 task: dict, formula_policy: str = "strict", limits: Limits = Limits()) -> dict:
    """Return structured diagnostics; do not mutate either workbook.

    strict: any predicted formula in a scored cell is inconclusive. cached:
    permit nonempty cached formula values, without claiming freshness. Missing
    reference caches are reference_error; missing prediction caches inconclusive.
    """
    if formula_policy not in ("strict", "cached"):
        raise AdapterError("formula_policy must be strict or cached")
    reference_path, prediction_path = Path(reference_path), Path(prediction_path)
    result: dict = {"status": "inconclusive", "reason": "not_evaluated", "warnings": [],
                    "cells_checked": 0, "reference_formula_cells": 0,
                    "prediction_formula_cells": 0, "mismatch_count": 0, "mismatch_examples": []}
    if not prediction_path.is_file():
        return {**result, "status": "fail", "reason": "prediction_missing"}
    for path, label, status in ((reference_path, "reference", "reference_error"),
                                (prediction_path, "prediction", "invalid_prediction")):
        try:
            result[f"{label}_artifact"] = inspect_xlsx(path, limits)
        except Exception as error:
            return {**result, "status": status, "reason": f"{label}_validation_failed", "detail": str(error)[:500]}
    try:
        import openpyxl
    except ImportError as error:
        raise AdapterError("openpyxl is required: install requirements.txt") from error
    with ExitStack() as stack:
        workbooks = {}
        for path, label, status in ((reference_path, "reference", "reference_error"),
                                    (prediction_path, "prediction", "invalid_prediction")):
            try:
                # read_only makes hostile dimensions inexpensive at load time;
                # iter_rows below is bounded by an explicit scored-cell limit.
                for data_only in (True, False):
                    workbook = openpyxl.load_workbook(path, read_only=True, data_only=data_only, keep_links=False)
                    stack.callback(workbook.close)
                    workbooks[label, data_only] = workbook
            except Exception as error:
                return {**result, "status": status, "reason": f"{label}_load_failed", "detail": str(error)[:500]}
        ref, pred = workbooks["reference", True], workbooks["prediction", True]
        try:
            ranges, warnings = resolve_ranges(task, ref, pred, limits.max_cells)
            result["warnings"].extend(warnings)
        except Exception as error:
            return {**result, "status": "reference_error", "reason": "answer_scope_invalid", "detail": str(error)[:500]}
        result["answer_scope"] = [{"sheet": s, "min_col": c1, "min_row": r1, "max_col": c2, "max_row": r2} for s, c1, r1, c2, r2 in ranges]
        missing_sheets = [s for s, *_ in ranges if s not in pred.sheetnames]
        reference_missing_cache = prediction_missing_cache = 0
        try:
            for sheet, c1, r1, c2, r2 in ranges:
                streams = [(workbooks[label, values][sheet].iter_rows(min_row=r1, max_row=r2, min_col=c1, max_col=c2)
                            if sheet in workbooks[label, values].sheetnames else iter(()))
                           for label, values in (("reference", True), ("prediction", True), ("reference", False), ("prediction", False))]
                empty_row = (openpyxl.cell.read_only.EmptyCell(),) * (c2 - c1 + 1)
                for row_index in range(r1, r2 + 1):
                    rows = [next(stream, empty_row) for stream in streams]
                    for col_index, (expected, actual, expected_formula, actual_formula) in enumerate(zip(*rows), start=c1):
                        result["cells_checked"] += 1
                        if expected_formula.data_type == "f":
                            result["reference_formula_cells"] += 1
                            reference_missing_cache += expected.value is None
                        if actual_formula.data_type == "f":
                            result["prediction_formula_cells"] += 1
                            prediction_missing_cache += actual.value is None
                        if not equal_values(expected.value, actual.value):
                            result["mismatch_count"] += 1
                            if len(result["mismatch_examples"]) < 5:
                                # Diagnostics intentionally contain reference values.
                                # Keep evaluator reports away from model inputs.
                                result["mismatch_examples"].append({"sheet": sheet, "cell": f"{openpyxl.utils.get_column_letter(col_index)}{row_index}",
                                    "expected": _safe_value(expected.value), "actual": _safe_value(actual.value)})
        except Exception as error:
            return {**result, "status": "inconclusive", "reason": "workbook_read_failed", "detail": str(error)[:500]}
        result["reference_missing_formula_caches"] = reference_missing_cache
        result["prediction_missing_formula_caches"] = prediction_missing_cache
        if reference_missing_cache:
            return {**result, "status": "reference_error", "reason": "reference_formula_cache_missing"}
        if missing_sheets:
            return {**result, "status": "fail", "reason": "prediction_worksheet_missing", "missing_sheets": missing_sheets}
        if prediction_missing_cache:
            return {**result, "status": "inconclusive", "reason": "prediction_formula_cache_missing"}
        if result["prediction_formula_cells"]:
            result["warnings"].append("prediction_formula_cache_freshness_not_verified")
            if formula_policy == "strict":
                return {**result, "status": "inconclusive", "reason": "prediction_formula_requires_verified_recalculation"}
        return {**result, "status": "fail" if result["mismatch_count"] else "pass",
                "reason": "value_mismatch" if result["mismatch_count"] else "all_scored_values_match"}


def evaluate(dataset_root: str | Path, predictions_dir: str | Path, *, provenance: str,
             task_ids: list[str] | None = None, formula_policy: str = "strict",
             limits: Limits = Limits(), run_metadata: dict | None = None) -> dict:
    if provenance not in PROVENANCE:
        raise AdapterError(f"provenance must be one of {PROVENANCE}")
    if formula_policy not in ("strict", "cached"):
        raise AdapterError("formula_policy must be strict or cached")
    root, predictions = Path(dataset_root).resolve(), Path(predictions_dir).resolve()
    # Keep predictions distinct. Prevent the prior upstream input-file fallback
    # and accidental direct evaluation inside the answer-bearing dataset tree.
    if predictions.is_relative_to(root) or root.is_relative_to(predictions):
        raise AdapterError("Predictions directory must be separate from dataset root")
    manifest = inventory(root, task_ids)
    tasks = []
    for task in manifest["tasks"]:
        cases = []
        for case in task["cases"]:
            prediction = contained(predictions, predictions / task["id"] / f"{case['case_index']}_{task['id']}_output.xlsx")
            entry = {"case_index": case["case_index"], "prediction_path": str(prediction),
                     "input_path": case["input_path"], "reference_path": case["reference_path"],
                     "prediction_exists": prediction.is_file()}
            if not case["input_path"] or not case["reference_path"]:
                result = {"status": "reference_error", "reason": "incomplete_dataset_pair", "warnings": []}
            else:
                result = compare_case(case["reference_path"], prediction, task, formula_policy, limits)
            result["warnings"] = case["warnings"] + result.get("warnings", [])
            cases.append({**entry, **result})
        passed = sum(case["status"] == "pass" for case in cases)
        tasks.append({"id": task["id"], "instruction_type": task["instruction_type"], "cases": cases,
                      "hard_pass": bool(cases) and passed == len(cases),
                      "soft_pass_rate": passed / len(cases) if cases else 0.0,
                      "fully_evaluated": all(case["status"] in ("pass", "fail", "invalid_prediction") for case in cases)})
    counts = Counter(case["status"] for task in tasks for case in task["cases"])
    total = sum(counts.values())
    evaluated = counts["pass"] + counts["fail"] + counts["invalid_prediction"]
    unscored = counts["inconclusive"] + counts["reference_error"]
    submitted = sum(case["prediction_exists"] for task in tasks for case in task["cases"])
    compared = sum(case["reason"] in ("all_scored_values_match", "value_mismatch")
                   for task in tasks for case in task["cases"])
    return {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": provenance, "provenance_is_caller_asserted": True,
            "run_metadata": run_metadata or {}, "dataset": manifest["dataset"],
            "predictions_dir": str(predictions), "formula_policy": formula_policy,
            "scope": "Only annotated cell values; no verification of visual formatting, charts, macros or other cells.",
            "limitations": ["Not the official leaderboard protocol: answer_sheet and actual case counts are respected.",
                "No formulas were executed or recalculated. Reference cached values are trusted as distributed.",
                "Cached mode can accept stale or forged formula caches; strict is the conservative default.",
                "Upstream rounding to two decimals, date-to-day and numeric-string coercion are preserved.",
                "Model-inference provenance is caller-declared, not independently verified.",
                "Read-only ZIP/XML screening is defense in depth, not an OS sandbox."],
            "summary": {"tasks_total": len(tasks), "cases_total": total,
                **{f"cases_{status}": counts[status] for status in STATUSES},
                "cases_evaluated": evaluated, "cases_unscored": unscored,
                "coverage": evaluated / total if total else 0.0,
                "decision_coverage": evaluated / total if total else 0.0,
                "cases_with_prediction": submitted, "prediction_coverage": submitted / total if total else 0.0,
                "cases_value_compared": compared, "comparison_coverage": compared / total if total else 0.0,
                "case_pass_rate": counts["pass"] / total if total else 0.0,
                "task_soft_pass_rate": sum(t["soft_pass_rate"] for t in tasks) / len(tasks) if tasks else 0.0,
                "task_hard_pass_rate": sum(t["hard_pass"] for t in tasks) / len(tasks) if tasks else 0.0,
                "rate_interpretation": "lower_bound_unscored_as_not_passed" if unscored else "complete_selected_scope",
                "denominator_policy": "All selected metadata tasks and discovered case indices, including missing predictions and unscored cases."},
            "tasks": tasks}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inventory", "evaluate"):
        child = sub.add_parser(command)
        child.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
        child.add_argument("--task-id", action="append", dest="task_ids")
        child.add_argument("--out", type=Path, required=True)
        if command == "evaluate":
            child.add_argument("--predictions-dir", type=Path, required=True)
            child.add_argument("--provenance", choices=PROVENANCE, required=True)
            child.add_argument("--formula-policy", choices=("strict", "cached"), default="strict")
            child.add_argument("--max-cells", type=int, default=Limits().max_cells)
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            report = inventory(args.dataset_root, args.task_ids)
        else:
            if args.max_cells <= 0 or args.max_cells > 1_000_000:
                raise AdapterError("--max-cells must be between 1 and 1000000")
            report = evaluate(args.dataset_root, args.predictions_dir, provenance=args.provenance,
                              task_ids=args.task_ids, formula_policy=args.formula_policy,
                              limits=Limits(max_cells=args.max_cells))
        # Avoid accidental mutation of benchmark inputs/references or predictions.
        output = args.out.resolve()
        if output.is_relative_to(args.dataset_root.resolve()):
            raise AdapterError("Report output must be outside dataset root")
        if args.command == "evaluate" and output.is_relative_to(args.predictions_dir.resolve()):
            raise AdapterError("Report output must be outside predictions directory")
        write_json(output, report)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        # Non-passing model answers are valid evaluation output, not CLI errors.
        return 0
    except (AdapterError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"Adapter error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
