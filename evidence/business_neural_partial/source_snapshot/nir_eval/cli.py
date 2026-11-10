"""Public CLI; run from the project directory with python -m nir_eval."""
import argparse
import json
from pathlib import Path
import sys

from .datasets import load_finqa, file_sha256
from .importing import import_results
from .comparing import compare_runs
from .selection import select_configuration
from .runner import load_config, export_plan, run, read_records
from .reporting import write_reports
from .scoring import execute_program, answers_equal


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "business":
        from .business.cli import main as business_main
        return business_main(argv[1:])
    if argv and argv[0] == "spreadsheet":
        from .spreadsheet import main as spreadsheet_main
        return spreadsheet_main(argv[1:])
    parser = argparse.ArgumentParser(description="NIR public business-task evaluation MVP")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("business", help="Custom business cases with neural-judge-only quality evaluation")
    sub.add_parser("spreadsheet", help="SpreadsheetBench inventory/evaluate; use spreadsheet --help")
    validate = sub.add_parser("validate", help="Validate FinQA schema and reference execution")
    validate.add_argument("dataset", type=Path)
    plan = sub.add_parser("plan", help="Export leakage-safe prompts without calling any model")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    running = sub.add_parser("run", help="Execute a complete controlled evaluation")
    running.add_argument("--config", type=Path, required=True)
    running.add_argument("--output", type=Path, required=True)
    running.add_argument("--resume", action="store_true")
    running.add_argument("--allow-remote", action="store_true",
                         help="Explicitly allow configured remote provider; check cost, data permission and terms first")
    report = sub.add_parser("report", help="Regenerate reports from existing scored records")
    report.add_argument("records", type=Path)
    report.add_argument("--output", type=Path, required=True)
    importing = sub.add_parser("import-results", help="Verify and score a real external local-model run")
    importing.add_argument("--records", type=Path, required=True)
    importing.add_argument("--manifest", type=Path, required=True)
    importing.add_argument("--dataset", type=Path, required=True)
    importing.add_argument("--case-ids", type=Path)
    importing.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare", help="Compare completed model runs with identical data and prompts")
    compare.add_argument("runs", type=Path, nargs="+")
    compare.add_argument("--output", type=Path, required=True)
    select = sub.add_parser("select", help="Choose a configuration from complete comparable saved evidence")
    select.add_argument("--comparison", type=Path, required=True)
    select.add_argument("--policy", type=Path, required=True)
    select.add_argument("--output", type=Path)
    sub.add_parser("doctor", help="Check local readiness without model or network calls")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .diagnostics import diagnose
            result = diagnose()
        elif args.command == "select":
            result = select_configuration(args.comparison, json.loads(args.policy.read_text()))
            if args.output:
                if args.output.exists():
                    raise ValueError("Selection output already exists; choose a new file")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        elif args.command == "validate":
            cases = load_finqa(args.dataset)
            failures = []
            for case in cases:
                try:
                    answer = execute_program(case.expected_program, case.table)
                    if not answers_equal(answer, case.expected_answer):
                        failures.append({"id": case.id, "reason": "reference answer mismatch"})
                except (ValueError, ZeroDivisionError, OverflowError) as exc:
                    failures.append({"id": case.id, "reason": str(exc)})
            result = {"cases": len(cases), "dataset_sha256": file_sha256(args.dataset),
                      "reference_execution_matches": len(cases) - len(failures),
                      "failures": failures, "model_inference_performed": False}
        elif args.command == "plan":
            result = export_plan(load_config(args.config), args.output)
        elif args.command == "run":
            result = run(load_config(args.config), args.output, args.resume, args.allow_remote)
        elif args.command == "compare":
            result = compare_runs(args.runs, args.output)
        elif args.command == "report":
            result = write_reports(read_records(args.records), args.output)
        else:
            result = import_results(args.records, args.manifest, args.dataset, args.output, args.case_ids)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if args.command == "validate" and result["failures"] else 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
