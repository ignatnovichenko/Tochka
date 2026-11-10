"""Score externally executed model runs without pretending to run inference again."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil

from .datasets import load_finqa, file_sha256
from .prompts import build_messages, PROTOCOLS
from .runner import read_records, atomic_json, fingerprint, utc_now, code_hash
from .scoring import score_output
from .reporting import write_reports


def import_results(records_path, manifest_path, dataset_path, output, case_ids_path=None):
    records_path, manifest_path = Path(records_path), Path(manifest_path)
    source = json.loads(manifest_path.read_text())
    if not source.get("completed_at_utc"):
        raise ValueError("Source run is not marked completed; do not score a file still being written")
    if source.get("records_sha256") and source["records_sha256"] != file_sha256(records_path):
        raise ValueError("Source records hash does not match its manifest")
    if source.get("inference_type") != "real_local_model_inference":
        raise ValueError("Manifest must explicitly identify real_local_model_inference")
    actual_dataset_hash = file_sha256(dataset_path)
    if source.get("dataset_sha256") != actual_dataset_hash:
        raise ValueError("Source manifest dataset hash does not match provided dataset")
    ids_path = Path(case_ids_path) if case_ids_path else manifest_path.parent / "case_ids.json"
    ids = json.loads(ids_path.read_text())
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or len(ids) != source.get("n_cases"):
        raise ValueError("Source case IDs missing, duplicate or inconsistent with manifest")
    protocols = source.get("protocols", [])
    if not protocols or len(protocols) != len(set(protocols)) or not set(protocols) <= set(PROTOCOLS):
        raise ValueError("Source protocols missing or unsupported")
    repetitions = source.get("repetitions", 1)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("Invalid repetitions in source manifest")
    expected = {(i, p, r) for i in ids for p in protocols for r in range(repetitions)}
    records = read_records(records_path)
    keys = [(r["id"], r["config"], r["repetition"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Source records must cover exact case IDs × protocols × repetitions; run may be incomplete")
    cases = {c.id: c for c in load_finqa(dataset_path)}
    if not set(ids) <= cases.keys():
        raise ValueError("Source IDs not found in dataset")
    scored = []
    for original in records:
        row = dict(original)
        case = cases[row["id"]]
        latency = row.get("latency_seconds")
        if latency is not None and (not isinstance(latency, (int, float)) or
                not math.isfinite(latency) or latency < 0):
            raise ValueError("Record latency must be finite and nonnegative")
        if row.get("parameters") is not None and row["parameters"] != source.get("parameters"):
            raise ValueError("Record parameters differ from source manifest")
        if row.get("model") != source.get("model_id"):
            raise ValueError("Record model differs from source manifest")
        if row.get("model_revision") != source.get("model_revision"):
            raise ValueError("Record model revision differs from source manifest")
        if row.get("messages") != build_messages(case, row["config"]):
            raise ValueError("Recorded messages differ from current leakage-safe prompt")
        row.update(schema_version="1.0", provenance="model-inference",
                   imported=True, expected_answer=case.expected_answer,
                   prompt_sha256=fingerprint(row["messages"]))
        if row.get("error"):
            row.update(parsed_answer=None, correct=False, parse_error=None,
                       predicted_program=None, program_exact_match=None)
        else:
            row.update(asdict(score_output(row["raw_output"], row["config"],
                              case.expected_answer, case.expected_program, case.table)))
        scored.append(row)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Import output must be empty to preserve prior reports")
    output.mkdir(parents=True, exist_ok=True)
    (output / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in scored))
    manifest = {"schema_version": "1.0", "status": "completed", "provenance": "model-inference",
                "imported": True, "imported_at": utc_now(), "model_inference_performed_by_import": False,
                "source_records_sha256": file_sha256(records_path), "source_manifest_sha256": file_sha256(manifest_path),
                "dataset_sha256": actual_dataset_hash, "scoring_code_sha256": code_hash(),
                "case_ids": ids, "planned_attempts": len(expected), "completed_attempts": len(scored),
                "source_manifest": source}
    atomic_json(output / "manifest.json", manifest)
    return write_reports(scored, output)
