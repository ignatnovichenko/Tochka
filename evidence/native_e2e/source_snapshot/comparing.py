"""Combine completed runs only when their dataset and selected tasks agree."""
import json
from pathlib import Path
from .datasets import file_sha256
from .runner import read_records, atomic_json, utc_now
from .reporting import write_reports


def compare_runs(run_paths, output):
    if len(run_paths) < 2:
        raise ValueError("Comparison requires at least two completed runs")
    records, sources, first_scope, seen = [], [], None, set()
    for run_path in run_paths:
        path = Path(run_path).resolve()
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("status") != "completed" or manifest.get("provenance") != "model-inference":
            raise ValueError("Only completed model-inference runs can be compared")
        prompt_version = manifest.get("prompt_version", manifest.get("source_manifest", {}).get("prompt_version"))
        scope = (manifest.get("dataset_sha256"), tuple(sorted(manifest.get("case_ids", []))), prompt_version)
        if not scope[0] or not scope[1] or not scope[2]:
            raise ValueError("Run is missing dataset hash, selected IDs or prompt version")
        if first_scope is not None and scope != first_scope:
            raise ValueError("Comparison requires identical dataset hash, case IDs and prompt version")
        first_scope = scope
        source_rows = read_records(path / "records.jsonl")
        if len(source_rows) != manifest.get("planned_attempts") or len(source_rows) != manifest.get("completed_attempts"):
            raise ValueError("Source run coverage differs from its manifest")
        for row in source_rows:
            key = (row["id"], row["provider"], row["model"], row["config"], row["repetition"])
            if key in seen:
                raise ValueError("Duplicate model/configuration attempt across source runs")
            seen.add(key)
            records.append({**row, "source_run": path.name})
        sources.append({"name":path.name,"manifest_sha256":file_sha256(path/"manifest.json"),
                        "records_sha256":file_sha256(path/"records.jsonl")})
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Comparison output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    (output/"records.jsonl").write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records))
    atomic_json(output/"manifest.json", {"schema_version":"1.0","status":"completed", "provenance":"model-inference",
                "comparison_created_at":utc_now(),"model_inference_performed_by_comparison":False,
                "dataset_sha256":first_scope[0],"case_ids":list(first_scope[1]),"prompt_version":first_scope[2],
                "planned_attempts":len(records),"completed_attempts":len(records),"sources":sources,
                "limitation":"Comparisons of different runtimes, model sizes or quantizations are comparisons of whole configurations, not isolated causal effects."})
    return write_reports(records, output)
