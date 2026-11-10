"""Sequential deterministic evaluation with durable records and guarded resume."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

from . import __version__
from .datasets import load_finqa, select_cases, file_sha256
from .prompts import build_messages, PROMPT_VERSION, PROTOCOLS
from .providers import make_provider, Generation, ProviderError
from .reporting import write_reports
from .scoring import score_output


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def code_hash():
    root = Path(__file__).parent
    return fingerprint({p.name: file_sha256(p) for p in sorted(root.glob("*.py"))})


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for forbidden in ("api_key", "token", "password", "secret"):
        if forbidden in config.get("provider", {}):
            raise ValueError("Secrets must be supplied via environment variables, never config")
    dataset = config.setdefault("dataset", {})
    dataset["path"] = str((path.parent / dataset["path"]).resolve())
    provider = config.get("provider", {})
    if provider.get("type") == "replay":
        provider["path"] = str((path.parent / provider["path"]).resolve())
    if provider.get("model_path"):
        provider["model_path"] = str((path.parent / provider["model_path"]).resolve())
    if provider.get("cwd"):
        provider["cwd"] = str((path.parent / provider["cwd"]).resolve())
    if not isinstance(config.get("model"), str) or not config["model"]:
        raise ValueError("model is required")
    configurations = config.get("configurations", [])
    if not configurations or len({x["name"] for x in configurations}) != len(configurations):
        raise ValueError("configurations must have unique names")
    for item in configurations:
        if item["name"] not in PROTOCOLS:
            raise ValueError(f"Unknown configuration: {item['name']}")
        if not isinstance(item.get("parameters", {}), dict):
            raise ValueError("Configuration parameters must be an object")
    repetitions = config.get("repetitions", 1)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if config.get("pause_seconds", 0) < 0:
        raise ValueError("pause_seconds cannot be negative")
    return config


def prepare(config):
    dataset = config["dataset"]
    cases = select_cases(load_finqa(dataset["path"]), dataset.get("limit"),
                         config.get("seed", 42), dataset.get("ids"))
    cases_by_id = {c.id: c for c in cases}
    jobs = [(case.id, c["name"], r) for case in cases
            for c in config["configurations"] for r in range(config.get("repetitions", 1))]
    if config.get("shuffle_jobs", True):
        random.Random(config.get("seed", 42)).shuffle(jobs)
    return cases_by_id, jobs


def read_records(path):
    if not Path(path).exists():
        return []
    records = []
    for i, line in enumerate(Path(path).read_text().splitlines(), start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt JSONL line {i}; preserve the file and repair explicitly") from exc
    return records


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


@contextmanager
def output_lock(output):
    lock = output / ".run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("Output directory is locked; confirm no run is active before removing .run.lock") from None
    os.write(fd, f"pid={os.getpid()}\n".encode())
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def export_plan(config, output):
    cases, jobs = prepare(config)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for case_id, protocol, repetition in jobs:
            item = next(c for c in config["configurations"] if c["name"] == protocol)
            row = {"id": case_id, "config": protocol, "repetition": repetition,
                   "model": config["model"], "parameters": item.get("parameters", {}),
                   "prompt_version": PROMPT_VERSION,
                   "messages": build_messages(cases[case_id], protocol)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"tasks": len(cases), "attempts": len(jobs), "prompts": str(path),
            "model_inference_performed": False}


def run(config, output, resume=False, allow_remote=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        cases, jobs = prepare(config)
        cfg_hash = fingerprint(config)
        dataset_hash = file_sha256(config["dataset"]["path"])
        implementation_hash = code_hash()
        manifest_path = output / "manifest.json"
        records_path = output / "records.jsonl"
        if manifest_path.exists():
            if not resume:
                raise ValueError("Run exists; use --resume or choose a new output directory")
            manifest = json.loads(manifest_path.read_text())
            if any(manifest.get(k) != v for k, v in {
                    "config_sha256": cfg_hash, "dataset_sha256": dataset_hash,
                    "code_sha256": implementation_hash}.items()):
                raise ValueError("Resume rejected: config, data or implementation changed")
        else:
            if records_path.exists():
                raise ValueError("Records exist without manifest; refusing to overwrite")
            manifest = {"schema_version": "1.0", "runner_version": __version__,
                        "created_at": utc_now(), "python_version": platform.python_version(),
                        "config_sha256": cfg_hash, "dataset_sha256": dataset_hash,
                        "code_sha256": implementation_hash, "prompt_version": PROMPT_VERSION,
                        "config": config, "case_ids": list(cases), "planned_attempts": len(jobs),
                        "status": "active", "hardware": {"platform": platform.platform(),
                        "machine": platform.machine(), "logical_cpus": os.cpu_count()},
                        "quantization": config.get("quantization", "unknown")}
            atomic_json(manifest_path, manifest)
        records = read_records(records_path)
        seen = set()
        job_set = set(jobs)
        for row in records:
            key = (row["id"], row["config"], row["repetition"])
            if key in seen or key not in job_set:
                raise ValueError("Duplicate or unexpected existing record")
            seen.add(key)
        try:
            provider = make_provider(config["provider"], allow_remote)
        except Exception as exc:
            manifest.update(status="blocked", updated_at=utc_now(), completed_attempts=len(records),
                            blocker_type=type(exc).__name__)
            atomic_json(manifest_path, manifest)
            raise
        provenance = provider.kind if provider else "control-not-llm"
        manifest.update(status="active", provenance=provenance)
        atomic_json(manifest_path, manifest)
        try:
            with records_path.open("a", encoding="utf-8") as stream:
                for case_id, protocol, repetition in jobs:
                    if (case_id, protocol, repetition) in seen:
                        continue
                    case = cases[case_id]
                    configuration = next(c for c in config["configurations"] if c["name"] == protocol)
                    messages = build_messages(case, protocol)
                    row = {"schema_version": "1.0", "id": case_id,
                           "provider": config["provider"].get("name", config["provider"]["type"]),
                           "model": config["model"], "config": protocol, "repetition": repetition,
                           "provenance": provenance, "parameters": configuration.get("parameters", {}),
                           "prompt_sha256": fingerprint(messages), "started_at": utc_now(),
                           "expected_answer": case.expected_answer, "error": None,
                           "raw_output": "", "token_usage": {}, "provider_metadata": {}}
                    started = time.monotonic()
                    try:
                        if provider is None:
                            if config["provider"]["type"] == "invalid-control":
                                generation = Generation("NOT_A_VALID_ANSWER")
                            elif protocol == "program":
                                generation = Generation(json.dumps({"program": case.expected_program}))
                            else:
                                generation = Generation(json.dumps({"answer": case.expected_answer}))
                        else:
                            generation = provider.generate(messages, config["model"],
                                configuration.get("parameters", {}),
                                {"id": case_id, "config": protocol, "repetition": repetition})
                        row.update(raw_output=generation.text, token_usage=generation.usage,
                                   provider_metadata=generation.metadata)
                    except (ProviderError, OSError) as exc:
                        row["error"] = str(exc)[:500]
                    row["latency_seconds"] = round(time.monotonic() - started, 6)
                    if row["error"]:
                        row.update(parsed_answer=None, correct=False, parse_error=None,
                                   predicted_program=None, program_exact_match=None)
                    else:
                        row.update(asdict(score_output(row["raw_output"], protocol,
                            case.expected_answer, case.expected_program, case.table)))
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    records.append(row)
                    print(f"{len(records)}/{len(jobs)} {protocol} {case_id} correct={row['correct']}", flush=True)
                    if config.get("pause_seconds", 0):
                        time.sleep(config["pause_seconds"])
            manifest.update(status="completed", completed_at=utc_now(), completed_attempts=len(records))
        except BaseException:
            manifest.update(status="interrupted", updated_at=utc_now(), completed_attempts=len(records))
            atomic_json(manifest_path, manifest)
            write_reports(records, output)
            raise
        atomic_json(manifest_path, manifest)
        return write_reports(records, output)
