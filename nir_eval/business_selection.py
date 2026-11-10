"""Read-only selection from completed, hash-verified business neural-judge runs.

No model calls. Diagnostics never determine quality. Unknown judgement is not zero.
Copy this module to nir_eval/business_selection.py for package integration.
"""
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import re
import statistics

from nir_eval.business.schema import ValidationError, digest, load_jsonl, strict_json


HEX = re.compile(r"[0-9a-f]{64}\Z")


def _number(value, name, low=0, high=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < low or (high is not None and value > high):
        raise ValidationError(f"Invalid {name}")
    return value


def _duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _hash(value):
    return isinstance(value, str) and bool(HEX.fullmatch(value))


def validate_business_policy(policy):
    if not isinstance(policy, dict):
        raise ValidationError("Policy must be an object")
    policy = dict(policy)
    for alias, canonical in [("max_mean_solver_latency", "max_mean_solver_latency_seconds"),
                             ("max_mean_total_latency", "max_mean_total_latency_seconds")]:
        if alias in policy:
            if canonical in policy:
                raise ValidationError("Duplicate latency policy field")
            policy[canonical] = policy.pop(alias)
    allowed = {"min_judge_coverage", "min_mean_judge_score", "max_generation_error_rate",
               "max_judge_critical_error_rate", "min_judge_pass_rate", "min_tasks", "objective",
               "max_mean_solver_latency_seconds", "max_mean_total_latency_seconds", "max_mean_cost_usd"}
    if policy.keys() - allowed:
        raise ValidationError("Unknown business selection policy field")
    p = {"min_judge_coverage": 1.0, "min_mean_judge_score": 0.0,
         "max_generation_error_rate": 0.0, "max_judge_critical_error_rate": 0.0,
         "min_tasks": 1, "objective": "quality", **policy}
    for key in ("min_judge_coverage", "min_mean_judge_score", "max_generation_error_rate", "max_judge_critical_error_rate"):
        _number(p[key], key, high=1)
    if p.get("min_judge_pass_rate") is not None:
        _number(p["min_judge_pass_rate"], "min_judge_pass_rate", high=1)
    for key in ("max_mean_solver_latency_seconds", "max_mean_total_latency_seconds", "max_mean_cost_usd"):
        if p.get(key) is not None:
            _number(p[key], key)
    if type(p["min_tasks"]) is not int or p["min_tasks"] < 1:
        raise ValidationError("min_tasks must be a positive integer")
    if p["objective"] != "quality":
        raise ValidationError("Business selector currently supports quality objective only")
    return p


def _key(row):
    try:
        key = tuple(row[k] for k in ("id", "model", "configuration", "repetition"))
    except (KeyError, TypeError):
        raise ValidationError("Record is missing attempt identity") from None
    if any(not isinstance(x, str) or not x for x in key[:3]) or type(key[3]) is not int or key[3] < 0:
        raise ValidationError("Invalid attempt identity")
    return key


def _index(rows):
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ValidationError("Duplicate attempt")
        result[key] = row
    return result


def load_business_evidence(run_dir):
    root = Path(run_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = strict_json(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "business-pipeline-v1" or manifest.get("state") != "completed":
        raise ValidationError("Selection requires a completed business-pipeline-v1 run")
    if manifest.get("target_quality_method") != "neural_judge_only":
        raise ValidationError("Business selection requires neural_judge_only")
    for field in ("cases_sha256", "rubrics_sha256", "solver_config_sha256", "judge_config_sha256"):
        if not _hash(manifest.get(field)):
            raise ValidationError("Missing input/configuration fingerprint")
    tables = {}
    for filename, field in (("responses.jsonl", "responses_sha256"),
                            ("solver_records.jsonl", "solver_records_sha256"),
                            ("records.jsonl", "records_sha256")):
        path = root / filename
        if not _hash(manifest.get(field)) or hashlib.sha256(path.read_bytes()).hexdigest() != manifest[field]:
            raise ValidationError(f"File hash mismatch: {filename}")
        tables[filename] = _index(load_jsonl(path))
    responses, solver, evaluations = (tables[x] for x in ("responses.jsonl", "solver_records.jsonl", "records.jsonl"))
    if responses.keys() != solver.keys() or responses.keys() != evaluations.keys():
        raise ValidationError("Solver/response/judge attempt coverage differs")
    n = len(responses)
    for field in ("planned_attempts", "solver_completed_attempts", "evaluated_attempts"):
        if type(manifest.get(field)) is not int or manifest[field] != n:
            raise ValidationError("Manifest attempt counts differ from records")
    ids = manifest.get("case_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise ValidationError("Invalid planned case identifiers")
    model = manifest.get("solver_model")
    if not isinstance(model, str) or not model or any(k[1] != model for k in responses):
        raise ValidationError("Solver model differs from manifest")
    configs = sorted({k[2] for k in responses})
    repetitions = sorted({k[3] for k in responses})
    if repetitions != list(range(len(repetitions))):
        raise ValidationError("Repetition indices must be contiguous from zero")
    expected = {(i, model, c, r) for i in ids for c in configs for r in repetitions}
    if set(responses) != expected:
        raise ValidationError("Incomplete or unexpected case/configuration/repetition matrix")
    explicit_plan = "configuration_names" in manifest and "repetitions" in manifest
    if "configuration_names" in manifest:
        plan = manifest["configuration_names"]
        if not isinstance(plan, list) or any(not isinstance(x, str) for x in plan) or len(plan) != len(set(plan)) or set(plan) != set(configs):
            raise ValidationError("Configuration plan differs from saved attempts")
    if "repetitions" in manifest and (type(manifest["repetitions"]) is not int or manifest["repetitions"] != len(repetitions)):
        raise ValidationError("Repetition plan differs from saved attempts")
    groups = defaultdict(list)
    case_signatures = {}
    judge_identities = set()
    evaluated_count = 0
    for key in sorted(expected):
        response, generation, row = responses[key], solver[key], evaluations[key]
        if response.get("provenance") != "model-inference" or generation.get("provenance") != "model-inference" or row.get("response_provenance") != "model-inference":
            raise ValidationError("Controls, replay and unverified answers cannot be selected")
        if row.get("schema_version") != "business-judge-v1" or row.get("checks_role") != "diagnostic_only":
            raise ValidationError("Unsupported evaluation or diagnostics quality contract")
        if response.get("status") not in {"ok", "error"} or response["status"] != row.get("response_status"):
            raise ValidationError("Generation/evaluation response status mismatch")
        ok = response["status"] == "ok"
        if ok == bool(generation.get("error")):
            raise ValidationError("Generation error status mismatch")
        if not isinstance(generation.get("messages"), list) or generation.get("prompt_sha256") != digest(generation["messages"]):
            raise ValidationError("Solver prompt fingerprint mismatch")
        if not isinstance(response.get("response", ""), str) or generation.get("raw_output") != response.get("response", ""):
            raise ValidationError("Saved solver response mismatch")
        if row.get("candidate_sha256") != (digest(response["response"]) if ok else None):
            raise ValidationError("Candidate fingerprint mismatch")
        signature = (row.get("case_sha256"), row.get("rubric_id"), row.get("rubric_sha256"))
        if not _hash(signature[0]) or not isinstance(signature[1], str) or not signature[1] or not _hash(signature[2]):
            raise ValidationError("Case/rubric fingerprint missing")
        if key[0] in case_signatures and signature != case_signatures[key[0]]:
            raise ValidationError("Case/rubric changed between configurations")
        case_signatures[key[0]] = signature
        judge = row.get("judge")
        if not isinstance(judge, dict) or judge.get("status") not in {"evaluated", "error", "unavailable", "not_evaluated"}:
            raise ValidationError("Invalid judge status")
        if not isinstance(judge.get("model"), str) or not isinstance(judge.get("provenance"), str):
            raise ValidationError("Missing judge identity")
        judge_identities.add((judge["model"], judge["provenance"]))
        if judge["status"] == "evaluated":
            evaluated_count += 1
            if not ok or judge.get("provenance") != "model-inference" or manifest.get("judge_type") == "disabled":
                raise ValidationError("Evaluated quality requires a real neural judgement and generated answer")
            _number(judge.get("score"), "judge score", high=1)
            if type(judge.get("passed")) is not bool or row.get("score") != judge["score"] or row.get("passed") is not judge["passed"]:
                raise ValidationError("Top-level quality must equal neural judgement")
            if row.get("status") != ("passed" if judge["passed"] else "failed"):
                raise ValidationError("Judgement/status mismatch")
            critical = judge.get("critical_errors")
            if not isinstance(critical, list) or any(not isinstance(x, dict) or type(x.get("detected")) is not bool for x in critical):
                raise ValidationError("Missing or invalid judge critical-error assessment")
            if judge["passed"] and any(x["detected"] for x in critical):
                raise ValidationError("Critical neural error cannot be marked passed")
        else:
            if any(obj.get(field) is not None for obj in (row, judge) for field in ("score", "passed")):
                raise ValidationError("Unknown judgement must have null score and pass status")
            if row.get("status") not in {"evaluation_error", "not_evaluated"}:
                raise ValidationError("Unknown judgement has inconsistent quality status")
        groups[(key[1], key[2])].append((generation, row))
    if len(judge_identities) != 1:
        raise ValidationError("One selection run must use one judge identity")
    if type(manifest.get("judge_evaluated")) is not int or manifest["judge_evaluated"] != evaluated_count:
        raise ValidationError("Manifest judgement count mismatch")
    candidates = []
    for (model, config), pairs in sorted(groups.items()):
        rows = [r for _, r in pairs]
        scored = [r for r in rows if r["judge"]["status"] == "evaluated"]
        scores = [r["judge"]["score"] for r in scored]
        solver_times = [s.get("latency_seconds") for s, _ in pairs]
        solver_scopes = {s.get("latency_scope") for s, _ in pairs}
        solver_complete = all(_duration(x) for x in solver_times) and len(solver_scopes) == 1 and all(isinstance(x, str) and x.strip() for x in solver_scopes)
        judge_times = [r["judge"].get("latency_seconds") for r in rows]
        judge_scope = manifest.get("judge_latency_scope")
        total_scope = manifest.get("total_latency_scope")
        # Historical business-pipeline-v1 does not declare judge timing boundaries.
        total_complete = (solver_complete and all(_duration(x) for x in judge_times)
                          and isinstance(judge_scope, str) and bool(judge_scope.strip())
                          and total_scope == "solver_plus_judge_call_seconds")
        cost_contract = manifest.get("cost_measurement")
        costs = [r.get("total_cost_usd") for r in rows]
        cost_complete = (cost_contract == {"currency": "USD", "scope": "solver_plus_judge_per_attempt", "basis": "provider_reported"}
                         and all(_duration(x) for x in costs))
        count = len(rows)
        candidates.append({"model": model, "configuration": config, "attempts": count,
            "unique_tasks": len(ids), "repetitions": len(repetitions), "attempt_coverage": 1.0,
            "judge_evaluated": len(scored), "judge_coverage": len(scored) / count,
            "judge_unknown": count - len(scored),
            "judge_status_counts": {status: sum(r["judge"]["status"] == status for r in rows)
                                    for status in ("evaluated", "unavailable", "error", "not_evaluated")},
            "mean_judge_score": statistics.fmean(scores) if scores else None,
            "judge_pass_rate_among_evaluated": sum(r["judge"]["passed"] for r in scored) / len(scored) if scored else None,
            "judge_critical_error_rate_among_evaluated": sum(any(x["detected"] for x in r["judge"]["critical_errors"]) for r in scored) / len(scored) if scored else None,
            "generation_error_rate": sum(r["response_status"] == "error" for r in rows) / count,
            "mean_solver_latency_seconds": statistics.fmean(solver_times) if solver_complete else None,
            "solver_latency_scope": next(iter(solver_scopes)) if solver_complete else None,
            "mean_total_latency_seconds": statistics.fmean(a + b for a, b in zip(solver_times, judge_times)) if total_complete else None,
            "total_latency_scope": total_scope if total_complete else None,
            "mean_cost_usd": statistics.fmean(costs) if cost_complete else None})
    return candidates, manifest, explicit_plan, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def select_business_run(run_dir, policy):
    p = validate_business_policy(policy)
    candidates, manifest, explicit_plan, manifest_hash = load_business_evidence(run_dir)
    result = {"schema_version": "business-selection-v1", "quality_method": "neural_judge_only",
              "policy": p, "status": "no_eligible_configuration", "selected": None, "tied": [],
              "candidates": candidates, "source_manifest_sha256": manifest_hash,
              "cases_sha256": manifest["cases_sha256"], "rubrics_sha256": manifest["rubrics_sha256"],
              "judge_config_sha256": manifest["judge_config_sha256"],
              "configuration_plan_explicit_in_manifest": explicit_plan,
              "model_calls_performed": False, "quality_basis": "observed_neural_scores_among_evaluated",
              "ranking_rule": "Highest mean neural-judge score among eligible configurations. Ties are returned without arbitrary tiebreak.",
              "warnings": ["This is a descriptive selection on saved cases, not a guarantee of future quality or an SLA.",
                  "Judge unknown/error/unavailable scores remain null. Coverage uses all planned attempts.",
                  "Deterministic checks are diagnostics only and never used for quality selection.",
                  "The judge is not independently certified. Scores and critical errors reflect its judgement.",
                  "Hash verification detects changed evidence relative to the manifest, not source authenticity.",
                  "Cost stays unknown without complete provider-reported solver-plus-judge costs.",
                  "Total latency means the sum of recorded solver and judge calls, not end-to-end wall time including setup and reporting."]}
    if not explicit_plan:
        result["warnings"].append("Legacy manifest lacks an explicit configuration/repetition list; full Cartesian coverage is checked against observed groups and planned counts.")
    eligible = []
    for c in candidates:
        reasons = []
        if c["unique_tasks"] < p["min_tasks"]: reasons.append("too_few_tasks")
        if c["judge_coverage"] < p["min_judge_coverage"]: reasons.append("judge_coverage_below_threshold")
        if c["mean_judge_score"] is None: reasons.append("judge_quality_unknown")
        elif c["mean_judge_score"] < p["min_mean_judge_score"]: reasons.append("judge_score_below_threshold")
        if c["generation_error_rate"] > p["max_generation_error_rate"]: reasons.append("generation_error_rate_above_limit")
        if c["judge_critical_error_rate_among_evaluated"] is not None and c["judge_critical_error_rate_among_evaluated"] > p["max_judge_critical_error_rate"]:
            reasons.append("neural_critical_error_rate_above_limit")
        if p.get("min_judge_pass_rate") is not None:
            if c["judge_pass_rate_among_evaluated"] is None: reasons.append("judge_pass_rate_unknown")
            elif c["judge_pass_rate_among_evaluated"] < p["min_judge_pass_rate"]: reasons.append("judge_pass_rate_below_threshold")
        for policy_key, metric, prefix in (("max_mean_solver_latency_seconds", "mean_solver_latency_seconds", "solver_latency"),
                                          ("max_mean_total_latency_seconds", "mean_total_latency_seconds", "total_latency"),
                                          ("max_mean_cost_usd", "mean_cost_usd", "cost")):
            if p.get(policy_key) is not None:
                if c[metric] is None: reasons.append(prefix + "_unknown")
                elif c[metric] > p[policy_key]: reasons.append(prefix + "_above_limit")
        c["rejection_reasons"] = reasons
        if not reasons: eligible.append(c)
    # One saved run; reject latency-dependent cross-configuration selection if timer scope differs.
    if p.get("max_mean_solver_latency_seconds") is not None or p.get("max_mean_total_latency_seconds") is not None:
        scopes = {c["solver_latency_scope"] for c in eligible}
        if len(scopes) > 1:
            for c in eligible: c["rejection_reasons"].append("incomparable_solver_latency_scopes")
            eligible = []
    result["eligible_count"] = len(eligible)
    if eligible:
        best = max(c["mean_judge_score"] for c in eligible)
        winners = [c for c in eligible if math.isclose(c["mean_judge_score"], best, rel_tol=0, abs_tol=1e-12)]
        if len(winners) == 1:
            result.update(status="selected", selected=winners[0])
        else:
            result.update(status="tied_configurations", tied=winners)
    return result
