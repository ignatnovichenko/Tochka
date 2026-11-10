"""Transparent aggregates; repetitions are clustered by original task for CIs."""
from __future__ import annotations
from collections import defaultdict
import csv
import html
import itertools
import json
import math
from pathlib import Path
import random
import statistics


def percentile(values, p):
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    low = int(pos)
    return xs[low] + (xs[min(low + 1, len(xs) - 1)] - xs[low]) * (pos - low)


def cluster_interval(task_values, seed=42, draws=2000):
    values = list(task_values)
    if not values:
        return [None, None]
    rng = random.Random(seed)
    samples = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(draws)]
    return [percentile(samples, .025), percentile(samples, .975)]


def wilson_interval(successes, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0, center-margin), min(1, center+margin)]


def summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in records:
        groups[(row["provider"], row["model"], row["config"], row["provenance"])].append(row)
    summaries, task_means = [], {}
    for key, rows in sorted(groups.items()):
        provider, model, config, provenance = key
        tasks = defaultdict(list)
        for row in rows:
            tasks[row["id"]].append(row)
        means = {case_id: statistics.fmean(int(x["correct"]) for x in rs)
                 for case_id, rs in tasks.items()}
        task_means[key] = means
        latencies = [r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None]
        tokens_in, tokens_out = [], []
        for row in rows:
            usage = row.get("token_usage") or {}
            inp = usage.get("prompt_tokens", usage.get("input_tokens"))
            out = usage.get("completion_tokens", usage.get("output_tokens"))
            if isinstance(inp, (int, float)): tokens_in.append(inp)
            if isinstance(out, (int, float)): tokens_out.append(out)
        repeated = [rs for rs in tasks.values() if len(rs) > 1]
        stability = [len({json.dumps(r["parsed_answer"], sort_keys=True) for r in rs}) == 1
                     and all(not r.get("parse_error") and not r.get("error") for r in rs)
                     for rs in repeated]
        exacts = [r["program_exact_match"] for r in rows
                  if r.get("program_exact_match") is not None]
        summary = {
            "provider": provider, "model": model, "config": config, "provenance": provenance,
            "attempts": len(rows), "unique_tasks": len(tasks),
            "correct": sum(r["correct"] for r in rows),
            "answer_accuracy": statistics.fmean(int(r["correct"]) for r in rows),
            "task_mean_accuracy": statistics.fmean(means.values()),
            "task_bootstrap_95_ci": cluster_interval(means.values()),
            "first_repetition_wilson_95_ci": wilson_interval(
                sum(min(rs, key=lambda r: r.get("repetition", 0))["correct"] for rs in tasks.values()), len(tasks)),
            "provider_errors": sum(bool(r.get("error")) for r in rows),
            "parse_errors": sum(bool(r.get("parse_error")) for r in rows),
            "latency_mean_seconds": statistics.fmean(latencies) if latencies else None,
            "latency_median_seconds": statistics.median(latencies) if latencies else None,
            "latency_p95_seconds": percentile(latencies, .95),
            "valid_answer_stability": statistics.fmean(stability) if stability else None,
            "repeated_tasks": len(repeated),
            "program_exact_match_rate": (sum(exacts) / len(rows)) if config == "program" else None,
            "input_tokens_reported": sum(tokens_in) if tokens_in else None,
            "output_tokens_reported": sum(tokens_out) if tokens_out else None,
            "token_usage_coverage": len(tokens_in) / len(rows),
            "cost_usd": None, "ttft_seconds": None,
        }
        summaries.append(summary)
    paired = []
    for ka, kb in itertools.combinations(sorted(groups), 2):
        if ka[3] != kb[3]:
            continue  # Do not compare oracle controls to real models.
        common = sorted(task_means[ka].keys() & task_means[kb].keys())
        if not common:
            continue
        diffs = [task_means[ka][i] - task_means[kb][i] for i in common]
        paired.append({"a": list(ka), "b": list(kb), "paired_tasks": len(common),
                       "mean_accuracy_difference_a_minus_b": statistics.fmean(diffs),
                       "paired_task_bootstrap_95_ci": cluster_interval(diffs)})
    return {"schema_version": "1.0", "groups": summaries, "paired_comparisons": paired,
            "notes": [
                "All attempted cases, including provider and parse errors, are in the accuracy denominator.",
                "Displayed 95% Wilson interval uses the first repetition for each distinct task; it does not count repeats as extra tasks.",
                "Additional 95% percentile bootstrap resamples tasks (2000 draws, seed 42); repeats are not independent tasks.",
                "Small pilots and all-zero/all-one samples give limited, sometimes degenerate intervals, not proof of certainty.",
                "Program exact match is token equality, not official symbolic program accuracy.",
                "No hidden LLM judge is used. Missing cost, quantization, TTFT and usage stay unknown.",
                "Latency across different backends or timer boundaries is not a controlled speed comparison; consult the source manifests.",
                "Comparing models with different sizes, quantization or runtimes compares whole configurations, not an isolated causal factor.",
                "Control and replay runs are not model-quality or model-latency measurements."]}


COLUMNS = ["provider", "model", "config", "provenance", "attempts", "unique_tasks",
           "correct", "answer_accuracy", "task_mean_accuracy", "provider_errors",
           "parse_errors", "latency_mean_seconds", "latency_median_seconds", "latency_p95_seconds",
           "valid_answer_stability", "input_tokens_reported", "output_tokens_reported",
           "token_usage_coverage", "cost_usd", "ttft_seconds"]


def _display(value):
    if value is None: return "unknown"
    if isinstance(value, float): return f"{value:.4f}"
    return str(value)


def write_reports(records: list[dict], output: str | Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["groups"])
    record_columns = ["id", "provider", "model", "config", "repetition", "provenance", "correct",
                      "parsed_answer", "expected_answer", "latency_seconds", "error", "parse_error"]
    with (output / "records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    headers = ["Model", "Config", "Source", "N", "Correct", "Accuracy", "Wilson 95% CI", "p50 s", "Errors"]
    table = []
    for g in summary["groups"]:
        table.append([g["model"], g["config"], g["provenance"], str(g["attempts"]), str(g["correct"]),
                      f'{g["answer_accuracy"]:.1%}',
                      "–".join(f"{x:.1%}" for x in g["first_repetition_wilson_95_ci"]),
                      _display(g["latency_median_seconds"]),
                      str(g["provider_errors"] + g["parse_errors"])])
    md = ["# NIR evaluation report", "", "Only rows marked model-inference represent actual model calls.", "",
          "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    md.extend("| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in table)
    md += ["", "## Interpretation", ""] + ["- " + n for n in summary["notes"]]
    (output / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    html_rows = "".join("<tr>" + "".join("<td>" + html.escape(str(c)) + "</td>" for c in r) + "</tr>" for r in table)
    detail_rows = "".join("<tr>" + "".join("<td>" + html.escape(_display(r.get(c))) + "</td>"
                      for c in ["id", "model", "config", "correct", "parsed_answer", "expected_answer", "parse_error"]) + "</tr>"
                      for r in records)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>NIR evaluation report</title><style>body{font:15px system-ui;background:#f4f6fa;color:#182a3b;margin:40px;line-height:1.5}h1{font-size:32px}table{border-collapse:collapse;background:white;width:100%;margin:20px 0}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce3eb;vertical-align:top}th{background:#17334b;color:white}.scroll{overflow:auto}.notice{padding:16px;border-left:4px solid #d59b38;background:#fff7df}details{margin-top:30px}</style><h1>NIR evaluation report</h1>'
    page += '<p class="notice">Public FinQA pilot. Controls and replay are not model inference. Small samples do not establish general model quality.</p>'
    page += '<div class="scroll"><table><thead><tr>' + "".join('<th>' + h + '</th>' for h in headers) + '</tr></thead><tbody>' + html_rows + '</tbody></table></div>'
    page += '<ul>' + "".join('<li>' + html.escape(n) + '</li>' for n in summary["notes"]) + '</ul>'
    page += '<details><summary>Per-attempt audit table</summary><div class="scroll"><table><tr>' + ''.join('<th>' + x + '</th>' for x in ['Task','Model','Config','Correct','Answer','Gold','Parse error']) + '</tr>' + detail_rows + '</table></div></details></html>'
    (output / "report.html").write_text(page, encoding="utf-8")
    return summary
