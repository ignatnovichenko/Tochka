"""Strict answer parsing and a bounded FinQA DSL interpreter; never eval/exec."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import re

OPS = {"add", "subtract", "multiply", "divide", "exp", "greater",
       "table_max", "table_min", "table_sum", "table_average"}
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\Z")
STEP = re.compile(r"([a-z_]+)\(\s*([^(),]+?)\s*,\s*([^(),]+?)\s*\)")
MAX_OUTPUT_CHARS = 100_000


@dataclass
class Score:
    parsed_answer: float | str | None = None
    correct: bool = False
    parse_error: str | None = None
    predicted_program: str | None = None
    program_exact_match: bool | None = None


def normalize_answer(value) -> float | str:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid financial answers")
    if isinstance(value, (float, int)):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in {"yes", "no"}:
            return text
        text = text.replace(",", "")
        pct = text.endswith("%")
        if pct:
            text = text[:-1].strip()
        if not NUMBER.fullmatch(text):
            raise ValueError("Expected one numeric value or yes/no")
        result = float(text) / (100 if pct else 1)
    else:
        raise ValueError("Expected one numeric value or yes/no")
    if not math.isfinite(result):
        raise ValueError("Non-finite answer")
    return result


def answers_equal(predicted, expected) -> bool:
    try:
        a, b = normalize_answer(predicted), normalize_answer(expected)
    except (ValueError, TypeError, OverflowError):
        return False
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    # Official FinQA execution evaluator rounds the executed result, not the gold.
    return round(a, 5) == b


def parse_object(text: str) -> dict:
    if not isinstance(text, str) or len(text) > MAX_OUTPUT_CHARS:
        raise ValueError("Output missing or exceeds maximum length")
    value = text.strip()
    if value.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", value)
        if not match:
            raise ValueError("Malformed JSON fence")
        value = match[1]
    try:
        def reject_constant(x):
            raise ValueError(f"Non-standard JSON constant: {x}")
        def unique_keys(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON field: {key}")
                result[key] = item
            return result
        obj = json.loads(value, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    except json.JSONDecodeError as exc:
        raise ValueError("Response is not a single JSON object") from exc
    if not isinstance(obj, dict):
        raise ValueError("Response is not a JSON object")
    return obj


def parse_program(program: str) -> list[tuple[str, str, str]]:
    if not isinstance(program, str) or not program.strip() or len(program) > 10_000:
        raise ValueError("Missing or oversized program")
    steps, cursor = [], 0
    for match in STEP.finditer(program):
        gap = program[cursor:match.start()].strip()
        if gap not in ({""} if not steps else {","}):
            raise ValueError("Malformed program separator")
        op, left, right = match.groups()
        if op not in OPS:
            raise ValueError(f"Unknown operation: {op}")
        steps.append((op, left.strip(), right.strip()))
        cursor = match.end()
        if len(steps) > 64:
            raise ValueError("Program exceeds 64 steps")
    if not steps or program[cursor:].strip():
        raise ValueError("Malformed program")
    return steps


def program_tokens(program: str) -> list[str]:
    result = []
    for op, left, right in parse_program(program):
        result.extend([op + "(", left, right, ")"])
    return result + ["EOF"]


def execute_program(program: str, table: list[list[str]]) -> float | str:
    results = []
    rows = {row[0]: row[1:] for row in table if row}

    def resolve(token):
        if re.fullmatch(r"#\d+", token):
            index = int(token[1:])
            if index >= len(results):
                raise ValueError("Invalid forward result reference")
            result = results[index]
        else:
            if token.startswith("const_"):
                token = "-1" if token == "const_m1" else token[6:]
            result = normalize_answer(token)
        if isinstance(result, str):
            raise ValueError("Non-numeric arithmetic operand")
        return result

    for op, left, right in parse_program(program):
        if op.startswith("table_"):
            if left not in rows or right != "none":
                raise ValueError("Unknown table row or invalid table operand")
            nums = [normalize_answer(x.replace("$", "").split("(")[0].strip())
                    for x in rows[left]]
            if not nums or not all(isinstance(n, float) for n in nums):
                raise ValueError("Non-numeric table row")
            value = {"table_max": max, "table_min": min, "table_sum": sum,
                     "table_average": lambda ns: sum(ns) / len(ns)}[op](nums)
        else:
            a, b = resolve(left), resolve(right)
            if op == "add": value = a + b
            elif op == "subtract": value = a - b
            elif op == "multiply": value = a * b
            elif op == "divide": value = a / b
            elif op == "greater": value = "yes" if a > b else "no"
            elif op == "exp":
                if abs(b) > 1000:
                    raise ValueError("Exponent exceeds safe execution bound")
                value = a ** b
        if not isinstance(value, str) and (
                isinstance(value, complex) or not math.isfinite(value)):
            raise ValueError("Non-finite program result")
        results.append(value)
    final = results[-1]
    return round(final, 5) if not isinstance(final, str) else final


def score_output(text: str, protocol: str, expected, expected_program: str,
                 table: list[list[str]]) -> Score:
    result = Score()
    try:
        obj = parse_object(text)
        if protocol == "program":
            result.predicted_program = obj.get("program")
            result.parsed_answer = execute_program(result.predicted_program, table)
            result.program_exact_match = (program_tokens(result.predicted_program) ==
                                          program_tokens(expected_program))
        elif protocol in {"direct", "cot"}:
            result.parsed_answer = normalize_answer(obj.get("answer"))
        else:
            raise ValueError("Unknown scoring protocol")
        result.correct = answers_equal(result.parsed_answer, expected)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        result.parse_error = str(exc)
    return result
