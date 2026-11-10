"""Dataset boundaries: providers receive public context, never reference fields."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    pre_text: list[str]
    table: list[list[str]]
    post_text: list[str]
    expected_answer: float | str
    expected_program: str

    def public_input(self) -> dict:
        return {"id": self.id, "question": self.question,
                "pre_text": self.pre_text, "table": self.table,
                "post_text": self.post_text}


def load_finqa(path: str | Path) -> list[Case]:
    path = Path(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("FinQA dataset must be a nonempty JSON array")
    cases, ids = [], set()
    for row in rows:
        qa = row["qa"]
        case = Case(str(row["id"]), qa["question"], row["pre_text"],
                    row["table"], row["post_text"], qa["exe_ans"], qa["program"])
        if not case.id or case.id in ids:
            raise ValueError(f"Empty or duplicate case ID: {case.id}")
        if not isinstance(case.question, str) or not case.question.strip():
            raise ValueError(f"Invalid question: {case.id}")
        if not all(isinstance(x, str) for x in case.pre_text + case.post_text):
            raise ValueError(f"Invalid text context: {case.id}")
        if not isinstance(case.table, list) or not all(
                isinstance(r, list) and all(isinstance(c, str) for c in r)
                for r in case.table):
            raise ValueError(f"Invalid table: {case.id}")
        ids.add(case.id)
        cases.append(case)
    return cases


def select_cases(cases: list[Case], limit: int | None = None, seed: int = 42,
                 ids: list[str] | None = None) -> list[Case]:
    if ids is not None:
        if len(ids) != len(set(ids)):
            raise ValueError("Requested case IDs contain duplicates")
        by_id = {c.id: c for c in cases}
        missing = set(ids) - by_id.keys()
        if missing:
            raise ValueError(f"Unknown case IDs: {sorted(missing)}")
        selected = [by_id[i] for i in ids]
    else:
        selected = sorted(cases, key=lambda c: hashlib.sha256(
            f"{seed}:{c.id}".encode()).hexdigest())
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No cases selected")
    return selected


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
