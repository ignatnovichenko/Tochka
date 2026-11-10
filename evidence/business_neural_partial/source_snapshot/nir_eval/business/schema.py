"""Strict, dependency-free JSONL contracts for private business-text evaluations."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re

VERSION = 'business-judge-v1'
MAX_LINE_BYTES = 2_000_000
MAX_ROWS = 10_000
ID = re.compile(r'^[A-Za-z0-9_.:-]{1,120}$')


class ValidationError(ValueError):
    """Invalid input. Messages deliberately exclude input values and text."""


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValidationError('Duplicate JSON key')
            result[key] = value
        return result
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValidationError('Non-finite JSON number')
        return result
    def constant(value):
        raise ValidationError('Non-finite JSON number')
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise ValidationError('Invalid JSON') from None


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys():
        raise ValidationError('Missing required object fields')
    if value.keys() - set(required) - set(optional):
        raise ValidationError('Unknown object fields')


def identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValidationError('Identifier must use 1–120 ASCII letters, digits, _, ., :, -')
    return value


def string(value, maximum=200_000, nonempty=True):
    if not isinstance(value, str) or len(value) > maximum or (nonempty and not value.strip()):
        raise ValidationError('Invalid text field or text exceeds size limit')
    return value


def number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValidationError('Numeric field is outside permitted range')
    return value


def unique(items):
    if len(items) != len(set(items)):
        raise ValidationError('Duplicate identifier')


def load_jsonl(path):
    rows = []
    with Path(path).open('rb') as stream:
        while True:
            line = stream.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_LINE_BYTES:
                raise ValidationError('JSONL line exceeds size limit')
            if not line.strip():
                continue
            if len(rows) >= MAX_ROWS:
                raise ValidationError('JSONL row count exceeds limit')
            try:
                rows.append(strict_json(line.decode('utf-8')))
            except (ValidationError, UnicodeDecodeError):
                raise ValidationError(f'Invalid JSONL row {len(rows) + 1}') from None
    if not rows:
        raise ValidationError('JSONL input is empty')
    return rows


def validate_rubric(row):
    fields(row, ('id', 'criteria'), ('critical_errors', 'pass_threshold', 'description'))
    identifier(row['id'])
    if 'description' in row:
        string(row['description'], 10_000)
    criteria = row['criteria']
    if not isinstance(criteria, list) or not 1 <= len(criteria) <= 40:
        raise ValidationError('Rubric must contain 1–40 criteria')
    for item in criteria:
        fields(item, ('id', 'description'), ('weight',))
        identifier(item['id'])
        string(item['description'], 10_000)
        number(item.get('weight', 1), 0.001, 1000)
    unique([item['id'] for item in criteria])
    critical = row.get('critical_errors', [])
    if not isinstance(critical, list) or len(critical) > 40:
        raise ValidationError('Invalid critical errors list')
    for item in critical:
        fields(item, ('id', 'description'))
        identifier(item['id'])
        string(item['description'], 10_000)
    unique([item['id'] for item in critical])
    number(row.get('pass_threshold', .75), 0, 1)
    return row


def validate_check(check):
    fields(check, ('id', 'type'), ('value', 'minimum', 'maximum', 'case_sensitive', 'keys'))
    identifier(check['id'])
    kind = check['type']
    allowed = {
        'contains': {'value', 'case_sensitive'},
        'not_contains': {'value', 'case_sensitive'},
        'word_count': {'minimum', 'maximum'},
        'character_count': {'minimum', 'maximum'},
        'json_object': {'keys'},
    }
    if not isinstance(kind, str) or kind not in allowed or check.keys() - {'id', 'type'} - allowed[kind]:
        raise ValidationError('Unsupported check type or parameters')
    if kind in {'contains', 'not_contains'}:
        string(check.get('value'), 10_000)
        if type(check.get('case_sensitive', False)) is not bool:
            raise ValidationError('case_sensitive must be boolean')
    elif kind in {'word_count', 'character_count'}:
        low, high = check.get('minimum', 0), check.get('maximum', 200_000)
        if type(low) is not int or type(high) is not int:
            raise ValidationError('Count bounds must be integers')
        number(low, 0, 200_000)
        number(high, low, 200_000)
    elif kind == 'json_object':
        keys = check.get('keys', [])
        if not isinstance(keys, list) or len(keys) > 100:
            raise ValidationError('Invalid required JSON keys')
        for key in keys:
            string(key, 200)
        unique(keys)


def validate_case(row):
    fields(row, ('id', 'task', 'context', 'rubric_id'), ('reference', 'checks', 'tags'))
    identifier(row['id'])
    identifier(row['rubric_id'])
    string(row['task'])
    string(row['context'], nonempty=False)
    if 'reference' in row:
        string(row['reference'], nonempty=False)
    checks = row.get('checks', [])
    if not isinstance(checks, list) or len(checks) > 100:
        raise ValidationError('Invalid deterministic checks')
    for check in checks:
        validate_check(check)
    unique([check['id'] for check in checks])
    tags = row.get('tags', [])
    if not isinstance(tags, list) or len(tags) > 40:
        raise ValidationError('Invalid case tags')
    for tag in tags:
        string(tag, 100)
    return row


def validate_response(row):
    fields(row, ('id', 'model', 'configuration'), ('response', 'repetition', 'status', 'provenance'))
    identifier(row['id'])
    string(row['model'], 200)
    identifier(row['configuration'])
    repetition = row.get('repetition', 0)
    if type(repetition) is not int or not 0 <= repetition <= 1000:
        raise ValidationError('Invalid repetition')
    if not isinstance(row.get('status', 'ok'), str) or row.get('status', 'ok') not in {'ok', 'error'}:
        raise ValidationError('Response status must be ok or error')
    if row.get('status', 'ok') == 'ok':
        string(row.get('response'), nonempty=False)
    elif 'response' in row:
        string(row['response'], nonempty=False)
    if 'provenance' in row:
        string(row['provenance'], 200)
    return row


def import_inputs(case_path, rubric_path, response_path=None):
    cases = [validate_case(row) for row in load_jsonl(case_path)]
    rubrics = [validate_rubric(row) for row in load_jsonl(rubric_path)]
    unique([row['id'] for row in cases])
    unique([row['id'] for row in rubrics])
    rubric_map = {row['id']: row for row in rubrics}
    if any(row['rubric_id'] not in rubric_map for row in cases):
        raise ValidationError('Case references unknown rubric')
    responses = [] if response_path is None else [validate_response(row) for row in load_jsonl(response_path)]
    known = {row['id'] for row in cases}
    if any(row['id'] not in known for row in responses):
        raise ValidationError('Response references unknown case')
    keys = [(row['id'], row['model'], row['configuration'], row.get('repetition', 0)) for row in responses]
    unique(keys)
    return cases, rubric_map, responses
