"""Deterministic checks, never arbitrary Python, shell, or user-supplied regex."""
import re
from .schema import strict_json, ValidationError


def run_checks(response, checks):
    results = []
    for check in checks:
        kind = check['type']
        observed = None
        if kind in {'contains', 'not_contains'}:
            text, needle = response, check['value']
            if not check.get('case_sensitive', False):
                text, needle = text.casefold(), needle.casefold()
            found = needle in text
            passed = found if kind == 'contains' else not found
            observed = {'found': found}
        elif kind in {'word_count', 'character_count'}:
            count = len(re.findall(r'\S+', response)) if kind == 'word_count' else len(response)
            passed = check.get('minimum', 0) <= count <= check.get('maximum', 200_000)
            observed = {'count': count}
        elif kind == 'json_object':
            try:
                data = strict_json(response)
                passed = isinstance(data, dict) and set(check.get('keys', [])) <= data.keys()
            except ValidationError:
                passed = False
            observed = {'valid_object_with_required_keys': passed}
        else:
            raise ValidationError('Unsupported deterministic check')
        results.append({'id': check['id'], 'type': kind, 'passed': passed, 'observed': observed})
    return results
