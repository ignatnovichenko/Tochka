"""Promptfoo-scheduled judging of existing candidates with Tochka validation."""
from __future__ import annotations
import time
import math
from datetime import datetime, timedelta, timezone

from .evaluation import evaluate
from .providers import JudgeReply, JudgeUnavailable
from .running import _ReplayJudge, _NeedsJudge, _judge_error
from .schema import ValidationError, digest
from ..providers import ProviderUnavailable


def evaluate_jobs(*args, **kwargs):
    from ..promptfoo_backend import evaluate_jobs as execute
    return execute(*args, **kwargs)


def evaluate_candidates(items, judge_config, max_concurrency=4, *, allow_remote=False,
                        on_record, on_raw=None, on_started=None):
    """Judge each unique item once, allowing only explicitly budgeted JSON repairs.

    Items contain id, case, rubric, candidate. IDs identify candidates rather than
    cases, so positive and negative calibration labels for one case stay separate.
    Promptfoo owns concurrent request execution; Python only validates replies and
    creates the next bounded repair round. Callbacks persist completed evidence.
    """
    if type(max_concurrency) is not int or not 1 <= max_concurrency <= 32:
        raise ValidationError('Concurrency must be an integer from 1 to 32')
    if judge_config.get('type') != 'generation-provider' or judge_config.get('provider', {}).get('type') != 'openai-compatible':
        raise ValidationError('Batch judging requires an API generation provider')
    if not items or len({item['id'] for item in items}) != len(items):
        raise ValidationError('Judging requires nonempty unique candidate identifiers')
    remaining = judge_config.get('max_calls', 100)
    if len(items) > remaining:
        raise ValidationError('Calibration exceeds judge budget')
    states = {item['id']: {'item': item, 'replies': [], 'latencies': [], 'validation_seconds': 0, 'done': False}
              for item in items}
    pending = {}
    repair_ids = set()
    cooldown_until = 0
    rounds = []
    counts = {'attempted_calls': 0, 'unattempted_calls': 0}

    def advance(item_id, *, reserve=True):
        nonlocal remaining
        state = states[item_id]
        item = state['item']
        started = time.monotonic()
        try:
            row = evaluate(item['case'], item['rubric'], item['candidate'],
                           _ReplayJudge(judge_config, state['replies']))
        except _NeedsJudge as suspended:
            state['validation_seconds'] += time.monotonic() - started
            if not reserve:
                return True
            if time.monotonic() < cooldown_until:
                error = JudgeUnavailable('judge_api_cooldown_active')
                error.retry_after_seconds = max(0, cooldown_until - time.monotonic())
                state['replies'].append(error)
                return advance(item_id)
            if remaining <= 0:
                state['replies'].append(JudgeUnavailable('judge_call_budget_exhausted'))
                return advance(item_id)
            remaining -= 1
            job_id = 'calibration-' + digest(item_id)[:32] + '-' + str(len(state['replies']))
            pending[job_id] = (item_id, suspended.messages)
            return {'job_id': job_id, 'provider': judge_config['provider'],
                    'messages': suspended.messages, 'model': judge_config['model'],
                    'parameters': judge_config.get('parameters', {'temperature': 0, 'max_tokens': judge_config.get('max_tokens', 8192)}),
                    'context': {'id': item['case']['id'], 'config': 'calibration', 'repetition': 0},
                    'allow_remote': allow_remote}
        state['validation_seconds'] += time.monotonic() - started
        row['judge']['latency_seconds'] = (round(sum(state['latencies']) + state['validation_seconds'], 6)
                                           if state['latencies'] else None)
        delay = row['judge'].get('retry_after_seconds')
        if type(delay) in (int, float) and math.isfinite(delay) and delay >= 0:
            row['judge']['retry_not_before_utc'] = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
        if state['done']:
            raise ValidationError('Duplicate completed judgement')
        state['done'] = True
        on_record(item_id, row)
        return None

    def completed(job_id, result, latency):
        nonlocal remaining, cooldown_until
        item_id, messages = pending.pop(job_id)
        state = states[item_id]
        if isinstance(result, Exception):
            attempted = getattr(result, 'request_attempted', True)
            if attempted:
                state['latencies'].append(latency)
            counts['attempted_calls' if attempted else 'unattempted_calls'] += 1
            if not attempted:
                remaining += 1
            delay = getattr(result, 'retry_after_seconds', None)
            if isinstance(result, ProviderUnavailable) and type(delay) in (int, float) and math.isfinite(delay) and delay > 0:
                cooldown_until = max(cooldown_until, time.monotonic() + delay)
            state['replies'].append(_judge_error(result))
        else:
            state['latencies'].append(latency)
            counts['attempted_calls'] += 1
            if on_raw:
                on_raw(item_id, {'case_id': state['item']['case']['id'], 'attempt': len(state['replies']) + 1,
                    'prompt_sha256': digest(messages), 'raw_output': result.text,
                    'usage': result.usage, 'metadata': result.metadata})
            state['replies'].append(JudgeReply(result.text, result.usage, result.metadata))
        if advance(item_id, reserve=False):
            repair_ids.add(item_id)

    jobs = [job for item in items if (job := advance(item['id'])) is not None]
    while jobs:
        rounds.append(evaluate_jobs(jobs, max_concurrency, completed, on_started=on_started))
        # Reserve repairs in input order, independent of completion timing.
        next_ids = repair_ids.copy()
        repair_ids.clear()
        jobs = [job for item in items if item['id'] in next_ids
                if (job := advance(item['id'])) is not None]
    if not all(state['done'] for state in states.values()):
        raise ValidationError('Incomplete judge matrix')
    return {'scheduler': 'promptfoo.evaluate', 'max_concurrency': max_concurrency,
            'rounds': rounds, **counts}
