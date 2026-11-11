"""Calibration matrix fixtures; no HTTP or external model requests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nir_eval.business.judging import evaluate_candidates
from nir_eval.business.schema import ValidationError, canonical, load_jsonl
from nir_eval.experiment import calibrate
from nir_eval.providers import Generation, ProviderUnavailable


class CalibrationBatchTests(unittest.TestCase):
    def setUp(self):
        self.case = {'id': 'same-case', 'task': 'Return five.', 'context': 'The answer is five.', 'rubric_id': 'r'}
        self.rubric = {'id': 'r', 'criteria': [{'id': 'correct', 'description': 'Answer five.'}],
                       'critical_errors': [{'id': 'wrong', 'description': 'Incorrect answer.'}]}
        self.items = [{'id': 'positive', 'case': self.case, 'rubric': self.rubric, 'candidate': 'five'},
                      {'id': 'negative', 'case': self.case, 'rubric': self.rubric, 'candidate': 'six'}]
        self.config = {'type': 'generation-provider', 'model': 'fixture-judge',
                       'provider': {'type': 'openai-compatible', 'engine': 'promptfoo', 'base_url': 'http://localhost:1/v1'},
                       'parameters': {'max_tokens': 8192}, 'max_calls': 4, 'structured_retries': 1}
        self.records, self.raw, self.rounds = {}, [], []

    def generation(self, job):
        payload = json.loads(job['messages'][1]['content'])
        correct = payload['candidate'] == 'five'
        body = {'case_id': payload['case_id'], 'criteria': [{'id': 'correct', 'score': 4 if correct else 0,
                'rationale': 'Controlled fixture.', 'evidence': []}],
                'critical_errors': [{'id': 'wrong', 'detected': not correct, 'rationale': 'Controlled fixture.',
                    'evidence': [] if correct else [{'source':'candidate','quote':'six'}]}]}
        return Generation(canonical(body), {'total_tokens': 10}, {'cost_usd': .01})

    def execute(self, jobs, max_concurrency, on_result, *, on_started=None):
        self.rounds.append(jobs)
        self.assertEqual(max_concurrency, 3)
        self.assertEqual(len({job['job_id'] for job in jobs}), len(jobs))
        for job in reversed(jobs):
            if on_started: on_started(job['job_id'], '2026-11-11T00:00:00+00:00')
            on_result(job['job_id'], self.generation(job), .2)
        return {'scheduler': 'promptfoo.evaluate', 'completed': len(jobs), 'total_calls': len(jobs),
                'test_count': len(jobs), 'max_active_requests': min(len(jobs), max_concurrency)}

    def run_batch(self, execute=None, items=None):
        with patch('nir_eval.business.judging.evaluate_jobs', side_effect=execute or self.execute):
            return evaluate_candidates(items or self.items, self.config, 3,
                on_record=lambda key, row: self.records.update({key: row}),
                on_raw=lambda key, row: self.raw.append((key, row)))

    def test_same_case_distinct_labels_and_out_of_order_completion(self):
        result = self.run_batch()
        self.assertEqual(result['attempted_calls'], 2)
        self.assertEqual(len(self.rounds), 1)
        self.assertEqual(len(self.rounds[0]), 2)
        self.assertTrue(self.records['positive']['passed'])
        self.assertFalse(self.records['negative']['passed'])
        self.assertEqual(set(key for key, _ in self.raw), {'positive','negative'})
        self.assertTrue(all(row['judge']['latency_seconds'] >= .2 for row in self.records.values()))

    def test_repairs_use_second_promptfoo_round_and_preserve_costs(self):
        def execute(jobs, concurrency, on_result, **kwargs):
            if not self.rounds:
                self.rounds.append(jobs)
                for job in jobs:
                    on_result(job['job_id'], Generation('{}', {}, {'cost_usd':.01}), .1)
                return {'scheduler':'promptfoo.evaluate','completed':len(jobs)}
            return self.execute(jobs, concurrency, on_result, **kwargs)
        result = self.run_batch(execute)
        self.assertEqual(len(self.rounds), 2)
        self.assertEqual(result['attempted_calls'], 4)
        self.assertTrue(all(len(job['messages']) == 3 for job in self.rounds[1]))
        for row in self.records.values():
            self.assertEqual(row['judge']['status'], 'evaluated')
            self.assertEqual(len(row['judge']['attempts']), 2)
            self.assertAlmostEqual(row['judge']['cost_usd'], .02)

    def test_budget_exhaustion_is_unknown_and_does_not_schedule_repair(self):
        self.config['max_calls'] = 2
        def execute(jobs, concurrency, on_result, **kwargs):
            self.rounds.append(jobs)
            for job in jobs: on_result(job['job_id'], Generation('{}', {}, {}), .1)
            return {'scheduler':'promptfoo.evaluate','completed':len(jobs)}
        self.run_batch(execute)
        self.assertEqual(len(self.rounds), 1)
        for row in self.records.values():
            self.assertIsNone(row['score'])
            self.assertEqual(row['judge']['reason_code'], 'judge_call_budget_exhausted')
            self.assertGreaterEqual(row['judge']['latency_seconds'], .1)

    def test_unattempted_local_failure_does_not_consume_call_budget(self):
        self.config['max_calls'] = 2
        def execute(jobs, concurrency, on_result, **kwargs):
            if not self.rounds:
                self.rounds.append(jobs)
                on_result(jobs[0]['job_id'], Generation('{}', {}, {}), .1)
                error = ProviderUnavailable('promptfoo_connection_unavailable')
                error.request_attempted = False
                on_result(jobs[1]['job_id'], error, 0)
                return {'scheduler':'promptfoo.evaluate','completed':len(jobs)}
            return self.execute(jobs, concurrency, on_result, **kwargs)
        result = self.run_batch(execute)
        self.assertEqual(len(self.rounds), 2)
        self.assertEqual(len(self.rounds[1]), 1)
        self.assertEqual(result['attempted_calls'], 2)
        self.assertEqual(result['unattempted_calls'], 1)
        self.assertEqual(self.records['positive']['judge']['status'], 'evaluated')
        self.assertIsNone(self.records['negative']['score'])
        self.assertIsNone(self.records['negative']['judge']['latency_seconds'])

    def test_cooldown_before_any_api_call_has_unknown_latency(self):
        def execute(jobs, concurrency, on_result, **kwargs):
            for job in jobs:
                error = ProviderUnavailable('api_cooldown_active')
                error.request_attempted = False
                error.retry_after_seconds = 60
                on_result(job['job_id'], error, .2)
            return {'scheduler':'promptfoo.evaluate','completed':len(jobs)}
        result = self.run_batch(execute)
        self.assertEqual(result['attempted_calls'], 0)
        for row in self.records.values():
            self.assertIsNone(row['judge']['latency_seconds'])
            self.assertIsNone(row['score'])

    def test_rate_limit_stops_pending_repairs_and_remains_unknown(self):
        def execute(jobs, concurrency, on_result, **kwargs):
            self.rounds.append(jobs)
            on_result(jobs[0]['job_id'], Generation('{}', {}, {}), .1)
            error = ProviderUnavailable('api_http_429'); error.retry_after_seconds = 60
            error.request_attempted = True
            on_result(jobs[1]['job_id'], error, .2)
            return {'scheduler':'promptfoo.evaluate','completed':len(jobs)}
        self.run_batch(execute)
        self.assertEqual(len(self.rounds), 1)
        self.assertEqual(self.records['positive']['judge']['reason_code'], 'judge_api_cooldown_active')
        self.assertEqual(self.records['negative']['judge']['reason_code'], 'judge_api_http_429')
        self.assertTrue(all(row['score'] is None for row in self.records.values()))
        self.assertGreaterEqual(self.records['positive']['judge']['latency_seconds'], .1)
        self.assertGreaterEqual(self.records['negative']['judge']['latency_seconds'], .2)

    def setup_experiment(self, root):
        root.mkdir()
        (root/'experiment.json').write_text(json.dumps({'judge':'judge.json'}))
        (root/'judge.json').write_text(json.dumps(self.config))
        (root/'cases.jsonl').write_text(canonical(self.case)+'\n')
        (root/'rubrics.jsonl').write_text(canonical(self.rubric)+'\n')
        labels = [{'id': item['id'], 'case_id': self.case['id'], 'candidate':item['candidate'],
                   'expected_pass':item['id']=='positive', 'expected_critical':item['id']=='negative'} for item in self.items]
        (root/'labels.jsonl').write_text(''.join(canonical(label)+'\n' for label in labels))

    def test_calibrate_snapshots_streams_and_keeps_summary_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory)/'experiment', Path(directory)/'results'
            self.setup_experiment(root)
            with patch('nir_eval.promptfoo_backend.require_runtime'), \
                 patch('nir_eval.business.judging.evaluate_jobs', side_effect=self.execute), \
                 contextlib.redirect_stdout(io.StringIO()):
                summary = calibrate(root, root/'labels.jsonl', output, max_concurrency=3)
            self.assertEqual(summary['evaluated'], 2)
            self.assertEqual(summary['unknown'], 0)
            self.assertEqual(summary['pass_agreement'], 1)
            self.assertEqual(summary['critical_agreement'], 1)
            self.assertEqual({row['label_id'] for row in load_jsonl(output/'raw.jsonl')}, {'positive','negative'})
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['state'], 'completed')
            self.assertEqual(manifest['scheduler'], 'promptfoo.evaluate')
            self.assertEqual(manifest['max_concurrency'], 3)
            self.assertEqual(manifest['started_calls'], 2)

    def test_interruption_retains_completed_raw_and_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory)/'experiment', Path(directory)/'results'
            self.setup_experiment(root)
            def interrupted(jobs, concurrency, callback, **kwargs):
                callback(jobs[0]['job_id'], self.generation(jobs[0]), .1)
                raise KeyboardInterrupt()
            with patch('nir_eval.promptfoo_backend.require_runtime'), \
                 patch('nir_eval.business.judging.evaluate_jobs', side_effect=interrupted), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    calibrate(root, root/'labels.jsonl', output, max_concurrency=3)
            self.assertEqual(len(load_jsonl(output/'records.jsonl')), 1)
            self.assertEqual(len(load_jsonl(output/'raw.jsonl')), 1)
            self.assertEqual(json.loads((output/'manifest.json').read_text())['state'], 'interrupted')
            self.assertFalse((output/'summary.json').exists())

    def test_bad_concurrency_and_runtime_fail_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory)/'experiment', Path(directory)/'results'
            self.setup_experiment(root)
            with self.assertRaises(ValidationError):
                calibrate(root, root/'labels.jsonl', output, max_concurrency=0)
            with patch('nir_eval.promptfoo_backend.require_runtime', side_effect=ValidationError('fixture missing runtime')):
                with self.assertRaises(ValidationError): calibrate(root, root/'labels.jsonl', output)
            self.assertFalse(output.exists())


if __name__ == '__main__': unittest.main()
