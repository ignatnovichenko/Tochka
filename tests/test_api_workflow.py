"""End-to-end tests against a local HTTP fixture, never an external model.

The fixture deliberately returns controlled scores. These are software tests and
must not be included in product model-quality statistics.
"""
import contextlib
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from nir_eval.business.schema import load_jsonl


def tree_hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


class APIWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.experiment = self.root / 'experiment'
        self.run = self.root / 'run'
        self.requests = []
        self.fail_once_model = None
        self.failure_sent = False
        self.omit_cost = False
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get('Content-Length', '0'))
                payload = json.loads(self.rfile.read(length))
                test.requests.append({'path': self.path, 'payload': payload,
                                      'authorization': self.headers.get('Authorization')})
                if payload['model'] == test.fail_once_model and not test.failure_sent:
                    test.failure_sent = True
                    self.send_response(429)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Retry-After', '0')
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"fixture rate limit"}}')
                    return
                if payload['model'] == 'fixture-judge':
                    source = json.loads(payload['messages'][1]['content'])
                    rubric = json.loads(payload['messages'][0]['content'].split('TRUSTED RUBRIC JSON:\n', 1)[1])['rubric']
                    score = 4 if 'fixture-a' in source['candidate'] else 3
                    text = json.dumps({'case_id': source['case_id'],
                        'criteria': [{'id': item['id'], 'score': score,
                                      'rationale': 'Controlled local test fixture; not a real judgement.',
                                      'evidence': [{'source': 'candidate', 'quote': 'Fixture answer'}]}
                                     for item in rubric['criteria']],
                        'critical_errors': [{'id': item['id'], 'detected': False,
                                             'rationale': 'Controlled local test fixture.', 'evidence': []}
                                            for item in rubric.get('critical_errors', [])]})
                    cost = .003
                else:
                    text = 'Fixture answer from ' + payload['model'] + '.'
                    cost = .002
                usage = {'prompt_tokens': 20, 'completion_tokens': 5, 'total_tokens': 25}
                if not test.omit_cost:
                    usage['cost'] = cost
                body = json.dumps({'model': payload['model'], 'choices': [
                    {'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
                    'usage': usage}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f'http://127.0.0.1:{self.server.server_port}/v1'
        self.key = 'fixture-secret-never-persist'
        self.env = patch.dict(os.environ, {'TOCHKA_TEST_KEY': self.key})
        self.env.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.env.stop()
        self.temp.cleanup()

    def cli(self, *args):
        from nir_eval.experiment import main
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = main(list(args))
        self.assertEqual(code, 0, output.getvalue())
        return output.getvalue()

    def prepare(self, max_cost=None):
        self.cli('init', str(self.experiment), '--demo')
        for name, model in [('alpha', 'fixture-a'), ('beta', 'fixture-b')]:
            self.cli('solver-add', str(self.experiment), '--provider', 'api',
                     '--base-url', self.endpoint, '--api-key-env', 'TOCHKA_TEST_KEY',
                     '--model', model, '--name', name, '--temperature', '0', '--max-tokens', '128')
        self.cli('judge-set', str(self.experiment), '--provider', 'api',
                 '--base-url', self.endpoint, '--api-key-env', 'TOCHKA_TEST_KEY',
                 '--model', 'fixture-judge', '--temperature', '0', '--max-tokens', '1024')
        policy = ['policy-set', str(self.experiment), '--min-quality', '0.8', '--min-coverage', '1']
        if max_cost is not None:
            policy += ['--max-cost', str(max_cost)]
        self.cli(*policy)
        self.assertEqual(self.requests, [], 'Setup must not query a model')

    def read_comparison(self, directory=None):
        return json.loads(((directory or self.run) / 'comparison.json').read_text())

    def test_full_api_workflow_preserves_usage_and_cost(self):
        self.prepare(max_cost=.01)
        self.cli('run', str(self.experiment), '--output', str(self.run))
        comparison = self.read_comparison()
        self.assertEqual(comparison['status'], 'selected')
        self.assertEqual(comparison['selected']['model'], 'fixture-a')
        self.assertEqual(len(comparison['candidates']), 2)
        self.assertEqual(len(self.requests), 16)
        self.assertTrue(all(r['path'] == '/v1/chat/completions' for r in self.requests))
        self.assertTrue(all(r['authorization'] == 'Bearer ' + self.key for r in self.requests))
        for candidate in comparison['candidates']:
            self.assertEqual(candidate['judge_coverage'], 1)
            self.assertAlmostEqual(candidate['mean_cost_usd'], .005)
            records = load_jsonl(self.run / candidate['run'] / 'solver_records.jsonl')
            self.assertTrue(all(r['token_usage']['total_tokens'] == 25 for r in records))
            self.assertTrue(all(r['provider_metadata']['cost_usd'] == .002 for r in records))
        self.assertTrue((self.run / 'report.html').exists())
        self.assertTrue((self.run / 'comparison.csv').exists())
        for path in self.root.rglob('*'):
            if path.is_file():
                self.assertNotIn(self.key.encode(), path.read_bytes(), str(path))

    def test_unknown_cost_is_not_zero_and_excludes_candidate(self):
        self.omit_cost = True
        self.prepare(max_cost=0)
        self.cli('run', str(self.experiment), '--output', str(self.run))
        result = self.read_comparison()
        self.assertEqual(result['status'], 'no_eligible_configuration')
        for candidate in result['candidates']:
            self.assertIsNone(candidate['mean_cost_usd'])
            self.assertIn('cost_unknown', candidate['rejection_reasons'])

    def test_reselect_has_no_http_calls_and_leaves_source_unchanged(self):
        self.prepare()
        self.cli('run', str(self.experiment), '--output', str(self.run))
        original = tree_hashes(self.run)
        request_count = len(self.requests)
        output = self.root / 'selection-strict'
        self.cli('reselect', str(self.run), '--output', str(output), '--max-latency', '0')
        self.assertEqual(len(self.requests), request_count)
        self.assertEqual(tree_hashes(self.run), original)
        self.assertEqual(self.read_comparison(output)['status'], 'no_eligible_configuration')
        self.assertTrue((output / 'report.html').exists())

    def test_rate_limit_resume_only_missing_work_and_keeps_source(self):
        self.prepare()
        self.fail_once_model = 'fixture-b'
        self.cli('run', str(self.experiment), '--output', str(self.run))
        self.assertTrue(self.failure_sent)
        source_hashes = tree_hashes(self.run)
        original = self.read_comparison()
        missing_solver = missing_judge = 0
        preserved = {}
        for candidate in original['candidates']:
            group = self.run / candidate['run']
            solver = load_jsonl(group / 'solver_records.jsonl')
            records = load_jsonl(group / 'records.jsonl')
            missing_solver += sum(r.get('error') is not None for r in solver)
            missing_judge += sum(r['judge']['status'] != 'evaluated' for r in records)
            for r in records:
                if r['judge']['status'] == 'evaluated':
                    preserved[(candidate['run'], r['id'])] = r
        self.assertGreater(missing_solver, 0)
        request_count = len(self.requests)
        output = self.root / 'resumed'
        # The provider intentionally clamps Retry-After:0 to a real one-second
        # deadline. Honor the saved deadline rather than disabling the guard.
        from nir_eval.resume import resume_plan
        deadline = resume_plan(self.run)['retry_not_before_utc']
        if deadline:
            delay = max(0, (datetime.fromisoformat(deadline) - datetime.now(timezone.utc)).total_seconds()) + .02
            self.assertLessEqual(delay, 2, 'The local fixture must never impose a long wait')
            time.sleep(delay)
        self.cli('resume', str(self.run), '--output', str(output))
        self.assertEqual(tree_hashes(self.run), source_hashes)
        self.assertEqual(len(self.requests) - request_count, missing_solver + missing_judge)
        updated = self.read_comparison(output)
        self.assertTrue(all(c['judge_coverage'] == 1 for c in updated['candidates']))
        self.assertEqual(updated['selected']['model'], 'fixture-a')
        for candidate in updated['candidates']:
            for r in load_jsonl(output / candidate['run'] / 'records.jsonl'):
                key = candidate['run'], r['id']
                if key in preserved:
                    self.assertEqual(r['score'], preserved[key]['score'])
                    self.assertEqual(r['judge']['output_sha256'], preserved[key]['judge']['output_sha256'])


if __name__ == '__main__':
    unittest.main()
