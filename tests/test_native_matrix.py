"""Actual Native evaluator contracts against a local HTTP fixture, no models."""
import copy
import json
import os
from pathlib import Path
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from nir_eval import native_backend as backend
from nir_eval.business.schema import ValidationError
from nir_eval.providers import Generation, ProviderError, ProviderUnavailable


class NativeMatrixTests(unittest.TestCase):
    def setUp(self):
        backend._COOLDOWNS.clear()
        self.addCleanup(backend._COOLDOWNS.clear)
        self.requests = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.secret = 'matrix-fixture-secret{{literal}}'
        self.completed = []
        self.started = []
        self.workers = []
        self.processes = []
        self.scratch_paths = []
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with test.lock:
                    test.active += 1
                    test.peak = max(test.peak, test.active)
                    test.requests.append({'path': self.path, 'body': payload,
                                          'authorization': self.headers.get('Authorization')})
                try:
                    time.sleep(payload.get('delay_seconds', .08))
                    status = payload.get('fixture_status', 200)
                    usage = {'prompt_tokens': 13, 'completion_tokens': 7, 'total_tokens': 20}
                    if payload.get('cost_mode') == 'known':
                        usage.update(cost=0, opencode={'billing_source':'go', 'tokens':{'input':13,'output':7}})
                    elif payload.get('cost_mode') == 'estimate':
                        usage.update(cost=.003, opencode={'estimated':True})
                    content = 'answer:' + payload['model'] + ':' + payload.get('marker', '')
                    if payload.get('echo_key'):
                        content += ':' + self.headers.get('Authorization', '')[7:]
                    body = json.dumps({'model': payload['model'], 'choices': [
                        {'message': {'content': content}, 'finish_reason': 'stop'}], 'usage':usage}).encode()
                    if payload.get('fixture_large_whitespace'):
                        # Valid <1 MiB HTTP JSON expands beyond the NDJSON wire limit.
                        body = b'\t' * 1_020_000 + body
                    self.send_response(status)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    if status == 429:
                        self.send_header('Retry-After', '120')
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                finally:
                    with test.lock:
                        test.active -= 1

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.config = {'type':'openai-compatible', 'engine':'native',
                       'base_url':f'http://127.0.0.1:{self.server.server_port}/v1',
                       'api_key_env':'TOCHKA_MATRIX_FIXTURE_KEY','timeout_seconds':5}
        env = patch.dict(os.environ, {'TOCHKA_MATRIX_FIXTURE_KEY':self.secret})
        env.start(); self.addCleanup(env.stop)
    def job(self, job_id, model='fixture-a', *, timeout=5, **parameters):
        return {'job_id':job_id, 'model':model,
                'provider':dict(self.config, timeout_seconds=timeout),
                'parameters':dict(marker=job_id, **parameters),
                'messages':[{'role':'user','content':job_id+' {{api_key}} {{7*7}} {% raw %} " \\ \n'}],
                'allow_remote':False}

    def evaluate(self, jobs, concurrency=2, callback=None):
        return backend.evaluate_jobs(jobs, concurrency,
            callback or (lambda job_id, answer, latency:self.completed.append((job_id, answer, latency))),
            on_started=lambda job_id, started:self.started.append((job_id, started)))

    def assert_workers_closed(self):
        # Native dispatch has no worker process, scratch database or persisted keys.
        self.assertEqual(self.processes, [])
        self.assertEqual(self.scratch_paths, [])

    def test_filtered_matrix_identity_params_repetitions_and_out_of_order(self):
        jobs = [self.job('a-rep0',delay_seconds=.4,temperature=0,cost_mode='known'),
                self.job('b-rep0','fixture-b',delay_seconds=.04,temperature=.7,cost_mode='known'),
                self.job('a-rep1',delay_seconds=.05,temperature=0,cost_mode='known'),
                self.job('b-rep1','fixture-b',delay_seconds=.05,temperature=.7,cost_mode='known')]
        result = self.evaluate(jobs)
        self.assertEqual(result['scheduler'],'native.threadpool')
        self.assertEqual(result['test_count'],4)
        self.assertEqual(result['completed'],4)
        self.assertEqual(result['total_calls'],4)
        self.assertEqual(result['max_active_requests'],2)
        self.assertEqual(self.peak,2)
        self.assertEqual(len(self.requests),4, 'No unintended test/provider cross product')
        self.assertEqual(self.completed[0][0],'b-rep0')
        by_id = {job_id:answer for job_id,answer,_ in self.completed}
        received = {r['body']['marker']:r for r in self.requests}
        self.assertEqual(set(by_id),{j['job_id'] for j in jobs})
        for job in jobs:
            sent = received[job['job_id']]
            self.assertEqual(sent['path'],'/v1/chat/completions')
            self.assertEqual(sent['authorization'],'Bearer '+self.secret)
            self.assertEqual(sent['body']['messages'],job['messages'])
            self.assertEqual(sent['body']['temperature'],job['parameters']['temperature'])
            self.assertEqual(sent['body']['model'],job['model'])
            answer = by_id[job['job_id']]
            self.assertIsInstance(answer,Generation)
            self.assertEqual(answer.text,'answer:'+job['model']+':'+job['job_id'])
            self.assertEqual(answer.metadata['execution_scheduler'],'native.threadpool')
            self.assertTrue(answer.metadata['request_attempted'])
        self.assertEqual({x[0] for x in self.started},set(by_id))
        self.assertTrue(all(isinstance(t,str) and 'T' in t for _,t in self.started))
        self.assert_workers_closed()

    def test_concurrency_one_is_serial_and_identical_jobs_are_not_cached(self):
        jobs=[self.job(str(i)) for i in range(3)]
        # The model input really is identical; job identities still remain unique.
        for job in jobs:
            job['messages']=[{'role':'user','content':'same prompt'}]
            job['parameters']={'temperature':0,'delay_seconds':.05}
        result=self.evaluate(jobs,1)
        self.assertEqual(self.peak,1)
        self.assertEqual(result['max_active_requests'],1)
        self.assertEqual(result['total_calls'],3)
        self.assertEqual(len(self.requests),3)
        self.assertEqual([x[0] for x in self.completed],['0','1','2'])
        self.assert_workers_closed()

    def test_429_only_one_request_then_queued_jobs_skip_without_retry(self):
        jobs=[self.job('limited',fixture_status=429)]+[self.job(str(i)) for i in range(3)]
        result=self.evaluate(jobs,1)
        self.assertEqual(len(self.requests),1)
        self.assertEqual(result['total_calls'],1)
        self.assertEqual(result['completed'],4)
        self.assertEqual(len(self.started),1)
        first=self.completed[0][1]
        self.assertIsInstance(first,ProviderUnavailable)
        self.assertEqual(str(first),'api_http_429')
        self.assertTrue(first.request_attempted)
        self.assertEqual(first.retry_after_seconds,120)
        for _,answer,_ in self.completed[1:]:
            self.assertEqual(str(answer),'api_cooldown_active')
            self.assertFalse(answer.request_attempted)
            self.assertGreater(answer.retry_after_seconds,0)
        self.assert_workers_closed()

    def test_concurrent_credentials_are_isolated_for_same_endpoint(self):
        other_secret = 'different-matrix-secret{{literal}}'
        first = self.job('first-key', echo_key=True)
        second = self.job('second-key', 'fixture-b', echo_key=True)
        second['provider']['api_key_env'] = 'TOCHKA_MATRIX_OTHER_KEY'
        with patch.dict(os.environ, {'TOCHKA_MATRIX_OTHER_KEY': other_secret}):
            self.evaluate([first, second], 2)
        by_id = {r['body']['marker']:r['authorization'] for r in self.requests}
        self.assertEqual(by_id, {'first-key':'Bearer '+self.secret,
                                 'second-key':'Bearer '+other_secret})
        self.assertEqual(self.peak, 2)
        for _, answer, _ in self.completed:
            self.assertIsInstance(answer, Generation)
            self.assertIn('[REDACTED_CREDENTIAL]', answer.text)
            self.assertNotIn(self.secret, answer.text)
            self.assertNotIn(other_secret, answer.text)
        self.assert_workers_closed()

    def test_429_cooldown_survives_a_new_evaluator_batch(self):
        first = self.evaluate([self.job('limited', fixture_status=429)], 1)
        self.assertEqual(first['total_calls'], 1)
        self.assertEqual(len(self.requests), 1)
        self.completed.clear(); self.started.clear()
        second = self.evaluate([self.job('next-batch', 'another-model')], 2)
        self.assertEqual(second['total_calls'], 0)
        self.assertEqual(second['max_active_requests'], 0)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.started, [])
        answer = self.completed[0][1]
        self.assertEqual(str(answer), 'api_cooldown_active')
        self.assertFalse(answer.request_attempted)
        self.assertGreater(answer.retry_after_seconds, 0)
        self.assert_workers_closed()

    def test_timeout_is_isolated_from_another_concurrent_request(self):
        result=self.evaluate([self.job('timeout',timeout=.4,delay_seconds=1.0),
                              self.job('success','fixture-b',timeout=3,delay_seconds=.1)])
        by_id={job_id:answer for job_id,answer,_ in self.completed}
        self.assertIsInstance(by_id['timeout'],ProviderUnavailable)
        self.assertTrue(by_id['timeout'].request_attempted)
        self.assertIsInstance(by_id['success'],Generation)
        self.assertEqual(result['total_calls'],2)
        self.assertEqual(len(self.requests),2)
        self.assert_workers_closed()

    def test_500_is_not_retried_and_does_not_cancel_other_provider(self):
        result=self.evaluate([self.job('bad',fixture_status=500),self.job('good','fixture-b')])
        answers={job_id:answer for job_id,answer,_ in self.completed}
        self.assertEqual(str(answers['bad']),'api_http_500')
        self.assertTrue(answers['bad'].request_attempted)
        self.assertIsInstance(answers['good'],Generation)
        self.assertEqual(len(self.requests),2)
        self.assertEqual(result['total_calls'],2)
        self.assert_workers_closed()

    def test_raw_costs_unknowns_and_key_redaction_not_persisted_by_native(self):
        snapshots=[]
        def record(job_id, answer, latency):
            self.completed.append((job_id,answer,latency))
            for path in self.scratch_paths:
                for file in path.rglob('*'):
                    if file.is_file():
                        snapshots.append(file.name)
                        self.assertNotIn(self.secret.encode(),file.read_bytes())
                        self.assertNotIn('results',file.name.lower())
                        self.assertFalse(file.suffix in {'.db','.sqlite','.sqlite3'})
        self.evaluate([self.job('known',cost_mode='known',echo_key=True),
                       self.job('unknown',echo_key=True),self.job('estimate',cost_mode='estimate')],callback=record)
        answers={job_id:answer for job_id,answer,_ in self.completed}
        self.assertEqual(answers['known'].metadata['cost_usd'],0)
        self.assertEqual(answers['known'].metadata['billing_source'],'go')
        self.assertEqual(answers['known'].metadata['native_tokens'],{'input':13,'output':7})
        self.assertIsNone(answers['unknown'].metadata['cost_usd'])
        self.assertEqual(answers['unknown'].metadata['cost_basis'],'unknown')
        self.assertIsNone(answers['estimate'].metadata['cost_usd'])
        self.assertEqual(answers['estimate'].metadata['estimated_cost_usd'],.003)
        for answer in answers.values():
            self.assertEqual(answer.usage,{'prompt_tokens':13,'completion_tokens':7,'total_tokens':20})
            self.assertNotIn(self.secret,answer.text)
            self.assertNotIn(self.secret,json.dumps(answer.metadata))
        self.assertIn('[REDACTED_CREDENTIAL]',answers['known'].text)
        self.assert_workers_closed()

    def test_callback_interruption_preserves_partial_completions_and_stops_worker(self):
        def record(job_id,answer,latency):
            self.completed.append((job_id,answer,latency))
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.evaluate([self.job('first',delay_seconds=.03),
                           self.job('inflight',delay_seconds=.8),self.job('queued',delay_seconds=.8)],callback=record)
        self.assertEqual(len(self.completed),1)
        self.assertEqual(self.completed[0][0],'first')
        self.assertIsInstance(self.completed[0][1],Generation)
        time.sleep(.9)
        self.assertLessEqual(len(self.requests),2, 'Cancelled queued work must never reach HTTP')
        self.assertEqual(len(self.completed),1, 'Cancelled results must not mutate durable evidence')
        self.assert_workers_closed()

    def test_oversized_wire_result_keeps_attempted_event_contract(self):
        result = self.evaluate([self.job('large-wire', fixture_large_whitespace=True)], 1)
        self.assertEqual(result['completed'], 1)
        self.assertEqual(result['total_calls'], 1)
        self.assertEqual(len(self.completed), 1)
        self.assertEqual(len(self.started), 1)
        self.assertIsInstance(self.completed[0][1], ProviderError)
        self.assertTrue(self.completed[0][1].request_attempted)
        self.assert_workers_closed()

    def test_invalid_matrix_rejected_before_any_request(self):
        good=self.job('good')
        duplicate=copy.deepcopy(good)
        for jobs,concurrency in [([good,duplicate],2),([good],0),([good],33),
                                 ([good,dict(self.job('bad'),parameters={'stream':True})],2)]:
            with self.subTest(concurrency=concurrency,jobs=len(jobs)):
                with self.assertRaises(ValidationError):self.evaluate(jobs,concurrency)
                self.assertEqual(self.requests,[])
        self.assertEqual(self.processes,[])


if __name__ == '__main__':
    unittest.main()
