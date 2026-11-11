"""Real pinned native against localhost fixtures; never calls a model.

TOCHKA_PROMPTFOO_DIR may point to an installed runtime containing worker.mjs
and node_modules. Tests skip when the optional runtime is not ready.
"""
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from nir_eval.native_backend import runtime_status, _COOLDOWNS
from nir_eval import __version__
from nir_eval.providers import make_provider, ProviderError, ProviderUnavailable


class APIErrorClassificationTests(unittest.TestCase):
    def test_restriction_codes_are_bounded_and_status_specific(self):
        from nir_eval.api_support import api_error_code
        phrase = b'HOSTED OUTSIDE EUROPE AND THE UNITED STATES'
        self.assertEqual(api_error_code(400, phrase), 'api_http_400_region_restricted')
        self.assertEqual(api_error_code(403, phrase), 'api_http_403')
        self.assertEqual(api_error_code(400, b'x' * 16384 + phrase), 'api_http_400')
        self.assertEqual(api_error_code(400, None), 'api_http_400')
        self.assertEqual(api_error_code(400, 'trains on request data'), 'api_http_400_training_restricted')


class NativeBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = []
        cls.response = b''
        cls.status = 200
        cls.key = 'fixture-key{{literal}}'
        cls.messages = [{'role': 'user', 'content': 'literal {{ untouched }} {% raw %} " \\ \n мир'}]

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                cls.requests.append((self.path, body, dict(self.headers)))
                self.send_response(cls.status)
                self.send_header('Content-Type', 'application/json')
                if cls.status == 429:
                    self.send_header('Retry-After', '120')
                if cls.status == 307:
                    self.send_header('Location', '/redirect-target')
                self.end_headers()
                self.wfile.write(cls.response)

        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        config = {'type': 'openai-compatible', 'base_url': f'http://127.0.0.1:{cls.server.server_port}/v1',
                  'api_key_env': 'TOCHKA_FIXTURE_API_KEY', 'timeout_seconds': 10}
        with patch.dict(os.environ, {'TOCHKA_FIXTURE_API_KEY': cls.key}):
            cls.native = make_provider(dict(config, engine='native'))
        cls.addClassCleanup(cls.server.server_close)
        cls.addClassCleanup(cls.server.shutdown)

    def setUp(self):
        _COOLDOWNS.clear()
        type(self).requests.clear()
        type(self).status = 200
        self.native.blocked_until = 0
        self.native.config.pop('input_price_per_million', None)
        self.native.config.pop('output_price_per_million', None)
        self.set_completion({'prompt_tokens': 13, 'completion_tokens': 7, 'total_tokens': 20})

    @classmethod
    def set_completion(cls, usage):
        cls.response = json.dumps({'model': 'fixture-model', 'choices': [{'message': {'content': 'fixture result'},
                                'finish_reason': 'stop'}], 'usage': usage}).encode()

    def generate(self, provider):
        return provider.generate(self.messages, 'fixture-model', {'temperature': 0}, {})

    def checked_generation(self):
        result = self.generate(self.native)
        self.assertEqual(result.text, 'fixture result')
        self.assertEqual(result.usage, {'prompt_tokens': 13, 'completion_tokens': 7, 'total_tokens': 20})
        self.assertEqual(result.metadata['execution_engine'], 'native')
        self.assertEqual(result.metadata['execution_engine_version'], __version__)
        return result

    def test_raw_reported_estimated_unknown_cost_and_usage(self):
        for mode in ('reported', 'provider-estimate', 'tariff-estimate', 'unknown'):
            with self.subTest(mode=mode):
                usage = {'prompt_tokens': 13, 'completion_tokens': 7, 'total_tokens': 20,
                         'opencode': {'billing_source': 'go', 'list_cost': 0.02,
                                     'tokens': {'input': 13, 'output': 7, 'reasoning': 2, 'cache_read': 3,
                                                'cache_write_5m': 4, 'cache_write_1h': 5, 'secret': 'omit'}}}
                if mode in {'reported', 'provider-estimate'}:
                    usage['cost'] = 0.012
                if mode == 'provider-estimate':
                    usage['opencode']['estimated'] = True
                if mode == 'tariff-estimate':
                    self.native.config.update(input_price_per_million=1, output_price_per_million=2)
                else:
                    self.native.config.pop('input_price_per_million', None)
                    self.native.config.pop('output_price_per_million', None)
                self.set_completion(usage)
                result = self.checked_generation()
                self.assertEqual(result.metadata['cost_usd'], 0.012 if mode == 'reported' else None)
                self.assertEqual(result.metadata['native_tokens'], {'input': 13, 'output': 7, 'reasoning': 2,
                                 'cache_read': 3, 'cache_write_5m': 4, 'cache_write_1h': 5})
                self.assertNotIn('secret', result.metadata['native_tokens'])
                expected_basis = {'reported': 'provider_reported', 'provider-estimate': 'provider_estimate',
                                  'tariff-estimate': 'configured_tariff_estimate', 'unknown': 'unknown'}
                self.assertEqual(result.metadata['cost_basis'], expected_basis[mode])
                if mode == 'provider-estimate':
                    self.assertEqual(result.metadata['estimated_cost_usd'], 0.012)
                if mode == 'tariff-estimate':
                    self.assertAlmostEqual(result.metadata['estimated_cost_usd'], 27 / 1_000_000)
                if mode == 'unknown':
                    self.assertNotIn('estimated_cost_usd', result.metadata)

    def test_literal_messages_and_headers_no_cache(self):
        self.generate(self.native)
        self.generate(self.native)
        self.assertEqual(len(self.requests), 2)
        for path, body, headers in self.requests:
            self.assertEqual(path, '/v1/chat/completions')
            self.assertEqual(body, {'model': 'fixture-model', 'messages': self.messages,
                                    'temperature': 0, 'stream': False})
            headers = {k.lower(): v for k, v in headers.items()}
            self.assertEqual(headers['authorization'], 'Bearer ' + self.key)
            self.assertTrue(headers['user-agent'].startswith('Tochka/'))
            self.assertEqual(headers['x-opencode-session'], self.native.session_id)

    def test_health_uses_no_http(self):
        result = runtime_status()
        self.assertTrue(result['ready'])
        self.assertEqual(result['engine'], 'native')
        self.assertEqual(self.requests, [])

    def test_429_no_retry_and_local_cooldown(self):
        type(self).status = 429
        type(self).response = b'{"error":{"message":"private provider diagnostic"}}'
        with self.assertRaisesRegex(ProviderUnavailable, '^api_http_429$') as first:
            self.generate(self.native)
        self.assertEqual(first.exception.retry_after_seconds, 120)
        with self.assertRaisesRegex(ProviderUnavailable, '^api_cooldown_active$'):
            self.generate(self.native)
        self.assertEqual(len(self.requests), 1)

    def test_redirect_never_follows_location(self):
        type(self).status = 307
        type(self).response = b'{}'
        with self.assertRaises(ProviderError):
            self.generate(self.native)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0][0], '/v1/chat/completions')

    def test_privacy_restrictions_use_safe_codes(self):
        type(self).status = 400
        samples = (
            ('This model is hosted outside Europe and the United States', 'api_http_400_region_restricted'),
            ('This provider trains on request data', 'api_http_400_training_restricted'),
            ('Unknown diagnostic', 'api_http_400'),
        )
        for message, expected in samples:
            with self.subTest(expected=expected):
                type(self).response = json.dumps({'error': {'message': message + ' private-key-do-not-save'}}).encode()
                with self.assertRaises(ProviderError) as caught:
                    self.generate(self.native)
                self.assertEqual(str(caught.exception), expected)
                self.assertNotIn('private-key', str(caught.exception))

    def test_malformed_and_duplicate_json_are_rejected(self):
        for raw in (b'not json', b'{"choices":[]}',
                    b'{"choices":[{"message":{"content":"ok"}}],"usage":{"cost":0,"cost":99}}'):
            with self.subTest(raw=raw):
                type(self).response = raw
                with self.assertRaises(ProviderError):
                    self.generate(self.native)


if __name__ == '__main__':
    unittest.main()
