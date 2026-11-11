"""API boundary tests: mocked transport only; never contact external models."""
import io
import json
import os
import time
import unittest
from urllib.error import HTTPError
from unittest.mock import Mock, patch
from pathlib import Path

from nir_eval.api_support import response_metadata, validate_api_config
from nir_eval.providers import OpenAICompatibleProvider, ProviderError, ProviderUnavailable, make_provider
from nir_eval.business.providers import make_judge, JudgeUnavailable

SECRET = 'fixture-api-key-not-a-real-credential'


def payload(**changes):
    result = {'model': 'fixture-model', 'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}],
              'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14}}
    result.update(changes)
    return result


def response(data):
    return io.BytesIO(json.dumps(data).encode())


def http_error(status, body='', retry_after=None):
    headers = {} if retry_after is None else {'Retry-After': retry_after}
    return HTTPError('http://127.0.0.1:12345/v1/chat/completions', status, 'fixture', headers,
                     io.BytesIO(body.encode()))


class APIContractTests(unittest.TestCase):
    def setUp(self):
        from nir_eval import native_backend
        native_backend._COOLDOWNS.clear()

    def provider(self, **changes):
        config = {'type': 'openai-compatible', 'base_url': 'http://127.0.0.1:12345/v1',
                  'api_key_env': 'NIR_FIXTURE_KEY'}
        config.update(changes)
        with patch.dict(os.environ, {'NIR_FIXTURE_KEY': SECRET}):
            return OpenAICompatibleProvider(config, allow_remote=True)

    def test_remote_consent_and_key_are_required_before_request(self):
        config = {'type': 'openai-compatible', 'base_url': 'https://api.example.test/v1', 'api_key_env': 'NIR_MISSING_TEST_KEY'}
        with patch.dict(os.environ, {}, clear=True), patch('nir_eval.native_backend.urllib.request.build_opener') as build_opener:
            with self.assertRaises(ValueError): OpenAICompatibleProvider(config, False)
            with self.assertRaises(ValueError): OpenAICompatibleProvider(config, True)
            build_opener.assert_not_called()

    def test_local_http_does_not_inherit_environment_proxy(self):
        with patch.dict(os.environ, {'HTTP_PROXY': 'http://private-proxy', 'HTTPS_PROXY': 'http://private-proxy'}):
            provider = self.provider()
        proxy_handlers = [handler for handler in provider.opener.handlers
                          if hasattr(handler, 'proxies')]
        self.assertTrue(all(handler.proxies == {} for handler in proxy_handlers))
        self.assertFalse(any(hasattr(handler, 'proxy_open') for handler in provider.opener.handlers))

    def test_credential_query_and_plain_remote_http_urls_rejected(self):
        for url in ['https://u:secret@example.test/v1', 'https://example.test/v1?api_key=secret',
                    'https://example.test/v1#key', 'http://example.test/v1']:
            with self.subTest(url=url):
                with self.assertRaises(ValueError): self.provider(base_url=url)

    def test_secret_echo_redacted_everywhere_in_saved_generation(self):
        p = self.provider()
        data = payload(model={'unexpected': SECRET},
                       choices=[{'message': {'content': 'echo ' + SECRET}, 'finish_reason': SECRET}],
                       usage={'prompt_tokens':10, 'completion_tokens':4,
                              'opencode': {'billing_source': SECRET}})
        p.opener.open = Mock(return_value=response(data))
        result = p.generate([], 'fixture-model', {}, {})
        serialized = json.dumps({'text': result.text, 'usage': result.usage, 'metadata': result.metadata})
        self.assertNotIn(SECRET, serialized)

    def test_unknown_cost_and_tariff_estimates_not_real_spend(self):
        data = payload()
        meta = response_metadata(data, {})
        self.assertIsNone(meta['cost_usd'])
        estimated = response_metadata(data, {'input_price_per_million':1,'output_price_per_million':2})
        self.assertIsNone(estimated['cost_usd'])
        self.assertAlmostEqual(estimated['estimated_cost_usd'], .000018)
        data['usage']['cost'] = 0
        self.assertEqual(response_metadata(data, {})['cost_usd'], 0)
        data['usage']['opencode'] = {'estimated':True}
        self.assertIsNone(response_metadata(data,{})['cost_usd'])

    def test_null_tariffs_do_not_break_valid_reply(self):
        config = {'type':'openai-compatible','base_url':'http://localhost/v1',
                  'input_price_per_million':None,'output_price_per_million':None}
        validate_api_config(config)
        self.assertIsNone(response_metadata(payload(),config)['cost_usd'])

    def test_invalid_costs_never_become_known_zero(self):
        for cost in [None, True, -1, float('nan'), float('inf'), '0']:
            data = payload(); data['usage']['cost'] = cost
            with self.subTest(cost=cost):
                self.assertIsNone(response_metadata(data,{})['cost_usd'])

    def test_429_sets_cooldown_without_repeated_http_request(self):
        p = self.provider()
        p.opener.open = Mock(side_effect=http_error(429, SECRET, '120'))
        with self.assertRaises(ProviderUnavailable) as first:
            p.generate([], 'fixture-model', {}, {})
        self.assertEqual(first.exception.retry_after_seconds, 120)
        self.assertNotIn(SECRET,str(first.exception))
        with self.assertRaises(ProviderUnavailable) as second:
            p.generate([], 'fixture-model', {}, {})
        self.assertEqual(str(second.exception),'api_cooldown_active')
        self.assertEqual(p.opener.open.call_count,1)

    def test_safe_400_restrictions_only_accept_known_codes(self):
        for code, expected in [('Hosted outside Europe and the United States', 'api_http_400_region_restricted'),
                               ('trains on request data', 'api_http_400_training_restricted'),
                               (SECRET, 'api_http_400')]:
            p = self.provider()
            p.opener.open = Mock(side_effect=http_error(400, code))
            with self.assertRaises(ProviderError) as error:
                p.generate([], 'fixture-model', {}, {})
            self.assertEqual(str(error.exception), expected)

    def test_invalid_retry_after_falls_back_to_finite_delay(self):
        for header in ['NaN', 'inf', 'not-a-date']:
            p = self.provider()
            from nir_eval import native_backend
            native_backend._COOLDOWNS.clear()
            p.opener.open = Mock(side_effect=http_error(429, retry_after=header))
            with self.assertRaises(ProviderUnavailable) as error: p.generate([], 'fixture-model', {}, {})
            self.assertEqual(error.exception.retry_after_seconds,60)

    def test_malformed_shapes_become_sanitized_provider_error(self):
        for data in [payload(choices=['bad']), payload(choices=[{'message': 'bad'}]), payload(choices=[]), []]:
            p = self.provider(); p.opener.open = Mock(return_value=response(data))
            with self.subTest(data=data):
                with self.assertRaises(ProviderError): p.generate([], 'fixture-model', {}, {})

    def test_legacy_provider_factory_never_executes_cli(self):
        with patch('subprocess.Popen') as process:
            for kind in ['command','opencode','transformers-local']:
                with self.subTest(kind=kind):
                    provider = make_provider({'type':kind,'command':['touch','must-not-run']},True)
                    with self.assertRaises(ProviderError):provider.generate([], 'model', {}, {})
            process.assert_not_called()

    def test_remote_direct_judge_also_requires_key(self):
        config = {'type':'openai-compatible','model':'fixture-judge',
                  'base_url':'https://api.example.test/v1','api_key_env':'NIR_MISSING_TEST_KEY'}
        with patch.dict(os.environ,{},clear=True), patch('nir_eval.native_backend.urllib.request.build_opener') as build_opener:
            with self.assertRaises(ValueError): make_judge(config,allow_remote=True)
            build_opener.assert_not_called()


class ExperimentPreflightTests(unittest.TestCase):
    def setUp(self):
        import contextlib
        import tempfile
        from pathlib import Path
        from nir_eval.experiment import main
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/'experiment'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['init', str(self.root), '--demo']),0)
            self.assertEqual(main(['solver-add', str(self.root), '--provider','api','--base-url','http://localhost:12345/v1','--model','solver','--name','baseline']),0)
            self.assertEqual(main(['judge-set', str(self.root), '--provider','api','--base-url','http://localhost:12345/v1','--model','judge']),0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_invalid_policy_rejected_before_any_model_call(self):
        from nir_eval.experiment import run_experiment
        path=self.root/'experiment.json';config=json.loads(path.read_text())
        config['policy']['min_mean_judge_score']=2
        path.write_text(json.dumps(config))
        with patch('nir_eval.experiment.run_business_matrix') as run:
            with self.assertRaises(ValueError):run_experiment(self.root,self.root/'output')
            run.assert_not_called()

    def test_duplicate_configuration_identity_rejected_in_plan(self):
        from nir_eval.experiment import experiment_plan
        source=self.root/'solver-baseline.json'
        (self.root/'solver-duplicate.json').write_bytes(source.read_bytes())
        path=self.root/'experiment.json';config=json.loads(path.read_text())
        config['solvers'].append('solver-duplicate.json');path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):experiment_plan(self.root)

    def test_run_uses_saved_input_snapshots(self):
        from nir_eval.experiment import run_experiment
        output=self.root/'output'
        result={'status':'no_eligible_configuration','candidates':[],'policy':{},'selected':None,'tied':[],'sources':[]}
        with patch('nir_eval.experiment.run_business_matrix') as run, patch('nir_eval.experiment.compare_runs',return_value=result), patch('nir_eval.experiment.report'):
            run_experiment(self.root,output)
        spec=run.call_args.args[0][0]
        cases_path,rubrics_path,solver_path=(spec[k] for k in ('cases_path','rubrics_path','solver_path'))
        self.assertEqual(cases_path,output/'inputs/cases.jsonl')
        self.assertEqual(rubrics_path,output/'inputs/rubrics.jsonl')
        self.assertEqual(solver_path,output/'inputs/solver-baseline.json')
        self.assertEqual(spec['judge_path'],output/'inputs/judge.json')


if __name__ == '__main__':
    unittest.main()
