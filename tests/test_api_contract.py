"""API boundary tests: mocked transport only; never contact external models."""
import io
import json
import os
import time
import unittest
import urllib.error
from unittest.mock import Mock, patch

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
    stream = io.BytesIO(json.dumps(data).encode())
    stream.__enter__ = lambda: stream
    return stream


class APIContractTests(unittest.TestCase):
    def provider(self, **changes):
        config = {'type': 'openai-compatible', 'base_url': 'http://127.0.0.1:12345/v1',
                  'api_key_env': 'NIR_FIXTURE_KEY'}
        config.update(changes)
        with patch.dict(os.environ, {'NIR_FIXTURE_KEY': SECRET}):
            return OpenAICompatibleProvider(config, allow_remote=True)

    def test_remote_consent_and_key_are_required_before_request(self):
        config = {'type': 'openai-compatible', 'base_url': 'https://api.example.test/v1', 'api_key_env': 'NIR_MISSING_TEST_KEY'}
        with patch.dict(os.environ, {}, clear=True), patch('urllib.request.build_opener') as opener:
            with self.assertRaises(ValueError): OpenAICompatibleProvider(config, False)
            with self.assertRaises(ValueError): OpenAICompatibleProvider(config, True)
            opener.assert_not_called()

    def test_localhost_does_not_use_environment_proxy(self):
        with patch('urllib.request.ProxyHandler', return_value=Mock()) as proxy, patch('urllib.request.build_opener'):
            self.provider()
        proxy.assert_called_once_with({})

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
        p.opener = Mock(); p.opener.open.return_value = response(data)
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
        p.opener = Mock()
        p.opener.open.side_effect = urllib.error.HTTPError(p.url,429,'rate limit',{'Retry-After':'120'},io.BytesIO(SECRET.encode()))
        with self.assertRaises(ProviderUnavailable) as first:
            p.generate([], 'fixture-model', {}, {})
        self.assertEqual(first.exception.retry_after_seconds, 120)
        self.assertNotIn(SECRET,str(first.exception))
        with self.assertRaises(ProviderUnavailable) as second:
            p.generate([], 'fixture-model', {}, {})
        self.assertEqual(str(second.exception),'api_cooldown_active')
        self.assertEqual(p.opener.open.call_count,1)

    def test_invalid_retry_after_falls_back_to_finite_delay(self):
        for header in ['NaN', 'inf', 'not-a-date']:
            p = self.provider(); p.opener = Mock()
            p.opener.open.side_effect = urllib.error.HTTPError(p.url,429,'rate limit',{'Retry-After':header},None)
            with self.assertRaises(ProviderUnavailable) as error: p.generate([], 'fixture-model', {}, {})
            self.assertEqual(error.exception.retry_after_seconds,60)

    def test_malformed_shapes_become_sanitized_provider_error(self):
        for data in [payload(choices=['bad']), payload(choices=[{'message': 'bad'}]), payload(choices=[]), []]:
            p = self.provider(); p.opener = Mock(); p.opener.open.return_value = response(data)
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
        with patch.dict(os.environ,{},clear=True), patch('urllib.request.build_opener') as opener:
            with self.assertRaises(ValueError): make_judge(config,allow_remote=True)
            opener.assert_not_called()


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
        with patch('nir_eval.experiment.run_business') as run:
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
        with patch('nir_eval.experiment.run_business') as run, patch('nir_eval.experiment.compare_runs',return_value=result), patch('nir_eval.experiment.report'):
            run_experiment(self.root,output)
        cases_path,rubrics_path,solver_path,_=run.call_args.args
        self.assertEqual(cases_path,output/'inputs/cases.jsonl')
        self.assertEqual(rubrics_path,output/'inputs/rubrics.jsonl')
        self.assertEqual(solver_path,output/'inputs/solver-baseline.json')
        self.assertEqual(run.call_args.kwargs['judge_path'],output/'inputs/judge.json')


if __name__ == '__main__':
    unittest.main()
