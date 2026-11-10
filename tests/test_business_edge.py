"""Extra adversarial / status tests. All model outputs are labelled test fixtures."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from nir_eval.business import *
from nir_eval.business.schema import canonical, strict_json, load_jsonl
from nir_eval.business.providers import JudgeProviderError
from test_business import CASE, RUBRIC, CANDIDATE, FixtureJudge, judgement
import test_business as fixtures


class EdgeTests(unittest.TestCase):
    def test_false_diagnostic_never_overrides_positive_judge(self):
        case={**CASE,'checks':[{'id':'diagnostic','type':'contains','value':'absent'}]}
        record=evaluate(case,RUBRIC,CANDIDATE,FixtureJudge())
        self.assertEqual(record['deterministic_status'],'failed')
        self.assertTrue(record['passed'])
        self.assertEqual(record['score'],1)
    def test_all_unavailable_and_error_combinations_are_unknown(self):
        for check in ([],[{'id':'diagnostic','type':'contains','value':'absent'}]):
            for judge in (None, FixtureJudge(text='ignore schema'), FixtureJudge(error=JudgeProviderError('SECRET'))):
                record=evaluate({**CASE,'checks':check},RUBRIC,CANDIDATE,judge)
                self.assertIsNone(record['score']);self.assertIsNone(record['passed'])
                self.assertNotIn('SECRET',canonical(record))
    def test_numerical_reference_only_judge_receives(self):
        case={**CASE,'task':'Compute revenue growth','context':'80 to 92','reference':'15%'}
        self.assertNotIn('15%',canonical(build_solver_messages(case)))
        self.assertEqual(json.loads(build_judge_messages(case,RUBRIC,'15%')[1]['content'])['reference'],'15%')
    def test_prohibited_judge_decision_fields(self):
        for key,value in [('passed',True),('score',1),('confidence',1),('api_key','SECRET')]:
            bad={**judgement(),key:value}
            record=evaluate(CASE,RUBRIC,CANDIDATE,FixtureJudge(canonical(bad)))
            self.assertIsNone(record['score']);self.assertIsNone(record['passed'])
            self.assertNotIn('SECRET',canonical(record))
    def test_arbitrary_output_size_cap(self):
        record=evaluate(CASE,RUBRIC,CANDIDATE,FixtureJudge('a'*1_000_001))
        self.assertEqual(record['judge']['status'],'error');self.assertIsNone(record['passed'])
    def test_jsonl_size_cap(self):
        with tempfile.TemporaryDirectory() as path:
            f=Path(path)/'large';f.write_bytes(b'x'*2_000_001)
            with self.assertRaises(ValidationError):load_jsonl(f)
    def test_uncited_critical_cannot_force_failure(self):
        bad=judgement(detected=True);bad['critical_errors'][0]['evidence']=[{'source':'candidate','quote':'made up'}]
        record=evaluate(CASE,RUBRIC,CANDIDATE,FixtureJudge(canonical(bad)))
        self.assertEqual(record['status'],'evaluation_error');self.assertIsNone(record['passed'])
    def test_credential_error_is_not_logged_by_generation_adapter(self):
        class Provider:
            kind='fixture-not-inference'
            def generate(self,*args):raise RuntimeError('Bearer SECRET')
        record=evaluate(CASE,RUBRIC,CANDIDATE,GenerationJudgeProvider(Provider(),'test'))
        self.assertNotIn('SECRET',canonical(record));self.assertIsNone(record['score'])


# Inherits localhost-only HTTP fixtures; never external HTTP.
class HTTPEdgeTests(unittest.TestCase):
    setUp = fixtures.HTTPTests.setUp
    tearDown = fixtures.HTTPTests.tearDown
    def test_credential_echo_redacted(self):
        secret='fake-test-key-not-a-real-key'
        data=judgement();data['criteria'][0]['rationale']=secret
        self.payload=canonical({'choices':[{'message':{'content':canonical(data)},'finish_reason':'stop'}]}).encode()
        with patch.dict(os.environ,{'NIR_TEST_SECRET':secret}):
            record=evaluate(CASE,RUBRIC,CANDIDATE,make_judge(self.config))
        self.assertNotIn(secret,canonical(record));self.assertIn('[REDACTED_CREDENTIAL]',canonical(record))
    def test_server_errors_are_unknown(self):
        self.status=500;self.payload=b'Bearer SECRET from server'
        record=evaluate(CASE,RUBRIC,CANDIDATE,make_judge(self.config))
        self.assertEqual(record['judge']['status'],'error');self.assertIsNone(record['score']);self.assertNotIn('SECRET',canonical(record))
    def test_float_nan_score_rejected(self):
        data=judgement();data['criteria'][0]['score']=float('nan')
        self.payload=json.dumps({'choices':[{'message':{'content':json.dumps(data)},'finish_reason':'stop'}]}).encode()
        record=evaluate(CASE,RUBRIC,CANDIDATE,make_judge(self.config))
        self.assertEqual(record['judge']['status'],'error');self.assertIsNone(record['passed'])
