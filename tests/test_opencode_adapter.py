"""Offline fixtures only: none of these tests launch OpenCode or a model."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime,timezone
from nir_eval.providers import decode_response,ProviderUnavailable
from nir_eval.business.providers import make_judge,validate_config,GenerationJudgeProvider,JudgeUnavailable

spec=importlib.util.spec_from_file_location('opencode_adapter',Path(__file__).parents[1]/'scripts/opencode_stdio_adapter.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class OpenCodeAdapterTests(unittest.TestCase):
    def events(self,rows):return '\n'.join(json.dumps({'sessionID':'fixture-session',**row}) for row in rows)
    def good(self):return [{'type':'step_start','part':{'type':'step-start'}},{'type':'text','part':{'type':'text','messageID':'m','text':'{"answer":5}'}},{'type':'step_finish','part':{'type':'step-finish','reason':'stop','tokens':{'input':10,'output':5},'cost':0}}]
    def test_source_shape_success_fixture(self):
        r=a.parse_events(self.events(self.good()));self.assertEqual(r['text'],'{"answer":5}');self.assertIsNone(r['metadata']['cost_usd']);self.assertEqual(r['usage'],{})
    def test_text_delta_concatenation(self):
        rows=self.good();rows[1]['part']['text']='a';rows.insert(2,{'type':'text','part':{'type':'text','messageID':'m','text':'b'}})
        self.assertEqual(a.parse_events(self.events(rows))['text'],'ab')
    def test_actual_error_shape_sanitized(self):
        with self.assertRaises(a.AdapterFailure) as e:a.parse_events(self.events([{'type':'error','error':{'type':'provider.free-limit','status':429,'retryAfterMs':20703000,'message':'SECRET','response':{'body':'SECRET'}}}]))
        self.assertTrue(e.exception.unavailable);self.assertNotIn('SECRET',str(e.exception));self.assertEqual(e.exception.code,'opencode_free_limit')
    def test_no_fallback_paid_or_parameters(self):
        for data in [{'model':'opencode/paid','messages':[]},{'model':next(iter(a.FREE_MODELS)),'parameters':{'temperature':0},'messages':[]}]:
            with self.assertRaises(a.AdapterFailure):a.validate_payload(data)
    def test_wrapped_roles_documented(self):
        text=a.validate_payload({'model':next(iter(a.FREE_MODELS)),'parameters':{},'messages':[{'role':'system','content':'x'},{'role':'user','content':'y'}]});self.assertIn('"role":"system"',text)
    def test_cooldown_before_any_config_read_or_launch(self):
        with tempfile.TemporaryDirectory() as p,patch.object(a.subprocess,'run') as run:
            with self.assertRaises(a.AdapterFailure) as e:a.offline_check(Path(p),datetime(2026,11,10,20,tzinfo=timezone.utc))
            self.assertEqual(e.exception.code,'opencode_cooldown_active');run.assert_not_called()
    def test_unrecognized_and_tools_rejected(self):
        for kind in ['tool_use','retry','background_wait','surprise']:
            with self.assertRaises(a.AdapterFailure):a.parse_events(self.events([{'type':kind}]))
    def test_missing_finish_rejected(self):
        with self.assertRaises(a.AdapterFailure):a.parse_events(self.events(self.good()[:-1]))
    def test_multiple_sessions_rejected(self):
        rows=self.good();rows[-1]['sessionID']='different'
        with self.assertRaises(a.AdapterFailure):a.parse_events(self.events(rows))
    def test_unavailable_envelope(self):
        with self.assertRaises(ProviderUnavailable):decode_response('{"text":"","error":"free_limit","unavailable":true}')
    def test_judge_unavailable_not_failed_quality(self):
        class Provider:
            def generate(self,*args):raise ProviderUnavailable('test')
        with self.assertRaises(JudgeUnavailable):GenerationJudgeProvider(Provider(),'fixture').judge([],'x')
    def test_empty_judge_parameters_honored(self):
        cfg={'type':'generation-provider','model':next(iter(a.FREE_MODELS)),'provider':{'type':'command','command':['python','x']},'parameters':{}}
        with patch('nir_eval.providers.make_provider'):
            self.assertEqual(make_judge(cfg).parameters,{})
    def test_conflicting_parameter_settings_rejected(self):
        cfg={'type':'generation-provider','model':'m','provider':{'type':'command','command':['x']},'parameters':{},'max_tokens':512}
        with self.assertRaises(ValueError):validate_config(cfg)

if __name__=='__main__':unittest.main()
