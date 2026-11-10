import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nir_eval.experiment import main, compare_runs, run_experiment, read
from nir_eval.opencode import parse_events, OpenCodeProvider
from nir_eval.providers import ProviderError, ProviderUnavailable

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'experiment'
        self.assertEqual(self.cli('init', str(self.root), '--demo'), 0)
    def tearDown(self): self.tmp.cleanup()
    def cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(list(args))
    def test_demo_valid(self):
        from nir_eval.business.schema import import_inputs
        cases, rubrics, _ = import_inputs(self.root/'cases.jsonl', self.root/'rubrics.jsonl')
        self.assertEqual(len(cases),4)
        self.assertEqual(len(rubrics),4)
    def test_add_document_and_rubric(self):
        doc=self.root/'policy.txt';doc.write_text('Возврат 14 дней')
        self.assertEqual(self.cli('rubric-add', str(self.root), '--id','new','--criterion','facts=Точность','--critical','false=Ложь'),0)
        self.assertEqual(self.cli('task-add', str(self.root), '--id','new','--task','Ответьте','--rubric','new','--document',str(doc)),0)
        self.assertIn('Возврат 14 дней', (self.root/'cases.jsonl').read_text())
    def test_duplicate_and_unknown_rubric(self):
        self.assertEqual(self.cli('task-add',str(self.root),'--id','new','--task','x','--rubric','absent'),2)
        self.assertEqual(self.cli('rubric-add',str(self.root),'--id','refund_v1','--criterion','x=y'),2)
    def solver(self, model='opencode/big-pickle', name='baseline'):
        return self.cli('solver-add',str(self.root),'--provider','opencode','--model',model,'--name',name)
    def judge(self, model='opencode/glm-5.3-flash-free'):
        return self.cli('judge-set',str(self.root),'--provider','opencode','--model',model)
    def test_paid_model_blocked(self):
        self.assertEqual(self.solver('opencode/gpt-paid'),2)
    def test_unsupported_parameter_blocked(self):
        self.assertEqual(self.cli('solver-add',str(self.root),'--provider','opencode','--model','opencode/big-pickle','--name','x','--temperature','0'),2)
    def test_duplicate_configuration_blocked(self):
        self.assertEqual(self.solver(),0);self.assertEqual(self.solver(),2)
    def test_missing_judge_preflight(self):
        self.solver()
        with self.assertRaises(ValueError): run_experiment(self.root, self.root/'run',True)
    def test_same_judge_solver_preflight(self):
        self.solver();self.judge('opencode/big-pickle')
        with self.assertRaises(ValueError): run_experiment(self.root, self.root/'run',True)
        self.assertFalse((self.root/'run').exists())
    def test_remote_consent_preflight(self):
        self.solver();self.judge()
        with self.assertRaises(ValueError): run_experiment(self.root,self.root/'run',False)
    def test_call_budget_preflight(self):
        self.solver();self.judge()
        c=read(self.root/'judge.json');c['max_calls']=1;(self.root/'judge.json').write_text(json.dumps(c))
        with self.assertRaises(ValueError): run_experiment(self.root,self.root/'run',True)
    def test_policy_validation(self):
        self.assertEqual(self.cli('policy-set',str(self.root),'--min-quality','2'),2)
        self.assertEqual(self.cli('policy-set',str(self.root),'--max-cost','0','--max-latency','30'),0)
        self.assertEqual(read(self.root/'experiment.json')['policy']['max_mean_cost_usd'],0)
    def test_local_remote_url_blocked(self):
        self.assertEqual(self.cli('solver-add',str(self.root),'--provider','local','--model','x','--base-url','https://remote.example','--name','a'),2)
    def test_cross_run_judge_mismatch(self):
        selections=[{'cases_sha256':'a','rubrics_sha256':'b','judge_config_sha256':x} for x in ['c','d']]
        with patch('nir_eval.experiment.select_business_run',side_effect=selections):
            with self.assertRaises(ValueError):compare_runs(['a','b'],{})
    def test_unknown_and_ties(self):
        def result(model,score,reasons):
            return {'cases_sha256':'a','rubrics_sha256':'b','judge_config_sha256':'c','source_manifest_sha256':'d','warnings':[], 'candidates':[{'model':model,'configuration':'x','mean_judge_score':score,'rejection_reasons':reasons}]}
        with patch('nir_eval.experiment.select_business_run',side_effect=[result('a',None,['judge_quality_unknown']),result('b',None,['judge_quality_unknown'])]):
            self.assertEqual(compare_runs(['a','b'],{})['status'],'no_eligible_configuration')
        with patch('nir_eval.experiment.select_business_run',side_effect=[result('a',1,[]),result('b',1,[])]):
            self.assertEqual(len(compare_runs(['a','b'],{})['tied']),2)

class NativeOpenCodeTests(unittest.TestCase):
    def events(self, cost=0, reason='stop'):
        return '\n'.join(json.dumps({'sessionID':'s',**x}) for x in [
            {'type':'step_start','part':{}},
            {'type':'text','part':{'messageID':'m','text':'OK'}},
            {'type':'step_finish','part':{'cost':cost,'reason':reason,'tokens':{'input':3,'output':1}}}])
    def test_native_cost_and_tokens(self):
        g=parse_events(self.events());self.assertEqual(g.metadata['cost_usd'],0);self.assertEqual(g.metadata['native_tokens']['input'],3)
    def test_unknown_cost_not_zero(self):
        self.assertIsNone(parse_events(self.events(None)).metadata['cost_usd'])
    def test_truncation_unknown(self):
        with self.assertRaises(ProviderError):parse_events(self.events(reason='length'))
    def test_tools_rejected(self):
        with self.assertRaises(ProviderError):parse_events('{"sessionID":"s","type":"tool_use"}')
    def test_429_sanitized(self):
        with self.assertRaises(ProviderUnavailable) as exc:
            parse_events('{"sessionID":"s","type":"error","error":{"status":429,"retryAfterMs":5000,"message":"SECRET"}}')
        self.assertEqual(exc.exception.retry_after_ms,5000);self.assertNotIn('SECRET',str(exc.exception))
    def test_paid_and_parameters_before_subprocess(self):
        p=OpenCodeProvider({},True)
        with patch('nir_eval.opencode.subprocess.Popen') as run:
            for model,params in [('paid',{}),('opencode/big-pickle',{'temperature':0})]:
                with self.assertRaises(ProviderError):p.generate([],model,params,{})
            run.assert_not_called()
    def test_cooldown_no_request(self):
        import time
        with tempfile.TemporaryDirectory() as t:
            p=OpenCodeProvider({'cwd':t},True)
            p.state.write_text(json.dumps({'retry_not_before_epoch':time.time()+100}))
            with patch('nir_eval.opencode.subprocess.Popen') as run:
                with self.assertRaises(ProviderUnavailable):p.generate([],'opencode/big-pickle',{}, {})
                run.assert_not_called()

if __name__=='__main__':unittest.main()
