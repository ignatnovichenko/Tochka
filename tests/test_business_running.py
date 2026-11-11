"""Mocked orchestration tests; these records are not model-quality evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nir_eval.business.running import run_business,load_solver_config
from nir_eval.business.schema import ValidationError,strict_json
from nir_eval.providers import Generation
from nir_eval.business.providers import JudgeReply
from batch_fixture import batch_fixture


class BusinessRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.case={'id':'fixture','task':'Return the total','context':'Two plus three','reference':'PRIVATE_REFERENCE','rubric_id':'rubric'}
        self.rubric={'id':'rubric','criteria':[{'id':'correctness','description':'Correct total'}]}
        (self.root/'cases.jsonl').write_text(json.dumps(self.case)+'\n')
        (self.root/'rubrics.jsonl').write_text(json.dumps(self.rubric)+'\n')
        self.config={'model':'fixture-model','provider':{'type':'command','command':['never-executed']},
                     'configurations':[{'name':'baseline'}],'max_calls':1}
        (self.root/'solver.json').write_text(json.dumps(self.config))

    def run_fixture(self,provider,judge=None,**kwargs):
        if judge is not None:
            path=self.root/'judge.json'
            path.write_text(json.dumps({'type':'generation-provider','provider':{'type':'openai-compatible','base_url':'http://127.0.0.1:1/v1'},'model':judge.model,'parameters':{},'max_calls':10}))
            kwargs['judge_path']=path
        with patch('nir_eval.business.running.evaluate_jobs',side_effect=batch_fixture(provider,judge)),contextlib.redirect_stdout(io.StringIO()):
            return run_business(self.root/'cases.jsonl',self.root/'rubrics.jsonl',self.root/'solver.json',self.root/'out',**kwargs)

    def test_disabled_judge_unknown_and_solver_whitelist(self):
        class Solver:
            def generate(_,messages,*args):
                self.assertNotIn('PRIVATE_REFERENCE',json.dumps(messages))
                self.assertNotIn('rubric',json.dumps(messages))
                return Generation('5')
        result=self.run_fixture(Solver())
        self.assertEqual(result['summary']['judge_evaluated'],0)
        self.assertIsNone(result['summary']['mean_judge_score'])
        self.assertEqual(json.loads((self.root/'out/manifest.json').read_text())['state'],'completed')

    def test_valid_judge_and_explicit_raw_capture(self):
        class Solver:
            def generate(self,*args):return Generation('5')
        class Judge:
            kind='test-fixture-not-model';model='fixture-judge'
            def judge(self,messages,case_id):
                return JudgeReply(json.dumps({'case_id':case_id,'criteria':[{'id':'correctness','score':4,
                            'rationale':'fixture','evidence':[{'source':'candidate','quote':'5'}]}],'critical_errors':[]}))
        result=self.run_fixture(Solver(),Judge(),save_raw_judge=True)
        self.assertEqual(result['summary']['mean_judge_score'],1)
        self.assertTrue((self.root/'out/judge_raw.jsonl').is_file())

    def test_solver_failure_never_becomes_zero_quality(self):
        class Solver:
            def generate(self,*args):raise RuntimeError('SECRET_EXCEPTION_BODY')
        result=self.run_fixture(Solver())
        self.assertEqual(result['summary']['candidate_generation_errors'],1)
        self.assertIsNone(result['summary']['mean_judge_score'])
        self.assertNotIn('SECRET_EXCEPTION_BODY',(self.root/'out/solver_records.jsonl').read_text())

    def test_budget_before_provider_call(self):
        self.config['repetitions']=2;(self.root/'solver.json').write_text(json.dumps(self.config))
        with self.assertRaises(ValidationError):self.run_fixture(None)
        self.assertFalse((self.root/'out').exists())

    def test_provider_initialization_failure_marks_interrupted(self):
        with patch('nir_eval.business.running.evaluate_jobs',side_effect=ValueError('missing dependency')):
            with self.assertRaises(ValueError):run_business(self.root/'cases.jsonl',self.root/'rubrics.jsonl',self.root/'solver.json',self.root/'out')
        self.assertEqual(json.loads((self.root/'out/manifest.json').read_text())['state'],'interrupted')

    def test_nonfinite_json_and_huge_integer_rejected(self):
        for text in ['1e309','{"x":1e309}', '1'*5000]:
            with self.assertRaises(ValidationError):strict_json(text)


if __name__=='__main__':unittest.main()
