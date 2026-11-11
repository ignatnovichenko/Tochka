"""Immutable resume software fixtures; never external model calls."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nir_eval.business.running import run_business, load_solver_config, load_judge_config
from nir_eval.business.schema import ValidationError, load_jsonl
from nir_eval.business_selection import select_business_run
from nir_eval.providers import Generation
from nir_eval.business.providers import JudgeReply
from batch_fixture import batch_fixture
from nir_eval.resume import load_reusable_run, resume_experiment, resume_plan


def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}

class Solver:
    def __init__(self, interrupt=None):self.calls=0;self.interrupt=interrupt
    def generate(self,*args):
        self.calls+=1
        if self.calls==self.interrupt:raise KeyboardInterrupt()
        return Generation('5',{'total_tokens':3},{'cost_usd':.01,'cost_basis':'provider_reported'})

class Judge:
    kind='model-inference';model='fixture-judge';structured_retries=0
    def __init__(self,invalid=None):self.calls=0;self.invalid=invalid
    def judge(self,messages,case_id):
        self.calls+=1
        text='invalid' if self.calls==self.invalid else json.dumps({'case_id':case_id,'criteria':[{'id':'correct','score':4,'rationale':'fixture','evidence':[{'source':'candidate','quote':'5'}]}],'critical_errors':[]})
        return JudgeReply(text,{}, {'cost_usd':.02,'cost_basis':'provider_reported'})

class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.cases=self.root/'cases.jsonl';self.rubrics=self.root/'rubrics.jsonl';self.solver=self.root/'solver.json';self.judge=self.root/'judge.json'
        self.cases.write_text(''.join(json.dumps({'id':str(i),'task':'Return 5','context':'Five','rubric_id':'r'})+'\n' for i in range(2)))
        self.rubrics.write_text(json.dumps({'id':'r','criteria':[{'id':'correct','description':'Correct five'}]})+'\n')
        provider={'type':'openai-compatible','base_url':'http://127.0.0.1:1/v1'}
        self.solver.write_text(json.dumps({'model':'fixture-solver','provider':provider,'configurations':[{'name':'base'}],'max_calls':2}))
        self.judge.write_text(json.dumps({'type':'generation-provider','provider':provider,'model':'fixture-judge','parameters':{},'max_calls':4}))
        self.source=self.root/'source';self.output=self.root/'resumed'
    def run_it(self,solver,judge,output=None,reuse=None):
        with patch('nir_eval.business.running.evaluate_jobs',side_effect=batch_fixture(solver,judge)),contextlib.redirect_stdout(io.StringIO()):
            return run_business(self.cases,self.rubrics,self.solver,output or self.source,judge_path=self.judge,save_raw_judge=True,reuse_from=reuse)
    def test_old_solver_or_judge_engine_cannot_be_mixed_on_resume(self):
        self.run_it(Solver(), Judge())
        path = self.source/'manifest.json'
        original = json.loads(path.read_text())
        for role in ('solver', 'judge'):
            with self.subTest(role=role):
                manifest = json.loads(json.dumps(original))
                manifest['execution_backend'].pop(role)
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValidationError, 'engine'):
                    self.run_it(Solver(), Judge(), self.output, self.source)
                self.assertFalse(self.output.exists())
        path.write_text(json.dumps(original))

    def test_reuses_solver_and_valid_judge_only(self):
        self.run_it(Solver(),Judge(invalid=2));before=hashes(self.source)
        solver=Solver();judge=Judge();self.run_it(solver,judge,self.output,self.source)
        self.assertEqual(solver.calls,0);self.assertEqual(judge.calls,1);self.assertEqual(hashes(self.source),before)
        result=select_business_run(self.output,{})
        self.assertEqual(result['candidates'][0]['judge_coverage'],1)
        manifest=json.loads((self.output/'manifest.json').read_text())
        self.assertEqual(manifest['execution']['reused_solver_answers'],2)
        self.assertEqual(manifest['execution']['reused_judgements'],1)
        self.assertAlmostEqual(manifest['execution']['new_call_cost_usd'],.02)
        self.assertEqual((self.output/'history/source/records.jsonl').read_bytes(),(self.source/'records.jsonl').read_bytes())
    def test_interrupt_resumes_only_missing_solver(self):
        with self.assertRaises(KeyboardInterrupt):self.run_it(Solver(interrupt=2),Judge())
        before=hashes(self.source);solver=Solver();judge=Judge()
        self.run_it(solver,judge,self.output,self.source)
        self.assertEqual(solver.calls,1);self.assertEqual(judge.calls,2);self.assertEqual(hashes(self.source),before)
        self.assertEqual(select_business_run(self.output,{})['candidates'][0]['judge_coverage'],1)
    def test_incomplete_final_write_preserved_and_skipped(self):
        with self.assertRaises(KeyboardInterrupt):self.run_it(Solver(interrupt=2),Judge())
        with (self.source/'solver_records.jsonl').open('ab') as stream:stream.write(b'{"id":')
        before=hashes(self.source);solver=Solver();self.run_it(solver,Judge(),self.output,self.source)
        self.assertEqual(solver.calls,1);self.assertEqual(hashes(self.source),before)
        self.assertIn('incomplete final write',json.loads((self.output/'manifest.json').read_text())['resumed_from']['notes'][0])
    def test_corrupted_complete_row_rejected(self):
        with self.assertRaises(KeyboardInterrupt):self.run_it(Solver(interrupt=2),Judge())
        with (self.source/'solver_records.jsonl').open('ab') as stream:stream.write(b'{bad}\n')
        with self.assertRaises(ValidationError):self.run_it(Solver(),Judge(),self.output,self.source)
        self.assertFalse(self.output.exists())
    def test_completed_tamper_rejected_before_provider(self):
        self.run_it(Solver(),Judge())
        with (self.source/'responses.jsonl').open('a') as stream:stream.write('{}\n')
        with self.assertRaises(ValidationError):self.run_it(Solver(),Judge(),self.output,self.source)
        self.assertFalse(self.output.exists())
    def test_mixed_old_new_judge_timing_is_not_relabelled(self):
        self.run_it(Solver(),Judge(invalid=2))
        path=self.source/'manifest.json';manifest=json.loads(path.read_text())
        manifest['judge_latency_scope']='Legacy full call including startup'
        path.write_text(json.dumps(manifest))
        self.run_it(Solver(),Judge(),self.output,self.source)
        updated=json.loads((self.output/'manifest.json').read_text())
        self.assertIsNone(updated['judge_latency_scope'])
        result=select_business_run(self.output,{})
        self.assertEqual(result['candidates'][0]['judge_coverage'],1)
        self.assertIsNone(result['candidates'][0]['mean_total_latency_seconds'])

    def test_changed_inputs_rejected(self):
        self.run_it(Solver(),Judge())
        self.cases.write_text(self.cases.read_text().replace('Return 5','Return 6'))
        with self.assertRaises(ValidationError):self.run_it(Solver(),Judge(),self.output,self.source)
    def test_complete_reuse_has_zero_new_calls_and_cost(self):
        self.run_it(Solver(),Judge());solver=Solver();judge=Judge()
        self.run_it(solver,judge,self.output,self.source)
        self.assertEqual((solver.calls,judge.calls),(0,0))
        self.assertEqual(json.loads((self.output/'manifest.json').read_text())['execution']['new_call_cost_usd'],0)
    def test_no_nested_output_and_no_legacy_auto_calls(self):
        self.run_it(Solver(),Judge())
        with self.assertRaises(ValidationError):self.run_it(Solver(),Judge(),self.source/'nested',self.source)
        data=json.loads(self.solver.read_text());data['provider']={'type':'command','command':['never-run']};self.solver.write_text(json.dumps(data))
        with self.assertRaises(ValidationError):self.run_it(Solver(),Judge(),self.output,self.source)
    def test_companion_response_recovered_after_interrupt(self):
        with self.assertRaises(KeyboardInterrupt):self.run_it(Solver(interrupt=2),Judge())
        (self.source/'responses.jsonl').write_text('')
        solver=Solver();self.run_it(solver,Judge(),self.output,self.source)
        self.assertEqual(solver.calls,1)

    def wrap_experiment(self):
        import shutil
        from nir_eval.business.running import sha
        root=self.root/'experiment-run';inputs=root/'inputs';inputs.mkdir(parents=True)
        for p in [self.cases,self.rubrics,self.solver,self.judge]:shutil.copy2(p,inputs/p.name)
        cfg={'solvers':['solver.json'],'judge':'judge.json','policy':{}}
        (inputs/'experiment.json').write_text(json.dumps(cfg))
        shutil.copytree(self.source,root/'solver')
        (root/'experiment-manifest.json').write_text(json.dumps({'state':'completed','planned_runs':['solver'],'runs':['solver'],'input_hashes':{p.name:sha(p) for p in inputs.iterdir()}}))
        return root
    def test_judge_quota_deadline_prevents_resume(self):
        from nir_eval.business.providers import JudgeUnavailable
        class Limited(Judge):
            def judge(self,*args):
                exc=JudgeUnavailable('judge_api_http_429');exc.retry_after_seconds=120
                raise exc
        self.run_it(Solver(),Limited())
        root=self.wrap_experiment();plan=resume_plan(root)
        self.assertIsNotNone(plan['retry_not_before_utc'])
        with patch('nir_eval.providers.make_provider') as provider:
            with self.assertRaises(ValidationError):resume_experiment(root,self.output)
            provider.assert_not_called()
        self.assertFalse(self.output.exists())
    def test_changed_judge_prompt_cannot_be_reused(self):
        self.run_it(Solver(),Judge())
        with patch('nir_eval.resume.build_judge_messages',return_value=[]):
            with self.assertRaises(ValidationError):self.run_it(Solver(),Judge(),self.output,self.source)
    def test_active_source_cannot_be_resumed(self):
        import fcntl
        self.run_it(Solver(),Judge())
        with (self.source/'.run.lock').open('r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(ValidationError):self.run_it(Solver(),Judge(),self.output,self.source)

    def test_interrupted_reusable_judgement_contract_preflight(self):
        self.run_it(Solver(),Judge(invalid=2))
        manifest=json.loads((self.source/'manifest.json').read_text());manifest['state']='interrupted';manifest.pop('records_sha256')
        (self.source/'manifest.json').write_text(json.dumps(manifest))
        records=load_jsonl(self.source/'records.jsonl');records[0]['schema_version']='invalid'
        (self.source/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        solver=Solver();judge=Judge()
        with self.assertRaises(ValidationError):self.run_it(solver,judge,self.output,self.source)
        self.assertEqual((solver.calls,judge.calls),(0,0));self.assertFalse(self.output.exists())

if __name__=='__main__':unittest.main()
