"""Business matrix contract fixtures. No network or model-quality claims."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nir_eval.business.running import run_business_matrix,run_business
from nir_eval.business.schema import load_jsonl,ValidationError
from nir_eval.providers import Generation,ProviderUnavailable


class BusinessBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.cases=self.root/'cases.jsonl';self.rubrics=self.root/'rubrics.jsonl'
        self.cases.write_text(''.join(json.dumps({'id':f'case{i}','task':'Return 5','context':'five','reference':'PRIVATE_REFERENCE','rubric_id':'r'})+'\n' for i in range(2)))
        self.rubrics.write_text(json.dumps({'id':'r','criteria':[{'id':'right','description':'five'}]})+'\n')
        self.provider={'type':'openai-compatible','engine':'promptfoo','base_url':'http://127.0.0.1:1/v1'}
        self.judge=self.root/'judge.json'
        self.judge.write_text(json.dumps({'type':'generation-provider','model':'separate-judge','provider':self.provider,
                                         'parameters':{},'max_calls':100,'structured_retries':0}))
        self.calls=[]

    def specs(self,models=1,configs=1,repetitions=1):
        specs=[]
        for i in range(models):
            solver=self.root/f'solver{i}.json'
            solver.write_text(json.dumps({'model':f'model{i}','provider':self.provider,
                'configurations':[{'name':f'c{c}','parameters':{'temperature':0}} for c in range(configs)],
                'repetitions':repetitions,'max_calls':100}))
            specs.append(dict(cases_path=self.cases,rubrics_path=self.rubrics,solver_path=solver,
                              judge_path=self.judge,output=self.root/f'out{i}',save_raw_judge=True))
        return specs

    def good(self,job,cost=.1):
        if job['context']['config']=='business-judge':
            text=json.dumps({'case_id':job['context']['id'],'criteria':[{'id':'right','score':4,
                'rationale':'fixture','evidence':[{'source':'candidate','quote':'5'}]}],'critical_errors':[]})
        else:text='5'
        return Generation(text,{'total_tokens':5},{'cost_usd':cost,'request_attempted':True})

    def dispatcher(self,jobs,max_concurrency,on_result,*,on_started=None):
        self.calls.append(jobs)
        for job in reversed(jobs):
            if on_started:on_started(job['job_id'],'2026-11-11T12:00:00+00:00')
            if job['context']['config']!='business-judge':
                self.assertNotIn('PRIVATE_REFERENCE',json.dumps(job['messages']))
                self.assertNotIn('rubric',json.dumps(job['messages']))
            on_result(job['job_id'],self.good(job),.25)
        return {'scheduler':'fixture-not-promptfoo','max_concurrency':max_concurrency,'completed':len(jobs)}

    def run_specs(self,specs,dispatcher=None,concurrency=3):
        with patch('nir_eval.business.running.evaluate_jobs',side_effect=dispatcher or self.dispatcher),contextlib.redirect_stdout(io.StringIO()):
            return run_business_matrix(specs,max_concurrency=concurrency)

    def test_all_models_configurations_cases_repetitions_share_one_solver_matrix(self):
        specs=self.specs(models=2,configs=2,repetitions=2)
        results=self.run_specs(specs)
        self.assertEqual([len(batch) for batch in self.calls],[16,16])
        self.assertEqual(len({job['job_id'] for batch in self.calls for job in batch}),32)
        self.assertEqual({job['model'] for job in self.calls[0]},{'model0','model1'})
        for spec,result in zip(specs,results):
            self.assertEqual(result['summary']['judge_evaluated'],8)
            output=spec['output'];rows=load_jsonl(output/'records.jsonl')
            self.assertEqual(len({_identity(r) for r in rows}),8)
            self.assertTrue(all(r['total_cost_usd']==.2 for r in rows))
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['scheduler']['owner'],'promptfoo.evaluate')
            self.assertEqual(manifest['scheduler']['max_concurrency'],3)
            self.assertEqual(manifest['execution']['new_solver_calls'],8)
            self.assertEqual(manifest['execution']['new_judge_calls'],8)
            self.assertEqual(manifest['execution']['inflight_jobs'],{})

    def test_unknown_cost_and_transport_error_remain_unknown(self):
        specs=self.specs()
        def dispatch(jobs,limit,on_result,*,on_started=None):
            for job in jobs:
                if job['context']['config']=='business-judge' and job['context']['id']=='case1':
                    result=ProviderUnavailable('api_http_503')
                else:result=self.good(job,None)
                on_result(job['job_id'],result,.2)
            return {'completed':len(jobs)}
        self.run_specs(specs,dispatch)
        rows=load_jsonl(specs[0]['output']/'records.jsonl')
        self.assertTrue(all(r['total_cost_usd'] is None for r in rows))
        bad=next(r for r in rows if r['id']=='case1')
        self.assertIsNone(bad['score']);self.assertIsNone(bad['passed'])
        self.assertEqual(bad['judge']['reason_code'],'judge_api_http_503')

    def test_repairs_use_original_validator_and_deterministic_shared_budget(self):
        config=json.loads(self.judge.read_text());config.update(max_calls=3,structured_retries=1)
        self.judge.write_text(json.dumps(config));specs=self.specs()
        attempts={};batches=[]
        def dispatch(jobs,limit,on_result,*,on_started=None):
            batches.append(jobs)
            for job in reversed(jobs):
                if job['context']['config']=='business-judge':
                    case=job['context']['id'];attempts[case]=attempts.get(case,0)+1
                    result=self.good(job) if attempts[case]>1 else Generation('{}',{}, {'cost_usd':.1})
                else:result=self.good(job)
                on_result(job['job_id'],result,.2)
            return {'completed':len(jobs)}
        self.run_specs(specs,dispatch)
        self.assertEqual([len(b) for b in batches],[2,2,1])
        self.assertEqual(batches[-1][0]['context']['id'],'case0')
        self.assertIn('failed strict validation',batches[-1][0]['messages'][-1]['content'])
        rows={r['id']:r for r in load_jsonl(specs[0]['output']/'records.jsonl')}
        self.assertEqual(rows['case0']['judge']['status'],'evaluated')
        self.assertEqual(len(rows['case0']['judge']['attempts']),2)
        self.assertAlmostEqual(rows['case0']['total_cost_usd'],.3)
        self.assertEqual(rows['case1']['judge']['reason_code'],'judge_call_budget_exhausted')
        self.assertIsNone(rows['case1']['score'])
        self.assertGreaterEqual(rows['case1']['judge']['latency_seconds'],.2)
        manifest=json.loads((specs[0]['output']/'manifest.json').read_text())
        self.assertEqual(manifest['execution']['new_judge_calls'],3)

    def test_partial_judgements_are_durable_and_resume_only_missing_work(self):
        specs=self.specs();output=specs[0]['output']
        def interrupted(jobs,limit,on_result,*,on_started=None):
            for index,job in enumerate(jobs):
                if on_started:on_started(job['job_id'],'2026-11-11T12:00:00+00:00')
                if job['context']['config']=='business-judge' and index==1:raise KeyboardInterrupt()
                on_result(job['job_id'],self.good(job),.1)
                path=output/('records.jsonl' if job['context']['config']=='business-judge' else 'responses.jsonl')
                self.assertEqual(len(load_jsonl(path)),index+1)
            return {'completed':len(jobs)}
        with self.assertRaises(KeyboardInterrupt):self.run_specs(specs,interrupted)
        manifest=json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['state'],'interrupted')
        self.assertEqual(manifest['execution']['unresolved_scheduled_jobs'],1)
        self.assertEqual(len(manifest['execution']['inflight_jobs']),1)
        self.assertIsNone(manifest['execution']['new_call_cost_usd'])
        original={p.relative_to(output):p.read_bytes() for p in output.rglob('*') if p.is_file()}
        target=self.root/'resume';spec=dict(specs[0],reuse_from=output,output=target)
        self.run_specs([spec])
        self.assertEqual([len(batch) for batch in self.calls],[1])
        self.assertEqual(self.calls[0][0]['context']['config'],'business-judge')
        self.assertEqual({p.relative_to(output):p.read_bytes() for p in output.rglob('*') if p.is_file()},original)
        resumed=json.loads((target/'manifest.json').read_text())
        self.assertEqual(resumed['execution']['reused_solver_answers'],2)
        self.assertEqual(resumed['execution']['reused_judgements'],1)

    def test_cooldown_refusal_is_not_counted_as_api_request(self):
        specs=self.specs()
        def dispatch(jobs,limit,on_result,*,on_started=None):
            for job in jobs:
                if on_started:on_started(job['job_id'],'2026-11-11T12:00:00+00:00')
                error=ProviderUnavailable('api_cooldown_active');error.request_attempted=False;error.retry_after_seconds=60
                on_result(job['job_id'],error,0)
            return {'completed':len(jobs)}
        self.run_specs(specs,dispatch)
        manifest=json.loads((specs[0]['output']/'manifest.json').read_text())
        self.assertEqual(manifest['execution']['new_solver_calls'],0)
        self.assertEqual(manifest['execution']['new_judge_calls'],0)
        self.assertEqual(manifest['execution']['new_call_cost_usd'],0)
        self.assertTrue(all(row['score'] is None for row in load_jsonl(specs[0]['output']/'records.jsonl')))
        for row in load_jsonl(specs[0]['output']/'solver_records.jsonl'):
            self.assertIsNone(row['latency_seconds'])
            self.assertIsNone(row['started_at_utc'])
        for row in load_jsonl(specs[0]['output']/'records.jsonl'):
            self.assertIsNone(row['judge']['latency_seconds'])

    def test_judge_cooldown_without_request_has_unknown_latency(self):
        specs=self.specs()
        def dispatch(jobs,limit,on_result,*,on_started=None):
            for job in jobs:
                if job['context']['config']=='business-judge':
                    result=ProviderUnavailable('api_cooldown_active');result.request_attempted=False
                else:result=self.good(job)
                on_result(job['job_id'],result,.25)
            return {'completed':len(jobs)}
        self.run_specs(specs,dispatch)
        for row in load_jsonl(specs[0]['output']/'records.jsonl'):
            self.assertIsNone(row['score'])
            self.assertIsNone(row['judge']['latency_seconds'])

    def test_judge_budget_without_request_has_unknown_latency(self):
        config=json.loads(self.judge.read_text());config['max_calls']=1
        self.judge.write_text(json.dumps(config));specs=self.specs()
        self.run_specs(specs)
        rows=load_jsonl(specs[0]['output']/'records.jsonl')
        skipped=next(row for row in rows if row['judge']['reason_code']=='judge_call_budget_exhausted')
        self.assertIsNone(skipped['judge']['latency_seconds'])

    def test_invalid_concurrency_and_colliding_outputs_fail_before_scheduling(self):
        specs=self.specs()
        for concurrency in (0,33,True):
            with self.assertRaises(ValidationError):self.run_specs(specs,concurrency=concurrency)
        with self.assertRaises(ValidationError):self.run_specs(specs*2)
        self.assertFalse(specs[0]['output'].exists())
        self.assertEqual(self.calls,[])


def _identity(row):
    return row['id'],row['model'],row['configuration'],row['repetition']


if __name__=='__main__':unittest.main()
