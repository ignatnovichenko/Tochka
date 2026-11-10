"""Synthetic evidence tests selection mechanics, never model performance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from nir_eval.runner import fingerprint
from nir_eval.selection import select_configuration,validate_policy


def save(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root, specs=None):
    specs=specs or [('a',[True,False],2.0),('b',[True,True],3.0)]
    sources=[];combined=[]
    for name,flags,latency in specs:
        directory=root/name;directory.mkdir()
        ids=[f'case{i}' for i in range(len(flags))]
        rows=[{'id':case,'provider':'fixture','model':name,'config':'direct','repetition':0,
               'provenance':'model-inference','correct':flag,'error':None,'parse_error':None,
               'latency_seconds':latency,'expected_answer':1,'parsed_answer':1,'raw_output':'fixture'}
              for case,flag in zip(ids,flags)]
        m={'status':'completed','provenance':'model-inference','dataset_sha256':'fixture-dataset',
           'case_ids':ids,'planned_attempts':len(rows),'completed_attempts':len(rows),
           'source_manifest':{'protocols':['direct'],'repetitions':1,'prompt_version':'fixture-v1',
                              'latency_measurement':'whole call','engine':'fixture','device':'cpu',
                              'platform':'fixture-platform','threads':1,'model_revision':'fixture-revision'}}
        save(directory/'manifest.json',m)
        (directory/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        sources.append({'name':name,'manifest_sha256':sha(directory/'manifest.json'),
                        'records_sha256':sha(directory/'records.jsonl')})
        combined.extend({**r,'source_run':name} for r in rows)
    output=root/'comparison';output.mkdir()
    save(output/'manifest.json',{'status':'completed','provenance':'model-inference','sources':sources,
          'dataset_sha256':'fixture-dataset','case_ids':ids,'prompt_version':'fixture-v1',
          'planned_attempts':len(combined),'completed_attempts':len(combined)})
    (output/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in combined))
    return output


def change_manifest(root,name,mutate):
    p=root/name/'manifest.json';m=json.loads(p.read_text());mutate(m);save(p,m)
    comp=root/'comparison/manifest.json';cm=json.loads(comp.read_text())
    next(s for s in cm['sources'] if s['name']==name)['manifest_sha256']=sha(p)
    save(comp,cm)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_quality_selects_from_observed_complete_data(self):
        out=fixture(self.root);result=select_configuration(out,{'min_accuracy':.7,'max_error_rate':0})
        self.assertEqual(result['status'],'selected');self.assertEqual(result['selected']['run'],'b')
        self.assertIsNone(result['selected']['mean_cost_usd']);self.assertFalse(result['llm_judge_used'])

    def test_no_eligible_quality_and_uncertainty_threshold(self):
        out=fixture(self.root)
        result=select_configuration(out,{'min_accuracy':.8,'quality_basis':'wilson_lower'})
        self.assertEqual(result['status'],'no_eligible_configuration')
        self.assertIsNone(result['selected'])

    def test_cost_unknown_never_zero(self):
        out=fixture(self.root);result=select_configuration(out,{'max_mean_cost_usd':0})
        self.assertEqual(result['status'],'no_eligible_configuration')
        self.assertTrue(all('cost_unknown' in c['rejection_reasons'] for c in result['candidates']))

    def test_time_limit_and_latency_objective(self):
        out=fixture(self.root);result=select_configuration(out,{'max_mean_latency_seconds':2.5,'objective':'latency'})
        self.assertEqual(result['selected']['run'],'a')
        self.assertIn('mean_latency_above_limit',result['candidates'][1]['rejection_reasons'])

    def test_incomparable_timers_are_not_ranked(self):
        out=fixture(self.root)
        change_manifest(self.root,'b',lambda m:m['source_manifest'].update(latency_measurement='generation only'))
        result=select_configuration(out,{'objective':'latency'})
        self.assertEqual(result['status'],'incomparable_latency_scopes');self.assertIsNone(result['selected'])
        result=select_configuration(out,{'objective':'latency','allowed_runs':['a']})
        self.assertEqual(result['selected']['run'],'a')

    def test_unknown_timer_is_explicit(self):
        out=fixture(self.root)
        for name in ['a','b']:change_manifest(self.root,name,lambda m:m['source_manifest'].pop('latency_measurement'))
        result=select_configuration(out,{'max_mean_latency_seconds':10})
        self.assertEqual(result['status'],'no_eligible_configuration')
        self.assertTrue(all('latency_scope_or_runtime_unknown' in c['rejection_reasons'] for c in result['candidates']))

    def test_tie_not_arbitrary(self):
        out=fixture(self.root,[('a',[True,False],2.0),('b',[True,False],3.0)])
        result=select_configuration(out,{})
        self.assertEqual(result['status'],'tied_configurations');self.assertIsNone(result['selected'])

    def test_source_tamper_rejected(self):
        out=fixture(self.root)
        with (self.root/'a/records.jsonl').open('a') as f:f.write('{}\n')
        with self.assertRaisesRegex(ValueError,'changed after comparison'):select_configuration(out,{})

    def test_combined_record_tamper_rejected(self):
        out=fixture(self.root);p=out/'records.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()]
        rows[0]['correct']=False;p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'differ from hash-verified'):select_configuration(out,{})

    def test_controls_rejected(self):
        out=fixture(self.root)
        change_manifest(self.root,'a',lambda m:m.update(provenance='control-not-llm'))
        with self.assertRaisesRegex(ValueError,'model-inference'):select_configuration(out,{})

    def test_missing_coverage_rejected(self):
        out=fixture(self.root)
        change_manifest(self.root,'a',lambda m:m.update(planned_attempts=3))
        with self.assertRaisesRegex(ValueError,'counts differ'):select_configuration(out,{})

    def test_bad_policy_rejected(self):
        for p in [{'min_accuracy':float('nan')},{'max_error_rate':2},{'max_mean_cost_usd':-1},
                  {'min_tasks':True},{'objective':'cheapest'},{'unknown':1},{'allowed_runs':[]}]:
            with self.assertRaises(ValueError):validate_policy(p)

    def test_min_tasks_and_unknown_run(self):
        out=fixture(self.root)
        self.assertEqual(select_configuration(out,{'min_tasks':3})['status'],'no_eligible_configuration')
        with self.assertRaisesRegex(ValueError,'unknown source run'):select_configuration(out,{'allowed_runs':['missing']})


if __name__=='__main__':unittest.main()
