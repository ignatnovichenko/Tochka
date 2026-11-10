#!/usr/bin/env python3
"""Read-only audit of current delivery, no model calls or network."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nir_eval.business_selection import select_business_run
from nir_eval.business.schema import load_jsonl, digest
from nir_eval.experiment import compare_runs, read
from nir_eval.opencode import FREE_MODELS

ROOT=Path(__file__).resolve().parents[1]
def audit_run(name):
    root=ROOT/'runs'/name
    em=read(root/'experiment-manifest.json')
    assert em['state']=='completed'
    for filename, expected in em['input_hashes'].items():
        assert hashlib.sha256((root/'inputs'/filename).read_bytes()).hexdigest()==expected
    policy=read(root/'inputs/experiment.json')['policy']
    result=compare_runs([root/x for x in em['runs']], policy)
    assert result==read(root/'comparison.json')
    solver_count=judge_count=unknown=0
    sessions=set()
    for run in em['runs']:
        r=root/run
        records=load_jsonl(r/'records.jsonl')
        solver=load_jsonl(r/'solver_records.jsonl')
        raw=[json.loads(l) for l in (r/'judge_raw.jsonl').read_text().splitlines()]
        responses={x['id']:x['response'] for x in load_jsonl(r/'responses.jsonl')}
        for x in solver:
            assert x['model'] in FREE_MODELS
            # Solver payload is task + context, not gold or rubric.
            assert set(json.loads(x['messages'][1]['content']))=={'task','context'}
            if x['error'] is None:
                solver_count+=1
                sid=x['provider_metadata']['session_id'];assert sid not in sessions;sessions.add(sid)
        for x in raw:
            judge_count+=1
            sid=x['metadata']['session_id'];assert sid not in sessions;sessions.add(sid)
        for x in records:
            assert x['judge']['model'] in FREE_MODELS and x['judge']['model']!=x['model']
            if x['judge']['status']!='evaluated':
                assert x['score'] is None and x['passed'] is None
                unknown+=1
            else:
                assert x['candidate_sha256']==digest(responses[x['id']])
                assert 0<=x['score']<=1
    impossible=compare_runs([root/x for x in em['runs']],{'min_mean_judge_score':1,'max_mean_solver_latency_seconds':0})
    assert impossible['status']=='no_eligible_configuration'
    return {'experiment':name,'successful_solver_responses':solver_count,'successful_judge_responses':judge_count,
            'unknown_judgements':unknown,'selection_status':result['status'],
            'impossible_policy_status':impossible['status'],'unique_successful_sessions':len(sessions)}

def calibration(name):
    root=ROOT/'runs'/name
    summary=read(root/'summary.json');rows=load_jsonl(root/'records.jsonl')
    assert summary['total']==len(rows)==8
    known=[r for r in rows if r['judge']['status']=='evaluated']
    assert summary['evaluated']==len(known)
    assert summary['unknown']==len(rows)-len(known)
    for r in known:
        assert r['pass_agreement']==(r['passed']==r['expected_pass'])
        detected=any(c['detected'] for c in r['judge']['critical_errors'])
        assert r['critical_agreement']==(detected==r['expected_critical'])
    return {'calibration':name,'evaluated':len(known),'total':len(rows),'pass_agreement':summary['pass_agreement'],'critical_agreement':summary['critical_agreement']}

if __name__=='__main__':
    print(json.dumps({'verified':True,'model_calls':0,
          'runs':[audit_run('current_business'),audit_run('final_business')],
          'calibrations':[calibration('current_calibration_02'),calibration('current_calibration_final'),calibration('final_calibration')]},ensure_ascii=False,indent=2))
