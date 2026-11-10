#!/usr/bin/env python3
"""Preserve a compact inventory of all current runs without changing raw records."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/current'
NAMES=['current_calibration','current_calibration_02','current_calibration_final','current_business','final_calibration','final_business']

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []

def native_total(meta):
    result={'input':0,'output':0,'reasoning':0,'cache_read':0,'cache_write':0}
    for m in meta:
        t=m.get('native_tokens') or {}
        for key in ['input','output','reasoning']:
            v=t.get(key)
            if type(v) is int:result[key]+=v
        for key in ['read','write']:
            v=(t.get('cache') or {}).get(key)
            if type(v) is int:result['cache_'+key]+=v
    return result

stats=[]
for name in NAMES:
    root=ROOT/'runs'/name
    groups=list(root.glob('solver-*')) if 'business' in name else [root]
    for group in groups:
        records=rows(group/'records.jsonl')
        raw=rows(group/'judge_raw.jsonl') if 'business' in name else rows(group/'raw.jsonl')
        solver=rows(group/'solver_records.jsonl')
        judged=[r for r in records if r['judge']['status']=='evaluated']
        sm=[s.get('provider_metadata',{}) for s in solver if s.get('error') is None]
        jm=[j.get('metadata',{}) for j in raw]
        costs=[m.get('cost_usd') for m in sm+jm]
        stats.append({'run':name,'configuration':group.name if group!=root else 'calibration',
            'planned_records':8 if 'calibration' in name and name!='current_calibration' else len(records),
            'saved_records':len(records),'successful_solver_calls':len(sm),
            'successful_judge_calls':len(raw),'valid_judgements':len(judged),
            'unknown_judgements':len(records)-len(judged),
            'mean_judge_score':sum(r['score'] for r in judged)/len(judged) if judged else None,
            'known_response_cost_sum_usd':sum(costs) if costs and all(c is not None for c in costs) else None,
            'cost_scope':'successful responses only; missing/failed requests not imputed',
            'solver_native_tokens':native_total(sm),'judge_native_tokens':native_total(jm),
            'solver_call_seconds':sum(s.get('latency_seconds',0) for s in solver) if solver else None,
            'judge_call_seconds':sum(r['judge'].get('latency_seconds') or 0 for r in records) if records else None,
            'path':str(group.relative_to(ROOT))})
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'run_statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
columns=list(stats[0])
with (OUT/'run_statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,columns);w.writeheader()
    for r in stats:
        w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v for k,v in r.items()})
print(json.dumps({'preserved_run_groups':len(stats),'successful_solver_calls':sum(x['successful_solver_calls'] for x in stats),'successful_judge_calls':sum(x['successful_judge_calls'] for x in stats),'model_calls_performed':0}))
# Flat per-attempt export for business analysis in Excel; JSONL remains canonical.
attempts=[]
for name in NAMES:
    root=ROOT/'runs'/name
    groups=list(root.glob('solver-*')) if 'business' in name else [root]
    for group in groups:
        solvers={x['id']:x for x in rows(group/'solver_records.jsonl')}
        for r in rows(group/'records.jsonl'):
            s=solvers.get(r['id'],{})
            jm=[a.get('provider_metadata',{}) for a in r['judge'].get('attempts',[])]
            if not jm and r['judge'].get('provider_metadata'):jm=[r['judge']['provider_metadata']]
            sm=s.get('provider_metadata',{})
            row={'run':name,'configuration':r.get('configuration','calibration'),'case_id':r['id'],
                 'label_id':r.get('label_id'),'solver_model':r.get('model'), 'judge_model':r['judge']['model'],
                 'solver_error':s.get('error'),'judge_status':r['judge']['status'],
                 'judge_error':r['judge'].get('reason_code'),'score':r['score'],'passed':r['passed'],
                 'solver_seconds':s.get('latency_seconds'),'judge_seconds':r['judge'].get('latency_seconds'),
                 'solver_cost_usd':sm.get('cost_usd'),'judge_cost_usd':r['judge'].get('cost_usd'),
                 'total_cost_usd':r.get('total_cost_usd'),
                 'solver_session_id':sm.get('session_id'),
                 'judge_session_ids':';'.join(m.get('session_id','') for m in jm)}
            for prefix,metadata in [('solver', [sm] if sm else []), ('judge',jm)]:
                totals=native_total(metadata)
                for k,v in totals.items():row[prefix+'_native_'+k]=v if metadata else None
            attempts.append(row)
with (OUT/'attempt_statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,list(attempts[0]));w.writeheader();w.writerows(attempts)
