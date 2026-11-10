#!/usr/bin/env python3
"""Independent audit of newly executed native MVP runs; no model calls/imports."""
import dataclasses,datetime,hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent;MVP=ROOT.parent.parent/'mvp';sys.path.insert(0,str(ROOT))
from source_snapshot.datasets import load_finqa
from source_snapshot.prompts import build_messages
from source_snapshot.runner import fingerprint,code_hash
from source_snapshot.scoring import score_output
cases={c.id:c for c in load_finqa(MVP/'data/finqa_dev.json')}
expected_ids=json.loads((ROOT.parent/'run_01/case_ids.json').read_text())
report={'audited_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'audit_runs_inference':False,'runs':[]}
all_rows=[]
for protocol in ['direct','cot']:
 folder=MVP/'runs'/('e2e_qwen_direct' if protocol=='direct' else 'e2e_qwen_cot_recovered');m=json.loads((folder/'manifest.json').read_text());raw=(folder/'records.jsonl').read_bytes();rows=[json.loads(s) for s in raw.decode().splitlines()]
 assert m['status']=='completed' and m['provenance']=='model-inference'
 assert m['config']['provider']['type']=='transformers-local'
 assert not m.get('imported',False) and 'source_manifest' not in m
 assert m['planned_attempts']==m['completed_attempts']==len(rows)==12
 assert m['case_ids']==expected_ids
 assert m['config']['provider']['revision']=='7ae557604adf67be50417f59c2c2f167def9a775'
 assert m['code_sha256']==code_hash(),'Frozen source snapshot differs from the code hash recorded by native run'
 assert len({r['id'] for r in rows})==12 and {r['id'] for r in rows}==set(expected_ids)
 for r in rows:
  c=cases[r['id']];assert r['config']==protocol;assert r['provider']=='local_transformers'
  assert r['provenance']=='model-inference' and not r.get('imported',False)
  assert r['prompt_sha256']==fingerprint(build_messages(c,protocol))
  assert r['error'] is None and r['raw_output']
  assert r['token_usage']['completion_tokens']>0 and r['token_usage']['prompt_tokens']>0
  assert r['provider_metadata']['parameter_count']==494032768
  assert r['provider_metadata']['torch_version']=='2.6.0+cpu' and r['provider_metadata']['transformers_version']=='4.48.3'
  assert r['provider_metadata']['model_revision']=='7ae557604adf67be50417f59c2c2f167def9a775'
  score=dataclasses.asdict(score_output(r['raw_output'],protocol,c.expected_answer,c.expected_program,c.table))
  assert all(r.get(k)==v for k,v in score.items()),(r['id'],score)
 assert all((folder/name).is_file() and (folder/name).stat().st_size>0 for name in ['summary.json','report.html','report.md','summary.csv'])
 report['runs'].append({'name':folder.name,'new_model_inference':True,'imported':False,'n':len(rows),'correct':sum(r['correct'] for r in rows),'parseable':sum(r['parse_error'] is None for r in rows),'runtime_errors':sum(r['error'] is not None for r in rows),'mean_latency_seconds':sum(r['latency_seconds'] for r in rows)/len(rows),'input_tokens':sum(r['token_usage']['prompt_tokens'] for r in rows),'output_tokens':sum(r['token_usage']['completion_tokens'] for r in rows),'records_sha256':hashlib.sha256(raw).hexdigest(),'manifest_sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),'code_sha256':m['code_sha256'],'actual_backend':'torch2.6.0+cpu/transformers4.48.3','scores_independently_recomputed':True,'created_at':m['created_at'],'completed_at':m['completed_at']})
 all_rows.extend(rows)
comp=MVP/'runs/e2e_qwen_verified_comparison';cm=json.loads((comp/'manifest.json').read_text());combined=[json.loads(s) for s in (comp/'records.jsonl').read_text().splitlines()]
assert cm['status']=='completed' and cm['provenance']=='model-inference';assert cm['planned_attempts']==cm['completed_attempts']==len(combined)==24
assert cm['model_inference_performed_by_comparison'] is False
assert {r['config'] for r in combined}=={'direct','cot'}
original={(r['id'],r['config']):r for r in all_rows}
for row in combined:
 clean={k:v for k,v in row.items() if k!='source_run'};assert clean==original[(row['id'],row['config'])]
report['comparison']={'rows':24,'sources':2,'unchanged_original_attempts':True,'additional_inference':False,'report_html':str(comp/'report.html')}
partial=MVP/'runs/e2e_qwen_cot'
partial_raw=(partial/'records.jsonl').read_bytes()
report['excluded_unfinished_run']={'path':str(partial),'records_saved':len(partial_raw.decode().splitlines()),'records_sha256':hashlib.sha256(partial_raw).hexdigest(),'manifest_status':json.loads((partial/'manifest.json').read_text())['status'],'reason':'Execution transport reset; session ID became unknown. Existing lock and records retained. A fresh full native cot run was executed in a new directory. This incomplete run is not imported, merged, or counted in the comparison.','process_state_after_transport_reset':'uncertain'}
report['interpretation']='Two newly executed native MVP runs using one real model and two prompt protocols, followed by native scoring/reporting and native compare. Not two models; prior imported experiments remain separate.'
(ROOT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
