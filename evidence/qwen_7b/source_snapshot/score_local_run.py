#!/usr/bin/env python3
"""Score completed real local inference records. This does not invoke any LLM."""
import argparse,collections,dataclasses,hashlib,json,math,pathlib,statistics,sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mvp'))
from nir_eval.datasets import load_finqa,select_cases
from nir_eval.prompts import build_messages
from nir_eval.scoring import score_output,parse_object,program_tokens
sys.path.insert(0,str(ROOT.parent/'datasets/finqa'))
from prepare_and_check import load_evaluator

def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def wilson(k,n,z=1.959963984540054):
 p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [max(0,center-half),min(1,center+half)]
def main():
 p=argparse.ArgumentParser();p.add_argument('run',nargs='?',default='run_01');a=p.parse_args();out=ROOT/a.run
 manifest=json.loads((out/'manifest.json').read_text());assert 'completed_at_utc' in manifest,'Do not score an unfinished run.'
 records=[json.loads(s) for s in (out/'records.jsonl').read_text().splitlines()]
 ids=json.loads((out/'case_ids.json').read_text())
 cases=select_cases(load_finqa(ROOT.parent/'datasets/finqa/official/dataset/dev.json'),ids=ids);by_id={c.id:c for c in cases}
 expected={(c.id,protocol,0) for c in cases for protocol in manifest['protocols']}
 keys=[(r['id'],r['protocol'],r['repetition']) for r in records]
 assert len(keys)==len(set(keys)) and set(keys)==expected,'Incomplete/duplicate/mismatched records.'
 evaluator,shim=load_evaluator();gold_checks=[]
 for c in cases:
  invalid,ans=evaluator.eval_program(evaluator.program_tokenization(c.expected_program),c.table)
  gold_checks.append({'id':c.id,'official_gold_program_valid':invalid==0,'official_gold_program_answer':ans,'expected':c.expected_answer,'match':invalid==0 and ans==c.expected_answer})
 assert all(x['match'] for x in gold_checks)
 scored=[]
 for r in records:
  c=by_id[r['id']]
  assert r['messages']==build_messages(c,r['protocol']),'Prompt differs from frozen recipe.'
  assert set(json.loads(r['messages'][1]['content'].split('\n',1)[1]))=={'question','pre_text','table','post_text'}
  score=score_output(r['raw_output'] or '',r['protocol'],c.expected_answer,c.expected_program,c.table)
  # Separate JSON syntax from parseability and strict requested answer schema.
  try:
   obj=json.loads(r['raw_output']);json_valid=isinstance(obj,dict)
   val=obj.get('answer') if json_valid else None
   schema_valid=json_valid and (type(val) in (int,float) or val in ('yes','no'))
   if r['protocol']=='cot':schema_valid=schema_valid and isinstance(obj.get('reasoning'),str)
   if r['protocol']=='program':schema_valid=json_valid and isinstance(obj.get('program'),str)
  except (ValueError,TypeError):json_valid=False;schema_valid=False
  official_program_equal=None; official_execution_match=None; official_matches_safe_interpreter=None
  if r['protocol']=='program':
   official_program_equal=False;official_execution_match=False;official_matches_safe_interpreter=False
   if score.parse_error is None:
    tokens=program_tokens(score.predicted_program);invalid,value=evaluator.eval_program(tokens,c.table)
    official_execution_match=invalid==0 and value==c.expected_answer
    official_matches_safe_interpreter=invalid==0 and value==score.parsed_answer
    official_program_equal=bool(evaluator.equal_program(evaluator.program_tokenization(c.expected_program),tokens))
  scored.append({k:r.get(k) for k in ['id','protocol','repetition','model','model_revision','latency_seconds','token_usage','raw_output','error','hit_token_limit']}|dataclasses.asdict(score)|{'expected_answer':c.expected_answer,'expected_program':c.expected_program,'json_object_valid':json_valid,'requested_answer_schema_valid':schema_valid,'official_program_equal':official_program_equal,'official_execution_match':official_execution_match,'official_matches_safe_interpreter':official_matches_safe_interpreter})
 summary={'inference_type':manifest['inference_type'],'n_cases':len(cases),'n_inferences':len(records),'numeric_match_rule':'Accept one JSON object, normalize numeric strings and yes/no; round(predicted,5) == canonical FinQA qa.exe_ans. Do not rescale numeric percent answers heuristically. Failed parsing and runtime errors count as incorrect.','comparison_axis':('Model-generated FinQA DSL program executed by a bounded local calculator.' if manifest['protocols']==['program'] else 'Prompt protocol only: direct answer versus concise calculation plus answer (cot). Not two models, not a hidden reasoning switch.'),'generation_parameters':manifest['parameters'],'protocols':{},'controls':{'gold_programs_verified_by_official_evaluator':sum(x['match'] for x in gold_checks),'gold_programs_total':len(gold_checks),'oracle_json_scoring_correct':sum(score_output(json.dumps({'answer':c.expected_answer}),'direct',c.expected_answer,c.expected_program,c.table).correct for c in cases),'invalid_json_scoring_correct':sum(score_output('not json','direct',c.expected_answer,c.expected_program,c.table).correct for c in cases),'controls_are_llm_results':False},'prompt_integrity':{'all_inputs_match_recipe':True,'public_input_fields_only':True,'id_omitted_from_model_context':True},'limitations':manifest['limitations']}
 for protocol in manifest['protocols']:
  group=[r for r in scored if r['protocol']==protocol];n=len(group);k=sum(r['correct'] for r in group);lat=[r['latency_seconds'] for r in group]
  summary['protocols'][protocol]={'n':n,'correct':k,'accuracy':k/n,'accuracy_wilson95_interval':wilson(k,n),'parseable_answers':sum(r['parse_error'] is None for r in group),'json_object_valid':sum(r['json_object_valid'] for r in group),'requested_answer_schema_valid':sum(r['requested_answer_schema_valid'] for r in group),'runtime_errors':sum(r['error'] is not None for r in group),'token_limit_hits':sum(r['hit_token_limit'] for r in group),'latency_mean_seconds':statistics.mean(lat),'latency_median_seconds':statistics.median(lat),'latency_min_seconds':min(lat),'latency_max_seconds':max(lat),'prompt_tokens_total':sum(r['token_usage']['prompt_tokens'] for r in group),'completion_tokens_total':sum(r['token_usage'].get('completion_tokens',0) for r in group)}
  if protocol=='program':
   summary['protocols'][protocol].update(official_execution_correct=sum(r['official_execution_match'] for r in group),official_program_correct=sum(r['official_program_equal'] for r in group),official_program_accuracy=sum(r['official_program_equal'] for r in group)/n,official_matches_safe_interpreter_on_valid_programs=all(r['official_matches_safe_interpreter'] for r in group if r['parse_error'] is None))
 if set(manifest['protocols'])=={'direct','cot'}:
  d={r['id']:r['correct'] for r in scored if r['protocol']=='direct'};c={r['id']:r['correct'] for r in scored if r['protocol']=='cot'}
  b=sum(d[i] and not c[i] for i in ids);cc=sum(c[i] and not d[i] for i in ids);n=b+cc
  summary['paired_comparison']={'direct_only_correct':b,'cot_only_correct':cc,'both_correct':sum(d[i] and c[i] for i in ids),'both_wrong':sum(not d[i] and not c[i] for i in ids),'mcnemar_exact_two_sided_p':min(1.0,2*sum(math.comb(n,j) for j in range(min(b,cc)+1))/(2**n)) if n else 1.0,'claim':'Descriptive pilot only; small sample cannot establish general superiority.'}
 with (out/'scored_records.jsonl').open('w') as f:
  for r in scored:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 dump(out/'summary.json',summary);dump(out/'gold_control_checks.json',gold_checks)
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
