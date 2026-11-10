#!/usr/bin/env python3
"""Real local inference. No network, credentials, replay, or golden data in prompts.
Run with this directory's .venv/bin/python after download_model.py.
"""
import argparse,datetime,hashlib,json,os,pathlib,platform,random,sys,time
ROOT=pathlib.Path(__file__).resolve().parent
MVP=ROOT.parent/'mvp'
sys.path.insert(0,str(MVP))
from nir_eval.datasets import load_finqa,select_cases
from nir_eval.prompts import build_messages,PROMPT_VERSION

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=12);p.add_argument('--seed',type=int,default=42);p.add_argument('--protocols',nargs='+',default=['direct','cot']);p.add_argument('--max-new-tokens',type=int,default=192);p.add_argument('--threads',type=int,default=4);p.add_argument('--outdir',default='run_01');a=p.parse_args()
 out=ROOT/a.outdir;out.mkdir(exist_ok=True)
 if (out/'records.jsonl').exists():raise RuntimeError('Refusing to overwrite existing records.jsonl')
 source=ROOT.parent/'datasets/finqa/official/dataset/dev.json'
 cases=select_cases(load_finqa(source),limit=a.limit,seed=a.seed)
 # Freeze public model inputs independently of evaluator/gold.
 with (out/'inputs.jsonl').open('w') as f:
  for c in cases:f.write(json.dumps(c.public_input(),ensure_ascii=False)+'\n')
 dump(out/'case_ids.json',[c.id for c in cases])
 dump(out/'gold.json',[{'id':c.id,'expected_answer':c.expected_answer,'expected_program':c.expected_program} for c in cases])
 import torch,transformers
 from transformers import AutoModelForCausalLM,AutoTokenizer
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(a.seed);random.seed(a.seed)
 torch.use_deterministic_algorithms(True)
 meta=json.loads((ROOT/'model_metadata.json').read_text())
 params={'do_sample':False,'max_new_tokens':a.max_new_tokens,'num_beams':1,'use_cache':True}
 manifest={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inference_type':'real_local_model_inference','model_id':meta['id'],'model_revision':meta['sha'],'model_license':meta['cardData']['license'],'model_dtype':'float32','device':'cpu','attention_implementation':'sdpa','python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'cpu_count':os.cpu_count(),'torch_threads':a.threads,'torch_interop_threads':1,'platform':platform.platform(),'dataset':'FinQA','split':'dev','dataset_sha256':sha(source),'sample_rule':f'first {a.limit} after sorting SHA256(str({a.seed})+":"+id)','n_cases':len(cases),'seed':a.seed,'prompt_version':PROMPT_VERSION,'prompts_py_sha256':sha(MVP/'nir_eval/prompts.py'),'parameters':params,'protocols':a.protocols,'repetitions':1,'paid_api_calls':0,'inputs_truncated':False,'limitations':['Small development pilot, not a leaderboard or test-set estimate.','One small open-weight model; comparison varies prompt protocol, not model family.','Published benchmark may have training contamination.','Greedy single repetition does not estimate sampling variation.','Hardware execution timings are specific to this shared CPU environment.']}
 dump(out/'manifest.json',manifest)
 start=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=False);model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='sdpa');model.eval()
 manifest['model_loading_seconds']=time.perf_counter()-start;manifest['model_parameter_count']=sum(p.numel() for p in model.parameters());dump(out/'manifest.json',manifest)
 print('loaded',manifest['model_loading_seconds'],manifest['model_parameter_count'],flush=True)
 # Alternating protocol order per case reduces simple warmup/order bias.
 with (out/'records.jsonl').open('w',buffering=1) as log:
  for i,c in enumerate(cases):
   protocols=a.protocols if i%2==0 else list(reversed(a.protocols))
   for protocol in protocols:
    messages=build_messages(c,protocol)
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
    record={'id':c.id,'case_id':c.id,'config':protocol,'protocol':protocol,'repetition':0,'provider':'local_transformers','model':meta['id'],'model_revision':meta['sha'],'parameters':params,'messages':messages,'rendered_prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'raw_output':None,'error':None,'token_usage':{'prompt_tokens':inputs['input_ids'].shape[1]},'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    started=time.perf_counter()
    try:
     with torch.inference_mode():
      result=model.generate(**inputs,**params,pad_token_id=tokenizer.eos_token_id)
     record['latency_seconds']=time.perf_counter()-started
     generated=result[0,inputs['input_ids'].shape[1]:]
     record['raw_output']=tokenizer.decode(generated,skip_special_tokens=True)
     record['token_usage']['completion_tokens']=generated.shape[0]
     record['hit_token_limit']=generated.shape[0]>=a.max_new_tokens
    except Exception as e:
     record['latency_seconds']=time.perf_counter()-started;record['error']=type(e).__name__+': '+str(e)
    log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush()
    print(i+1,protocol,c.id,record['token_usage'],round(record['latency_seconds'],3),repr(record['raw_output']),record['error'],flush=True)
 manifest['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();manifest['records_sha256']=sha(out/'records.jsonl');dump(out/'manifest.json',manifest)
if __name__=='__main__':main()
