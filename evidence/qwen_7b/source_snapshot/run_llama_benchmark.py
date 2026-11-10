#!/usr/bin/env python3
"""Run fixed FinQA pilot using pinned official Qwen GGUF and local llama.cpp."""
import argparse,datetime,hashlib,json,os,pathlib,platform,shutil,sys,time,resource
ROOT=pathlib.Path(__file__).resolve().parent;MVP=ROOT.parent/'mvp';sys.path.insert(0,str(MVP))
from nir_eval.datasets import load_finqa,select_cases
from nir_eval.prompts import build_messages,PROMPT_VERSION

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def render(messages):return ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n'
def main():
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=12);p.add_argument('--seed',type=int,default=42);p.add_argument('--protocols',nargs='+',default=['direct','cot']);p.add_argument('--max-new-tokens',type=int,default=192);p.add_argument('--threads',type=int,default=4);p.add_argument('--outdir',default='run_02');a=p.parse_args()
 out=ROOT/a.outdir;out.mkdir(exist_ok=True)
 if (out/'records.jsonl').exists():raise RuntimeError('Refusing overwrite of inference records')
 src=ROOT.parent/'datasets/finqa/official/dataset/dev.json';cases=select_cases(load_finqa(src),limit=a.limit,seed=a.seed)
 dump(out/'case_ids.json',[c.id for c in cases]);dump(out/'gold.json',[{'id':c.id,'expected_answer':c.expected_answer,'expected_program':c.expected_program} for c in cases])
 with (out/'inputs.jsonl').open('w') as f:
  for c in cases:f.write(json.dumps(c.public_input(),ensure_ascii=False)+'\n')
 import llama_cpp
 from llama_cpp import Llama
 meta=json.loads((ROOT/'qwen7b/model_metadata.json').read_text());gguf=ROOT/'qwen7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf'
 assert (ROOT/'qwen7b/model_manifest.json').exists(),'Download and verify both shards first'
 params={'max_tokens':a.max_new_tokens,'temperature':0.0,'top_p':1.0,'top_k':0,'repeat_penalty':1.1,'seed':a.seed,'echo':False,'stop':['<|im_end|>','<|endoftext|>']}
 manifest={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inference_type':'real_local_model_inference','model_id':meta['id'],'model_revision':meta['sha'],'model_license':meta['cardData']['license'],'model_quantization':'Q4_K_M','device':'cpu','engine':'llama-cpp-python','llama_cpp_python_version':llama_cpp.__version__,'python':sys.version,'cpu_count':os.cpu_count(),'threads':a.threads,'n_ctx':4096,'n_batch':512,'n_ubatch':512,'use_mmap':True,'n_gpu_layers':0,'platform':platform.platform(),'dataset':'FinQA','split':'dev','dataset_sha256':sha(src),'sample_rule':f'first {a.limit} after sorting SHA256(str({a.seed})+":"+id)','n_cases':len(cases),'seed':a.seed,'prompt_version':PROMPT_VERSION,'prompts_py_sha256':sha(MVP/'nir_eval/prompts.py'),'parameters':params,'protocols':a.protocols,'repetitions':1,'paid_api_calls':0,'inputs_truncated':False,'latency_measurement':'time.perf_counter around entire llama.cpp completion call: includes prompt tokenization, prefill, token generation and conversion to string; excludes model loading and scoring. Not directly comparable to run_01 generate-only timer.','warmup':'No separate warm-up; first observed inference retained. KV cache reset before every measured completion.','limitations':['Small development pilot, not a leaderboard or test-set estimate.','Cross-model difference changes parameter count, quantization and engine, not a single causal factor.','Published benchmark may have training contamination.','Greedy single repetition does not estimate sampling variation.','Hardware execution timings are specific to this shared CPU environment.','One original annotation uses tangible assets while the gold corresponds to identifiable net assets; original gold remains unchanged.']}
 dump(out/'manifest.json',manifest)
 start=time.perf_counter();model=Llama(model_path=str(gguf),n_ctx=4096,n_threads=a.threads,n_threads_batch=a.threads,n_batch=512,n_ubatch=512,n_gpu_layers=0,seed=a.seed,use_mmap=True,verbose=False);manifest['model_loading_seconds']=time.perf_counter()-start;manifest['gguf_metadata']=model.metadata;dump(out/'manifest.json',manifest)
 print('loaded',manifest['model_loading_seconds'],flush=True)
 # Cross-check exact byte prompts against the smaller-model run when same cases exist.
 prior={ (r['id'],r['protocol']):r['rendered_prompt'] for r in map(json.loads,(ROOT/'run_01/records.jsonl').read_text().splitlines())}
 with (out/'records.jsonl').open('w',buffering=1) as log:
  for i,c in enumerate(cases):
   order=a.protocols if i%2==0 else list(reversed(a.protocols))
   for protocol in order:
    messages=build_messages(c,protocol);prompt=render(messages)
    if (c.id,protocol) in prior:assert prompt==prior[(c.id,protocol)],'Chat template mismatch'
    prompt_token_count=len(model.tokenize(prompt.encode(),add_bos=False,special=True));assert prompt_token_count+a.max_new_tokens<=4096,'No silent truncation'
    model.reset()
    record={'id':c.id,'case_id':c.id,'config':protocol,'protocol':protocol,'repetition':0,'provider':'local_llama_cpp','model':meta['id'],'model_revision':meta['sha'],'parameters':params,'messages':messages,'rendered_prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'raw_output':None,'error':None,'token_usage':{'prompt_tokens':prompt_token_count},'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    start=time.perf_counter()
    try:
     result=model(prompt,**params);record['latency_seconds']=time.perf_counter()-start;record['raw_output']=result['choices'][0]['text'];record['token_usage']=result['usage'];record['finish_reason']=result['choices'][0]['finish_reason'];record['hit_token_limit']=record['finish_reason']=='length'
    except Exception as e:
     record['latency_seconds']=time.perf_counter()-start;record['error']=type(e).__name__+': '+str(e);record['hit_token_limit']=False
    log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush();print(i+1,protocol,c.id,record['token_usage'],round(record['latency_seconds'],3),repr(record['raw_output']),record['error'],flush=True)
 manifest['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();manifest['records_sha256']=sha(out/'records.jsonl');manifest['peak_process_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;manifest['peak_rss_method']='resource.getrusage(RUSAGE_SELF).ru_maxrss, Linux KiB, includes loading and all inference in this Python process; not a container RAM measurement';dump(out/'manifest.json',manifest)
if __name__=='__main__':main()
