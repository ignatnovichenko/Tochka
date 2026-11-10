#!/usr/bin/env python3
"""Validate exported GPU logs against fixed CPU probe; no inference is run."""
import argparse,hashlib,json,math,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parent

def read_run(path):
 manifest=json.loads((path/'manifest.json').read_text())
 raw=(path/'records.jsonl').read_bytes()
 assert manifest.get('completed_at_utc'),'Run not completed'
 assert hashlib.sha256(raw).hexdigest()==manifest['records_sha256'],'Raw records SHA mismatch'
 rows=[json.loads(s) for s in raw.decode().splitlines()]
 keys=[(r['id'],r['protocol']) for r in rows]
 assert len(keys)==len(set(keys)),'Duplicate records'
 for row in rows:
  assert row['parameters']==manifest['parameters'],'Parameters differ inside run'
  assert hashlib.sha256(row['rendered_prompt'].encode()).hexdigest()==row['prompt_sha256'],'Prompt SHA mismatch'
  perf=row['raw_native_counters']
  for name in ['n_eval','n_p_eval','t_eval_ms','t_p_eval_ms']:
   assert isinstance(perf[name],(int,float)) and math.isfinite(perf[name]) and perf[name]>0,(name,perf)
  assert perf['n_p_eval']==row['token_usage']['prompt_tokens'],'Prefill count differs from prompt tokens'
 return manifest,{(r['id'],r['protocol']):r for r in rows}
def aggregate(rows):
 seq=list(rows.values());p_tokens=sum(r['raw_native_counters']['n_p_eval'] for r in seq);p_ms=sum(r['raw_native_counters']['t_p_eval_ms'] for r in seq);d_tokens=sum(r['raw_native_counters']['n_eval'] for r in seq);d_ms=sum(r['raw_native_counters']['t_eval_ms'] for r in seq);o_tokens=sum(r['token_usage']['completion_tokens'] for r in seq);wall=sum(r['latency_seconds'] for r in seq)
 return {'n_inferences':len(seq),'native_decode_tokens':d_tokens,'native_decode_ms':d_ms,'decode_tokens_per_second_native':d_tokens*1000/d_ms,'native_prefill_tokens':p_tokens,'native_prefill_ms':p_ms,'prefill_tokens_per_second_native':p_tokens*1000/p_ms,'output_tokens':o_tokens,'completion_wall_seconds':wall,'output_tokens_per_second_end_to_end':o_tokens/wall}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('gpu',type=pathlib.Path);ap.add_argument('--offload-log',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
 cm,cr=read_run(ROOT/'speed_probe');gm,gr=read_run(a.gpu)
 assert set(cr)==set(gr) and len(cr)==2,'GPU must match exact two CPU probe requests'
 for f in ['model_id','model_revision','quantization','engine','engine_version','n_ctx','n_batch','n_ubatch','n_threads','parameters']:
  assert cm.get(f)==gm.get(f),('Configuration mismatch',f,cm.get(f),gm.get(f))
 assert gm.get('device')=='cuda' and gm.get('n_gpu_layers')==-1
 per_request=[]
 for key,cpu in cr.items():
  gpu=gr[key]
  assert cpu['rendered_prompt']==gpu['rendered_prompt'] and cpu['prompt_sha256']==gpu['prompt_sha256'],'Input prompt changed'
  assert cpu['parameters']==gpu['parameters'],'Generation configuration changed'
  assert cpu['token_usage']['prompt_tokens']==gpu['token_usage']['prompt_tokens']
  per_request.append({'id':key[0],'protocol':key[1],'prompt_sha256':cpu['prompt_sha256'],'prompt_tokens':cpu['token_usage']['prompt_tokens'],'outputs_equal':cpu['raw_output']==gpu['raw_output'],'cpu_output_tokens':cpu['token_usage']['completion_tokens'],'gpu_output_tokens':gpu['token_usage']['completion_tokens'],'cpu_decode_tokens_per_second_native':1000*cpu['raw_native_counters']['n_eval']/cpu['raw_native_counters']['t_eval_ms'],'gpu_decode_tokens_per_second_native':1000*gpu['raw_native_counters']['n_eval']/gpu['raw_native_counters']['t_eval_ms']})
 log=a.offload_log.read_text(errors='replace');matches=[(int(x),int(y)) for x,y in re.findall(r'offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU',log,re.I)]
 assert any(x==y and x>0 for x,y in matches),'Full GPU offload not established by native log'
 cpu=aggregate(cr);gpu=aggregate(gr)
 out={'checks':{'record_hashes_verified':True,'two_request_keys_match':True,'all_text_prompts_byte_identical':True,'generation_parameters_identical':True,'model_revision_and_quantization_match':True,'prefill_counts_match_input_tokens':True,'complete_gpu_offload_logged':matches,'offload_log_sha256':hashlib.sha256(a.offload_log.read_bytes()).hexdigest()},'cpu':cpu,'gpu':gpu,'ratios':{'gpu_vs_cpu_native_decode':gpu['decode_tokens_per_second_native']/cpu['decode_tokens_per_second_native'],'gpu_vs_cpu_native_prefill':gpu['prefill_tokens_per_second_native']/cpu['prefill_tokens_per_second_native'],'cpu_vs_gpu_completion_wall_time':cpu['completion_wall_seconds']/gpu['completion_wall_seconds']},'per_request':per_request,'limits':['Only two prompts from one task, not a large performance benchmark.','Different machines and CPU/CUDA builds. Same source version does not make hardware or build flags identical.','Native decode token rate excludes prefill and some host/sampling overhead. End-to-end rate is a separate metric.','No separate warm-up. OS cache not cleared. Free Colab availability/performance are not guaranteed.','Greedy CPU/GPU arithmetic may produce different outputs; matched prompts do not prove identical generation paths.','Performance does not establish answer quality or office-task usefulness.']}
 a.output.mkdir(exist_ok=True,parents=True);(a.output/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
