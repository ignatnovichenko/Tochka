"""User case → real solver → neural judge → report. No numeric answer scorer."""
from datetime import datetime,timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

from ..providers import make_provider,ProviderError
from ..provenance import source_hashes
from .schema import ValidationError,canonical,digest,strict_json,import_inputs,fields,identifier,string
from .evaluation import build_solver_messages,evaluate_matrix
from .providers import make_judge,validate_config
from .reporting import write_report


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()


def resolve_provider(config,base):
    p=dict(config)
    if any(k in p for k in {'api_key','token','password','secret'}):
        raise ValidationError('Credentials must use environment variables')
    if p.get('type') not in {'transformers-local','openai-compatible','command','opencode'}:
        raise ValidationError('Business solver requires a real generation provider')
    for field in ['model_path','cwd']:
        if field in p:p[field]=str((base/p[field]).resolve())
    return p


def load_solver_config(path):
    path=Path(path).resolve();config=strict_json(path.read_text())
    fields(config,('provider','model','configurations'),('repetitions','max_calls','label'))
    string(config['model'],200)
    if not isinstance(config['provider'],dict):raise ValidationError('Missing solver provider object')
    config['provider']=resolve_provider(config['provider'],path.parent)
    configurations=config['configurations']
    if not isinstance(configurations,list) or not configurations:raise ValidationError('Missing configurations')
    seen=set()
    for cfg in configurations:
        fields(cfg,('name',),('parameters','instruction'))
        identifier(cfg['name'])
        if cfg['name'] in seen:raise ValidationError('Duplicate solver configuration')
        seen.add(cfg['name'])
        if not isinstance(cfg.get('parameters',{}),dict):raise ValidationError('Invalid generation parameters')
        if 'instruction' in cfg:string(cfg['instruction'],10_000)
    for field,default in [('repetitions',1),('max_calls',100)]:
        if type(config.get(field,default)) is not int or not 1<=config.get(field,default)<=10000:
            raise ValidationError('Invalid call or repetition bound')
    return config


def load_judge_config(path):
    if path is None:return {'type':'disabled'}
    path=Path(path).resolve();config=strict_json(path.read_text())
    if config.get('type')=='generation-provider':
        config['provider']=resolve_provider(config['provider'],path.parent)
    return validate_config(config)


def _append(stream,row):
    stream.write(canonical(row)+'\n');stream.flush();os.fsync(stream.fileno())


def run_business(cases_path,rubrics_path,solver_path,output,*,judge_path=None,
                 allow_remote_solver=False,allow_remote_judge=False,save_raw_judge=False,
                 reuse_from=None):
    """Run into a NEW directory; optional validated reuse never mutates its source."""
    from .evaluation import evaluate
    cases,rubrics,_=import_inputs(cases_path,rubrics_path)
    solver_config=load_solver_config(solver_path)
    judge_config=load_judge_config(judge_path)
    count=len(cases)*len(solver_config['configurations'])*solver_config.get('repetitions',1)
    reusable={'solver':{},'judge':{},'notes':[]}
    if reuse_from is not None:
        from ..resume import load_reusable_run, require_api_configs
        require_api_configs(solver_config, judge_config)
        reusable=load_reusable_run(reuse_from,cases_path,rubrics_path,solver_config,judge_config)
    new_solver_count=count-len(reusable['solver'])
    if new_solver_count>solver_config.get('max_calls',100):raise ValidationError('Solver call budget would be exceeded')
    output=Path(output)
    if reuse_from is not None and output.resolve().is_relative_to(Path(reuse_from).resolve()):
        raise ValidationError('Resume output must not be inside the source run')
    output.mkdir(parents=True,exist_ok=False)
    import fcntl
    run_lock=(output/'.run.lock').open('w')
    fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest={'schema_version':'business-pipeline-v1','state':'running','phase':'solver','started_at_utc':now(),
              'target_quality_method':'neural_judge_only','cases_sha256':sha(cases_path),'rubrics_sha256':sha(rubrics_path),
              'case_ids':[c['id'] for c in cases],'planned_attempts':count,
              'configuration_names':[c['name'] for c in solver_config['configurations']],
              'repetitions':solver_config.get('repetitions',1),
              'judge_latency_scope':'Whole JudgeProvider.judge call plus strict output validation; excludes judge construction. With raw logging includes local raw log write.',
              'total_latency_scope':'solver_plus_judge_call_seconds',
              'solver_model':solver_config['model'],'solver_config_sha256':digest(solver_config),
              'judge_config_sha256':digest(judge_config),'judge_type':judge_config['type'],
              'save_raw_judge':save_raw_judge,'cost_usd':None,
              'code_sha256':source_hashes(),
              'execution_backend': {'solver': 'promptfoo' if solver_config['provider'].get('type') == 'openai-compatible' else 'legacy-disabled',
                                    'judge': 'promptfoo' if judge_config.get('type') in {'generation-provider','openai-compatible'} else 'disabled',
                                    'version': '0.125.1', 'cache': False, 'transport_retries': 0},
              'execution':{'new_solver_calls':0,'new_judge_calls':0,'reused_solver_answers':0,
                           'reused_judgements':0,'new_call_cost_usd':None,
                           'cost_scope':'new attempted requests, excluding known local cooldown/budget refusals; effective per-task costs retain original calls'},
              'notes':['Only valid neural-judge outputs determine quality, including numeric answers.',
                       'Unavailable/error judgements remain unknown, never numerical quality zero.',
                       'Resume creates a new run and retains old latency/cost for reused results.',
                       'Configured call limits apply to NEW calls, including structured repairs.',
                       'Cost remains unknown if any newly attempted call has no provider-reported cost.']}
    if reuse_from is not None:
        import shutil
        manifest['resumed_from']={'manifest_sha256':sha(Path(reuse_from)/'manifest.json'),
                                  'source_name':Path(reuse_from).name,'notes':reusable['notes']}
        shutil.copytree(reuse_from,output/'history'/'source')
    mp=output/'manifest.json'
    def save_manifest():
        temp=mp.with_suffix('.tmp');temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');temp.replace(mp)
    save_manifest();responses=[];rows=[];solver_costs={};new_costs=[]
    def cost(value):
        return value if type(value) in (int,float) and math.isfinite(value) and value>=0 else None
    def key(record):return (record['id'],record['model'],record['configuration'],record.get('repetition',0))
    try:
        solver=make_provider(solver_config['provider'],allow_remote_solver) if new_solver_count else None
        with (output/'solver_records.jsonl').open('x',encoding='utf-8') as raw_stream,(output/'responses.jsonl').open('x',encoding='utf-8') as response_stream:
            for cfg in solver_config['configurations']:
                for repetition in range(solver_config.get('repetitions',1)):
                    for case in cases:
                        messages=build_solver_messages(case)
                        if cfg.get('instruction'):messages[0]['content']+='\n'+cfg['instruction']
                        base={'id':case['id'],'model':solver_config['model'],'configuration':cfg['name'],
                              'repetition':repetition,'provenance':'model-inference'}
                        k=key(base)
                        if k in reusable['solver']:
                            record,response=reusable['solver'][k]
                            record=dict(record);response=dict(response)
                            record['reused_from_manifest_sha256']=manifest['resumed_from']['manifest_sha256']
                            manifest['execution']['reused_solver_answers']+=1
                        else:
                            response={**base,'status':'ok','response':''}
                            record={**base,'messages':messages,'prompt_sha256':digest(messages),
                                    'parameters':cfg.get('parameters',{}),'started_at_utc':now(),'error':None,
                                    'token_usage':{},'raw_output':'','latency_scope':'Whole Provider.generate call; excludes outer provider construction and judge. For command providers includes child model loading.'}
                            start=time.monotonic()
                            manifest['execution']['new_solver_calls']+=1
                            try:
                                generated=solver.generate(messages,solver_config['model'],cfg.get('parameters',{}),
                                                {'id':case['id'],'config':cfg['name'],'repetition':repetition})
                                if not isinstance(generated.text,str):raise ProviderError('nontext_response')
                                response['response']=generated.text
                                record.update(raw_output=generated.text,token_usage=generated.usage,provider_metadata=generated.metadata)
                                new_costs.append(cost(generated.metadata.get('cost_usd')))
                            except Exception as exc:
                                response['status']='error';record['error']='solver_provider_error'
                                safe=str(exc)
                                if safe in {'api_cooldown_active','opencode_cooldown_active'}:
                                    manifest['execution']['new_solver_calls']-=1
                                else:new_costs.append(None)
                                if safe.startswith(('opencode_','api_')) and safe.replace('_','').isalnum():record['error']=safe
                                delay=getattr(exc,'retry_after_seconds',None)
                                if type(delay) in (int,float) and math.isfinite(delay) and delay>=0:
                                    from datetime import timedelta
                                    record['retry_after_seconds']=delay
                                    record['retry_not_before_utc']=(datetime.now(timezone.utc)+timedelta(seconds=delay)).isoformat()
                            record['latency_seconds']=round(time.monotonic()-start,6)
                        solver_costs[k]=cost(record.get('provider_metadata',{}).get('cost_usd'))
                        _append(raw_stream,record);_append(response_stream,response);responses.append(response)
                        print(f'solver {len(responses)}/{count} status={response["status"]}'+(' reused' if k in reusable['solver'] else ''),flush=True)
        del solver;gc.collect()
        manifest.update(phase='judge',solver_completed_attempts=len(responses),responses_sha256=sha(output/'responses.jsonl'),
                        solver_records_sha256=sha(output/'solver_records.jsonl'))
        save_manifest()
        pending=any(key(r) not in reusable['judge'] for r in responses)
        actual_judge=make_judge(judge_config,allow_remote_judge) if pending else None
        raw_judge_stream=(output/'judge_raw.jsonl').open('x',encoding='utf-8') if save_raw_judge else None
        if actual_judge is not None:
            class RecordingJudge:
                kind=actual_judge.kind;model=actual_judge.model
                structured_retries=getattr(actual_judge,'structured_retries',0)
                def judge(self,messages,case_id):
                    manifest['execution']['new_judge_calls']+=1
                    try:
                        reply=actual_judge.judge(messages,case_id)
                    except Exception as exc:
                        if str(exc) in {'judge_api_cooldown_active','judge_opencode_cooldown_active','judge_call_budget_exhausted'}:
                            manifest['execution']['new_judge_calls']-=1
                        else:new_costs.append(None)
                        raise
                    new_costs.append(cost(reply.metadata.get('cost_usd')))
                    if raw_judge_stream is not None:
                        _append(raw_judge_stream,{'id':case_id,'prompt_sha256':digest(messages),'raw_output':reply.text,
                                                 'usage':reply.usage,'metadata':reply.metadata})
                    return reply
            judge=RecordingJudge()
        else:judge=None
        case_index={c['id']:c for c in cases}
        try:
            with (output/'records.jsonl').open('x',encoding='utf-8') as stream:
                for response in sorted(responses,key=lambda r:(r['model'],r['configuration'],r['repetition'],[c['id'] for c in cases].index(r['id']))):
                    k=key(response);case=case_index[response['id']]
                    if k in reusable['judge']:
                        row=dict(reusable['judge'][k])
                        row['reused_from_manifest_sha256']=manifest['resumed_from']['manifest_sha256']
                        manifest['execution']['reused_judgements']+=1
                    else:
                        row=evaluate(case,rubrics[case['rubric_id']],response['response'],judge,response_status=response['status'])
                        delay=row['judge'].get('retry_after_seconds')
                        if type(delay) in (int,float) and math.isfinite(delay) and delay>=0:
                            from datetime import timedelta
                            row['judge']['retry_not_before_utc']=(datetime.now(timezone.utc)+timedelta(seconds=delay)).isoformat()
                        row.update(model=response['model'],configuration=response['configuration'],repetition=response['repetition'],
                                   response_provenance=response['provenance'])
                    solver_cost=solver_costs.get(k);judge_cost=cost(row['judge'].get('cost_usd'))
                    row['total_cost_usd']=solver_cost+judge_cost if solver_cost is not None and judge_cost is not None else None
                    _append(stream,row);rows.append(row)
                    print(f'judge {len(rows)}/{count} status={row["judge"]["status"]}'+(' reused' if k in reusable['judge'] else ''),flush=True)
        finally:
            if raw_judge_stream is not None:raw_judge_stream.close()
        if all(r.get('total_cost_usd') is not None for r in rows):
            manifest['cost_measurement']={'currency':'USD','scope':'solver_plus_judge_per_attempt','basis':'provider_reported'}
        manifest['execution']['new_call_cost_usd']=sum(new_costs) if all(c is not None for c in new_costs) else None
        result=write_report(rows,output)
        manifest.update(state='completed',phase='complete',completed_at_utc=now(),evaluated_attempts=len(rows),
                        judge_evaluated=result['summary']['judge_evaluated'],records_sha256=sha(output/'records.jsonl'))
        save_manifest();return result
    except BaseException as exc:
        manifest.update(state='interrupted',updated_at_utc=now(),solver_completed_attempts=len(responses),
                        evaluated_attempts=len(rows),failure_type=type(exc).__name__)
        save_manifest()
        raise
    finally:
        run_lock.close()
