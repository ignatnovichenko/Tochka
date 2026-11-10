"""User case → real solver → neural judge → report. No numeric answer scorer."""
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from ..providers import make_provider,ProviderError
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
    if p.get('type') not in {'transformers-local','openai-compatible','command'}:
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
                 allow_remote_solver=False,allow_remote_judge=False,save_raw_judge=False):
    cases,rubrics,_=import_inputs(cases_path,rubrics_path)
    solver_config=load_solver_config(solver_path)
    judge_config=load_judge_config(judge_path)
    count=len(cases)*len(solver_config['configurations'])*solver_config.get('repetitions',1)
    if count>solver_config.get('max_calls',100):raise ValidationError('Solver call budget would be exceeded')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest={'schema_version':'business-pipeline-v1','state':'running','phase':'solver','started_at_utc':now(),
              'target_quality_method':'neural_judge_only','cases_sha256':sha(cases_path),'rubrics_sha256':sha(rubrics_path),
              'case_ids':[c['id'] for c in cases],'planned_attempts':count,
              'solver_model':solver_config['model'],'solver_config_sha256':digest(solver_config),
              'judge_config_sha256':digest(judge_config),'judge_type':judge_config['type'],
              'save_raw_judge':save_raw_judge,'cost_usd':None,
              'code_sha256':{str(p.relative_to(Path(__file__).parents[1])):sha(p)
                             for p in sorted(Path(__file__).parents[1].rglob('*.py'))},
              'notes':['The solver sees only task/context plus explicit configuration instructions.',
                       'Only a valid neural-judge output determines quality, including numbers.',
                       'Unavailable/error judgements remain unknown, never numerical quality zero.',
                       'All model calls are explicit. No LLM judge has been certified as reliable.',
                       'No automatic resume or retries: interrupted directories are preserved.',
                       'Cost is unknown unless independently measured; max_calls is not a currency budget.']}
    mp=output/'manifest.json'
    def save_manifest():mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    save_manifest();responses=[];rows=[]
    try:
        solver=make_provider(solver_config['provider'],allow_remote_solver)
        with (output/'solver_records.jsonl').open('x',encoding='utf-8') as raw_stream,(output/'responses.jsonl').open('x',encoding='utf-8') as response_stream:
            for cfg in solver_config['configurations']:
                for repetition in range(solver_config.get('repetitions',1)):
                    for case in cases:
                        messages=build_solver_messages(case)
                        if cfg.get('instruction'):messages[0]['content']+='\n'+cfg['instruction']
                        base={'id':case['id'],'model':solver_config['model'],'configuration':cfg['name'],
                              'repetition':repetition,'provenance':'model-inference'}
                        response={**base,'status':'ok','response':''}
                        record={**base,'messages':messages,'prompt_sha256':digest(messages),
                                'parameters':cfg.get('parameters',{}),'started_at_utc':now(),'error':None,
                                'token_usage':{},'raw_output':'','latency_scope':'Whole Provider.generate call; excludes outer provider construction and judge. For command providers includes child model loading.'}
                        start=time.monotonic()
                        try:
                            generated=solver.generate(messages,solver_config['model'],cfg.get('parameters',{}),
                                            {'id':case['id'],'config':cfg['name'],'repetition':repetition})
                            if not isinstance(generated.text,str):raise ProviderError('nontext_response')
                            response['response']=generated.text
                            record.update(raw_output=generated.text,token_usage=generated.usage,provider_metadata=generated.metadata)
                        except Exception:
                            response['status']='error';record['error']='solver_provider_error'
                        record['latency_seconds']=round(time.monotonic()-start,6)
                        _append(raw_stream,record);_append(response_stream,response);responses.append(response)
                        print(f'solver {len(responses)}/{count} status={response["status"]}',flush=True)
        # Phases are separated to avoid concurrently retaining two large local models.
        del solver;gc.collect()
        manifest.update(phase='judge',solver_completed_attempts=len(responses),responses_sha256=sha(output/'responses.jsonl'),
                        solver_records_sha256=sha(output/'solver_records.jsonl'))
        save_manifest()
        actual_judge=make_judge(judge_config,allow_remote_judge)
        if save_raw_judge:
            raw_judge_stream=(output/'judge_raw.jsonl').open('x',encoding='utf-8')
            class RecordingJudge:
                kind=actual_judge.kind;model=actual_judge.model
                def judge(self,messages,case_id):
                    reply=actual_judge.judge(messages,case_id)
                    _append(raw_judge_stream,{'id':case_id,'prompt_sha256':digest(messages),'raw_output':reply.text,
                                             'usage':reply.usage,'metadata':reply.metadata})
                    return reply
            judge=RecordingJudge()
        else:judge=actual_judge;raw_judge_stream=None
        try:
            with (output/'records.jsonl').open('x',encoding='utf-8') as stream:
                for row in evaluate_matrix(cases,rubrics,responses,judge,judge_config.get('max_calls',100)):
                    _append(stream,row);rows.append(row)
                    print(f'judge {len(rows)}/{count} status={row["judge"]["status"]}',flush=True)
        finally:
            if raw_judge_stream is not None:raw_judge_stream.close()
        result=write_report(rows,output)
        manifest.update(state='completed',phase='complete',completed_at_utc=now(),evaluated_attempts=len(rows),
                        judge_evaluated=result['summary']['judge_evaluated'],records_sha256=sha(output/'records.jsonl'))
        save_manifest();return result
    except BaseException as exc:
        manifest.update(state='interrupted',updated_at_utc=now(),solver_completed_attempts=len(responses),
                        evaluated_attempts=len(rows),failure_type=type(exc).__name__)
        save_manifest()
        raise
