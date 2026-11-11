"""User case → real solver → neural judge → report. No numeric answer scorer."""
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from ..providers import Generation, ProviderError, ProviderUnavailable
from ..provenance import source_hashes
from .schema import ValidationError,canonical,digest,strict_json,import_inputs,fields,identifier,string
from .evaluation import build_solver_messages, evaluate
from .providers import validate_config, JudgeReply, JudgeUnavailable, JudgeProviderError
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


def evaluate_jobs(*args,**kwargs):
    from ..promptfoo_backend import evaluate_jobs as execute
    return execute(*args,**kwargs)


def _cost(value):
    return value if type(value) in (int,float) and math.isfinite(value) and value>=0 else None


def _key(record):
    return (record['id'],record['model'],record['configuration'],record.get('repetition',0))


def _deadline(row, delay):
    if type(delay) in (int,float) and math.isfinite(delay) and delay>=0:
        from datetime import timedelta
        row['retry_after_seconds']=delay
        row['retry_not_before_utc']=(datetime.now(timezone.utc)+timedelta(seconds=delay)).isoformat()


def _judge_error(error):
    if isinstance(error,ProviderUnavailable):
        import re
        code=str(error)
        mapped=JudgeUnavailable('judge_'+code if re.fullmatch(r'api_[a-z0-9_]+',code) else 'judge_connection_unavailable')
        if hasattr(error,'retry_after_seconds'):mapped.retry_after_seconds=error.retry_after_seconds
        return mapped
    return JudgeProviderError('judge_provider_error')


class _NeedsJudge(BaseException):
    """Suspend existing strict evaluation before its next scheduled model call."""
    def __init__(self,messages):self.messages=messages


class _ReplayJudge:
    """Replay durable replies through the original validator and repair policy."""
    def __init__(self,config,replies):
        self.kind='disabled' if config['type']=='disabled' else 'model-inference'
        self.model=config.get('model','not-connected')
        self.structured_retries=config.get('structured_retries',0)
        self.replies=replies;self.position=0
    def judge(self,messages,case_id):
        if self.position>=len(self.replies):raise _NeedsJudge(messages)
        reply=self.replies[self.position];self.position+=1
        if isinstance(reply,Exception):raise reply
        return reply


class _Run:
    """Per-model evidence writer. Scheduling belongs entirely to Promptfoo."""
    def __init__(self,spec,index,max_concurrency):
        self.spec=spec;self.index=index;self.max_concurrency=max_concurrency
        self.cases,self.rubrics,_=import_inputs(spec['cases_path'],spec['rubrics_path'])
        self.solver=load_solver_config(spec['solver_path'])
        self.judge=load_judge_config(spec.get('judge_path'))
        self.count=len(self.cases)*len(self.solver['configurations'])*self.solver.get('repetitions',1)
        self.reusable={'solver':{},'judge':{},'notes':[]}
        self.output=Path(spec['output'])
        self.reuse_from=spec.get('reuse_from')
        if self.reuse_from is not None:
            from ..resume import load_reusable_run,require_api_configs
            require_api_configs(self.solver,self.judge)
            self.reusable=load_reusable_run(self.reuse_from,spec['cases_path'],spec['rubrics_path'],self.solver,self.judge)
            if self.output.resolve().is_relative_to(Path(self.reuse_from).resolve()):
                raise ValidationError('Resume output must not be inside the source run')
        if self.count-len(self.reusable['solver'])>self.solver.get('max_calls',100):
            raise ValidationError('Solver call budget would be exceeded')
        if self.output.exists():raise FileExistsError(str(self.output))
        self.responses={};self.rows={};self.solver_costs={};self.new_costs=[];self.pending={};self.started=set()
        self.judge_states={};self.judge_remaining=self.judge.get('max_calls',100)
        self.manifest=None

    def open(self,stack):
        import fcntl
        self.output.mkdir(parents=True,exist_ok=False)
        lock=stack.enter_context((self.output/'.run.lock').open('w'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.manifest={'schema_version':'business-pipeline-v1','state':'running','phase':'solver','started_at_utc':now(),
            'target_quality_method':'neural_judge_only','cases_sha256':sha(self.spec['cases_path']),'rubrics_sha256':sha(self.spec['rubrics_path']),
            'case_ids':[c['id'] for c in self.cases],'planned_attempts':self.count,
            'configuration_names':[c['name'] for c in self.solver['configurations']],
            'repetitions':self.solver.get('repetitions',1),
            'judge_latency_scope':'Sum of Promptfoo scheduled judge API call durations and strict validation; excludes queue wait and worker startup; includes explicit repair attempts.',
            'total_latency_scope':'solver_plus_judge_call_seconds',
            'solver_model':self.solver['model'],'solver_config_sha256':digest(self.solver),
            'judge_config_sha256':digest(self.judge),'judge_type':self.judge['type'],
            'save_raw_judge':self.spec.get('save_raw_judge',False),'cost_usd':None,
            'code_sha256':source_hashes(),
            'execution_backend':{'solver':'promptfoo','judge':'disabled' if self.judge['type']=='disabled' else 'promptfoo',
                                 'version':'0.125.1','cache':False,'transport_retries':0},
            'scheduler':{'owner':'promptfoo.evaluate','max_concurrency':self.max_concurrency,'cache':False,'automatic_retries':0},
            'execution':{'new_solver_calls':0,'new_judge_calls':0,'reused_solver_answers':0,'reused_judgements':0,
                         'new_call_cost_usd':None,'unresolved_scheduled_jobs':0,'inflight_jobs':{},
                         'cost_scope':'new completed attempted requests, excluding known local cooldown/budget refusals; interrupted outstanding requests have unknown cost'},
            'notes':['Only valid neural-judge outputs determine quality, including numeric answers.',
                     'Unavailable/error judgements remain unknown, never numerical quality zero.',
                     'Promptfoo owns matrix scheduling and bounded concurrency; queue wait is not model latency.',
                     'Resume creates a new run and retains old latency/cost for reused results.',
                     'Configured call limits include explicit structured repairs.',
                     'Interrupted in-flight requests may have reached the provider; their costs remain unknown.']}
        if self.reuse_from is not None:
            import shutil
            self.manifest['resumed_from']={'manifest_sha256':sha(Path(self.reuse_from)/'manifest.json'),
                'source_name':Path(self.reuse_from).name,'notes':self.reusable['notes']}
            source_manifest=strict_json((Path(self.reuse_from)/'manifest.json').read_text())
            if self.reusable['judge']:
                old_scope=source_manifest.get('judge_latency_scope')
                if old_scope!=self.manifest['judge_latency_scope']:
                    self.manifest['judge_latency_scope']=old_scope if len(self.reusable['judge'])==self.count else None
                    self.manifest['notes'].append('Reused judge timing retains its original scope; mixed old/new scopes are not used for total-latency selection.')
            shutil.copytree(self.reuse_from,self.output/'history'/'source')
        self.raw=stack.enter_context((self.output/'solver_records.jsonl').open('x',encoding='utf-8'))
        self.response_stream=stack.enter_context((self.output/'responses.jsonl').open('x',encoding='utf-8'))
        self.record_stream=stack.enter_context((self.output/'records.jsonl').open('x',encoding='utf-8'))
        self.raw_judge=stack.enter_context((self.output/'judge_raw.jsonl').open('x',encoding='utf-8')) if self.spec.get('save_raw_judge') else None
        self.save()

    def save(self):
        path=self.output/'manifest.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)

    def started_job(self,job_id,started_at):
        if job_id not in self.pending or job_id in self.started:
            raise ValidationError('Unexpected scheduled job start')
        self.started.add(job_id)
        phase='new_solver_calls' if job_id.startswith('solver-') else 'new_judge_calls'
        self.manifest['execution'][phase]+=1
        self.manifest['execution']['inflight_jobs'][job_id]=started_at
        self.save()

    def account_call(self,job_id,result,phase):
        attempted=getattr(result,'request_attempted',str(result)!='api_cooldown_active') if isinstance(result,Exception) else result.metadata.get('request_attempted',True)
        if job_id in self.started:
            self.started.remove(job_id)
            if attempted is False:self.manifest['execution'][phase]-=1
        elif attempted is not False:self.manifest['execution'][phase]+=1
        self.manifest['execution']['inflight_jobs'].pop(job_id,None)
        return attempted is not False

    def solver_jobs(self):
        jobs=[]
        for cfg in self.solver['configurations']:
            for repetition in range(self.solver.get('repetitions',1)):
                for case in self.cases:
                    base={'id':case['id'],'model':self.solver['model'],'configuration':cfg['name'],
                          'repetition':repetition,'provenance':'model-inference'}
                    key=_key(base)
                    if key in self.reusable['solver']:
                        record,response=self.reusable['solver'][key]
                        record=dict(record);response=dict(response)
                        record['reused_from_manifest_sha256']=self.manifest['resumed_from']['manifest_sha256']
                        self.manifest['execution']['reused_solver_answers']+=1
                        self.write_solver(record,response)
                        continue
                    messages=build_solver_messages(case)
                    if cfg.get('instruction'):messages[0]['content']+='\n'+cfg['instruction']
                    job_id=f'solver-{self.index}-{len(jobs)}'
                    record={**base,'messages':messages,'prompt_sha256':digest(messages),'parameters':cfg.get('parameters',{}),
                            'queued_at_utc':now(),'error':None,'token_usage':{},'raw_output':'',
                            'latency_scope':'Promptfoo scheduled API call; excludes queue wait, worker startup and judge.'}
                    self.pending[job_id]=(record,{**base,'status':'ok','response':''})
                    jobs.append({'job_id':job_id,'provider':self.solver['provider'],'messages':messages,
                                 'model':self.solver['model'],'parameters':cfg.get('parameters',{}),
                                 'context':{'id':case['id'],'config':cfg['name'],'repetition':repetition},
                                 'allow_remote':self.spec.get('allow_remote_solver',False)})
        return jobs

    def write_solver(self,record,response):
        key=_key(record)
        if key in self.responses:raise ValidationError('Duplicate scheduled solver result')
        self.solver_costs[key]=_cost(record.get('provider_metadata',{}).get('cost_usd'))
        _append(self.raw,record);_append(self.response_stream,response)
        self.responses[key]=response
        self.manifest['solver_completed_attempts']=len(self.responses)
        self.save()
        print(f'solver {len(self.responses)}/{self.count} model={self.solver["model"]} status={response["status"]}',flush=True)

    def solver_result(self,job_id,result,latency):
        record,response=self.pending.pop(job_id)
        started_at=self.manifest['execution']['inflight_jobs'].get(job_id)
        attempted=self.account_call(job_id,result,'new_solver_calls')
        record['latency_seconds']=round(latency,6) if attempted else None
        from datetime import timedelta
        record['started_at_utc']=(started_at if isinstance(started_at,str) else (datetime.now(timezone.utc)-timedelta(seconds=latency)).isoformat()) if attempted else None
        if isinstance(result,Generation):
            if not isinstance(result.text,str):result=ProviderError('nontext_response')
        if isinstance(result,Exception):
            response['status']='error';record['error']='solver_provider_error'
            safe=str(result)
            if safe.startswith(('api_','promptfoo_')) and safe.replace('_','').isalnum():record['error']=safe
            if attempted:self.new_costs.append(None)
            _deadline(record,getattr(result,'retry_after_seconds',None))
        else:
            response['response']=result.text
            record.update(raw_output=result.text,token_usage=result.usage,provider_metadata=result.metadata)
            self.new_costs.append(_cost(result.metadata.get('cost_usd')))
        self.write_solver(record,response)

    def judge_provider(self):
        if self.judge['type']=='generation-provider':
            return self.judge['provider'],self.judge.get('parameters',{'temperature':0,'max_tokens':self.judge.get('max_tokens',2048)})
        provider={key:self.judge[key] for key in ('base_url','api_key_env','timeout_seconds') if key in self.judge}
        provider.update(type='openai-compatible',engine='promptfoo')
        provider.setdefault('api_key_env','NIR_JUDGE_API_KEY');provider.setdefault('timeout_seconds',90)
        return provider,{'temperature':0,'max_tokens':self.judge.get('max_tokens',2048)}

    def judge_jobs(self):
        self.manifest.update(phase='judge',responses_sha256=sha(self.output/'responses.jsonl'),
                             solver_records_sha256=sha(self.output/'solver_records.jsonl'))
        self.save();jobs=[]
        for key,response in sorted(self.responses.items()):
            if key in self.reusable['judge']:
                row=dict(self.reusable['judge'][key]);row['reused_from_manifest_sha256']=self.manifest['resumed_from']['manifest_sha256']
                self.manifest['execution']['reused_judgements']+=1;self.write_judge(key,row)
            else:
                self.judge_states[key]={'replies':[],'latencies':[],'validation_seconds':0}
                job=self.advance_judge(key)
                if job:jobs.append(job)
        return jobs

    def advance_judge(self,key,*,reserve=True):
        state=self.judge_states[key];response=self.responses[key]
        case=next(c for c in self.cases if c['id']==response['id'])
        start=time.monotonic()
        try:
            row=evaluate(case,self.rubrics[case['rubric_id']],response['response'],
                         _ReplayJudge(self.judge,state['replies']),response_status=response['status'])
        except _NeedsJudge as suspended:
            state['validation_seconds']+=time.monotonic()-start
            if not reserve:return key
            if self.judge_remaining<=0:
                state['replies'].append(JudgeUnavailable('judge_call_budget_exhausted'))
                return self.advance_judge(key)
            self.judge_remaining-=1
            job_id=f'judge-{self.index}-{len(self.judge_states)}-{digest(key)[:24]}-{len(state["replies"])}'
            provider,parameters=self.judge_provider()
            self.pending[job_id]=(key,suspended.messages)
            return {'job_id':job_id,'provider':provider,'messages':suspended.messages,'model':self.judge['model'],
                    'parameters':parameters,'context':{'id':case['id'],'config':'business-judge','repetition':key[3]},
                    'allow_remote':self.spec.get('allow_remote_judge',False)}
        state['validation_seconds']+=time.monotonic()-start
        row['judge']['latency_seconds']=(round(sum(state['latencies'])+state['validation_seconds'],6)
                                         if state['latencies'] else None)
        _deadline(row['judge'],row['judge'].get('retry_after_seconds'))
        row.update(model=response['model'],configuration=response['configuration'],repetition=response['repetition'],
                   response_provenance=response['provenance'])
        self.write_judge(key,row)
        return None

    def judge_result(self,job_id,result,latency):
        key,messages=self.pending.pop(job_id);state=self.judge_states[key]
        attempted=self.account_call(job_id,result,'new_judge_calls')
        if attempted:state['latencies'].append(latency)
        if isinstance(result,Exception):
            if not attempted:self.judge_remaining+=1
            else:self.new_costs.append(None)
            state['replies'].append(_judge_error(result))
        else:
            self.new_costs.append(_cost(result.metadata.get('cost_usd')))
            if self.raw_judge is not None:
                _append(self.raw_judge,{'id':key[0],'model':key[1],'configuration':key[2],'repetition':key[3],
                    'attempt':len(state['replies'])+1,'prompt_sha256':digest(messages),'raw_output':result.text,
                    'usage':result.usage,'metadata':result.metadata})
            state['replies'].append(JudgeReply(result.text,result.usage,result.metadata))
        pending_key=self.advance_judge(key,reserve=False);self.save();return pending_key

    def write_judge(self,key,row):
        if key in self.rows:raise ValidationError('Duplicate scheduled judge result')
        solver_cost=self.solver_costs.get(key);judge_cost=_cost(row['judge'].get('cost_usd'))
        row['total_cost_usd']=solver_cost+judge_cost if solver_cost is not None and judge_cost is not None else None
        _append(self.record_stream,row);self.rows[key]=row
        self.manifest['evaluated_attempts']=len(self.rows);self.save()
        print(f'judge {len(self.rows)}/{self.count} model={self.solver["model"]} status={row["judge"]["status"]}',flush=True)

    def finish(self):
        rows=[self.rows[key] for key in sorted(self.rows)]
        if len(rows)!=self.count or self.pending:raise ValidationError('Scheduled matrix is incomplete')
        if all(r.get('total_cost_usd') is not None for r in rows):
            self.manifest['cost_measurement']={'currency':'USD','scope':'solver_plus_judge_per_attempt','basis':'provider_reported'}
        self.manifest['execution']['new_call_cost_usd']=sum(self.new_costs) if all(c is not None for c in self.new_costs) else None
        result=write_report(rows,self.output)
        self.manifest.update(state='completed',phase='complete',completed_at_utc=now(),evaluated_attempts=len(rows),
                             judge_evaluated=result['summary']['judge_evaluated'],records_sha256=sha(self.output/'records.jsonl'))
        self.save();return result


def run_business_matrix(specs,*,max_concurrency=4):
    """Execute one matrix through Promptfoo, streaming durable per-model evidence.

    A spec contains cases_path, rubrics_path, solver_path, output and optional
    judge_path, allow_remote_solver/judge, save_raw_judge, reuse_from. Matrix
    orchestration remains shared even when multiple solver configurations exist.
    """
    from contextlib import ExitStack
    if type(max_concurrency) is not int or not 1<=max_concurrency<=32:
        raise ValidationError('max_concurrency must be an integer between 1 and 32')
    if not isinstance(specs,list) or not specs:raise ValidationError('No matrix specifications')
    runs=[_Run(spec,index,max_concurrency) for index,spec in enumerate(specs)]
    outputs=[run.output.resolve() for run in runs]
    sources=[Path(run.reuse_from).resolve() for run in runs if run.reuse_from is not None]
    if any(output.is_relative_to(source) for output in outputs for source in sources):
        raise ValidationError('Matrix outputs must not modify any resume source')
    if len(set(outputs))!=len(outputs) or any(a!=b and a.is_relative_to(b) for a in outputs for b in outputs):
        raise ValidationError('Matrix output directories must be distinct and disjoint')
    with ExitStack() as stack:
        try:
            for run in runs:run.open(stack)
            jobs=[job for run in runs for job in run.solver_jobs()]
            owners={job['job_id']:run for run in runs for job in jobs if job['job_id'] in run.pending}
            if jobs:
                summary=evaluate_jobs(jobs,max_concurrency,lambda jid,result,seconds:owners[jid].solver_result(jid,result,seconds),
                                      on_started=lambda jid,started:owners[jid].started_job(jid,started))
                for run in runs:run.manifest['scheduler']['solver_matrix']=summary;run.save()
            jobs=[job for run in runs for job in run.judge_jobs()]
            while jobs:
                owners={job['job_id']:run for run in runs for job in jobs if job['job_id'] in run.pending}
                next_round=[]
                def completed(job_id,result,seconds):
                    run=owners[job_id]
                    key=run.judge_result(job_id,result,seconds)
                    if key:next_round.append((run.index,key))
                summary=evaluate_jobs(jobs,max_concurrency,completed,
                                      on_started=lambda jid,started:owners[jid].started_job(jid,started))
                for run in runs:run.manifest['scheduler'].setdefault('judge_rounds',[]).append(summary);run.save()
                jobs=[]
                for index,key in sorted(next_round):
                    job=runs[index].advance_judge(key)
                    if job:jobs.append(job)
            return [run.finish() for run in runs]
        except BaseException as exc:
            for run in runs:
                if run.manifest is None:continue
                run.manifest.update(state='interrupted',updated_at_utc=now(),solver_completed_attempts=len(run.responses),
                                    evaluated_attempts=len(run.rows),failure_type=type(exc).__name__)
                run.manifest['execution']['unresolved_scheduled_jobs']=len(run.pending)
                run.manifest['execution']['possibly_attempted_job_ids']=sorted(run.pending)
                run.manifest['notes'].append('Started events can be in transit at interruption: every submitted unfinished job may have reached the API, including jobs absent from the inflight journal.')
                run.manifest['execution']['new_call_cost_usd']=None
                run.save()
            raise


def run_business(cases_path,rubrics_path,solver_path,output,*,judge_path=None,
                 allow_remote_solver=False,allow_remote_judge=False,save_raw_judge=False,
                 reuse_from=None,max_concurrency=4):
    """Compatibility entry point; a single model uses the same matrix scheduler."""
    return run_business_matrix([dict(cases_path=cases_path,rubrics_path=rubrics_path,solver_path=solver_path,
        output=output,judge_path=judge_path,allow_remote_solver=allow_remote_solver,
        allow_remote_judge=allow_remote_judge,save_raw_judge=save_raw_judge,reuse_from=reuse_from)],
        max_concurrency=max_concurrency)[0]
