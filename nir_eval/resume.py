"""Immutable resume: validate saved evidence, reuse successes, save a new execution."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from .business.schema import ValidationError, strict_json, digest, canonical, import_inputs
from .business.running import sha, load_solver_config, load_judge_config, run_business
from .business.evaluation import build_solver_messages, build_judge_messages, parse_judgement


def require_api_configs(solver, judge):
    if solver['provider'].get('type') != 'openai-compatible':
        raise ValidationError('Resume supports API configurations only; legacy CLI results remain readable')
    kind=judge.get('type')
    if kind not in {'disabled','openai-compatible'} and not (kind=='generation-provider' and judge.get('provider',{}).get('type')=='openai-compatible'):
        raise ValidationError('Resume supports API judges only; legacy CLI results remain readable')


def _key(row):
    try:k=(row['id'],row['model'],row['configuration'],row.get('repetition',0))
    except (KeyError,TypeError):raise ValidationError('Saved attempt identity is missing') from None
    if any(not isinstance(v,str) or not v for v in k[:3]) or type(k[3]) is not int or k[3]<0:
        raise ValidationError('Saved attempt identity is invalid')
    return k


def _rows(path, interrupted, notes):
    """Only an unparseable, unterminated final JSONL row can be an interrupted write."""
    if not path.exists():return {}
    if path.is_symlink():raise ValidationError('Resume evidence must not be symlinked')
    raw=path.read_bytes()
    result={}
    lines=raw.splitlines(keepends=True)
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:row=strict_json(line)
        except ValueError:
            if interrupted and i==len(lines)-1 and not line.endswith(b'\n'):
                notes.append('Ignored incomplete final write in '+path.name+'; original bytes retained in history')
                break
            raise ValidationError('Saved JSONL contains a corrupted complete row') from None
        k=_key(row)
        if k in result:raise ValidationError('Duplicate saved attempt identity')
        result[k]=row
    return result


def load_reusable_run(source,cases_path,rubrics_path,solver,judge):
    source=Path(source).resolve()
    lock_path=source/'.run.lock'
    if lock_path.exists():
        import fcntl
        with lock_path.open('r') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError:raise ValidationError('Source run is still active; stop it before resuming') from None
    mp=source/'manifest.json'
    manifest=strict_json(mp.read_text())
    if manifest.get('schema_version')!='business-pipeline-v1' or manifest.get('state') not in {'running','interrupted','completed'}:
        raise ValidationError('Unsupported resume source manifest')
    if solver['provider'].get('type') == 'openai-compatible' and solver['provider'].get('engine') != 'promptfoo' and manifest.get('execution_backend',{}).get('solver') != 'promptfoo':
        raise ValidationError('Old API engine results remain readable; create a new Promptfoo experiment instead of mixing execution engines on resume')
    judge_provider = judge.get('provider',{}) if judge.get('type') == 'generation-provider' else judge
    if judge_provider.get('type') == 'openai-compatible' and judge_provider.get('engine') != 'promptfoo' and manifest.get('execution_backend',{}).get('judge') != 'promptfoo':
        raise ValidationError('Old judge engine results remain readable; start a new Promptfoo experiment instead of mixing judges on resume')
    expected_hashes={'cases_sha256':sha(cases_path),'rubrics_sha256':sha(rubrics_path),
                     'solver_config_sha256':digest(solver),'judge_config_sha256':digest(judge)}
    if any(manifest.get(k)!=v for k,v in expected_hashes.items()):
        raise ValidationError('Resume inputs or configurations differ from source')
    cases,rubrics,_=import_inputs(cases_path,rubrics_path)
    expected={(c['id'],solver['model'],cfg['name'],r) for c in cases for cfg in solver['configurations'] for r in range(solver.get('repetitions',1))}
    if manifest.get('planned_attempts')!=len(expected) or manifest.get('case_ids')!=[c['id'] for c in cases] or manifest.get('solver_model')!=solver['model']:
        raise ValidationError('Resume plan differs from source')
    notes=[];interrupted=manifest['state']!='completed'
    for name,field in [('solver_records.jsonl','solver_records_sha256'),('responses.jsonl','responses_sha256'),('records.jsonl','records_sha256')]:
        if field in manifest and (not (source/name).exists() or sha(source/name)!=manifest[field]):
            raise ValidationError('Saved evidence hash mismatch: '+name)
    solver_rows=_rows(source/'solver_records.jsonl',interrupted,notes)
    responses=_rows(source/'responses.jsonl',interrupted,notes)
    evaluations=_rows(source/'records.jsonl',interrupted,notes)
    if any(not set(rows)<=expected for rows in [solver_rows,responses,evaluations]):
        raise ValidationError('Unexpected attempt outside source plan')
    if not interrupted:
        from .business_selection import load_business_evidence
        load_business_evidence(source)
    case_index={c['id']:c for c in cases};cfg_index={c['name']:c for c in solver['configurations']}
    good_solver={};good_judge={};deadlines=[]
    for k,row in solver_rows.items():
        case=case_index[k[0]];cfg=cfg_index[k[2]]
        messages=build_solver_messages(case)
        if cfg.get('instruction'):messages[0]['content']+='\n'+cfg['instruction']
        if row.get('provenance')!='model-inference' or row.get('messages')!=messages or row.get('prompt_sha256')!=digest(messages) or row.get('parameters')!=cfg.get('parameters',{}):
            raise ValidationError('Saved solver prompt/parameters/provenance mismatch')
        if row.get('retry_not_before_utc'):deadlines.append(row['retry_not_before_utc'])
        if row.get('error') is not None:continue
        if not isinstance(row.get('raw_output'),str):raise ValidationError('Saved successful answer is invalid')
        if not isinstance(row.get('provider_metadata',{}),dict) or not isinstance(row.get('token_usage',{}),dict):
            raise ValidationError('Saved solver metadata is invalid')
        response=responses.get(k)
        if response is None:
            # Solver was fsynced before its companion response record; no need to call again.
            response={'id':k[0],'model':k[1],'configuration':k[2],'repetition':k[3],
                      'provenance':'model-inference','status':'ok','response':row['raw_output']}
            notes.append('Reconstructed companion response from durable solver record')
        if response.get('status')!='ok' or response.get('response')!=row['raw_output'] or response.get('provenance')!='model-inference':
            raise ValidationError('Saved successful solver response mismatch')
        good_solver[k]=(row,response)
    for k,row in responses.items():
        if row.get('status')=='ok' and k not in good_solver:
            raise ValidationError('Saved successful response has no verified solver record')
    for k,row in evaluations.items():
        j=row.get('judge',{})
        if j.get('retry_not_before_utc'):deadlines.append(j['retry_not_before_utc'])
        if j.get('status')!='evaluated':continue
        if k not in good_solver:raise ValidationError('Saved judgement has no successful solver answer')
        if row.get('schema_version')!='business-judge-v1' or row.get('checks_role')!='diagnostic_only':
            raise ValidationError('Saved judgement contract is invalid')
        case=case_index[k[0]];rubric=rubrics[case['rubric_id']];candidate=good_solver[k][1]['response']
        if row.get('case_sha256')!=digest(case) or row.get('rubric_sha256')!=digest(rubric) or row.get('candidate_sha256')!=digest(candidate):
            raise ValidationError('Saved judgement fingerprints differ')
        if row.get('rubric_id')!=rubric['id'] or row.get('status')!=('passed' if j.get('passed') is True else 'failed') or type(j.get('passed')) is not bool:
            raise ValidationError('Saved judgement rubric/status is invalid')
        if not isinstance(row.get('checks'),list) or row.get('deterministic_status') not in {'passed','failed','not_configured','not_evaluated'}:
            raise ValidationError('Saved diagnostic contract is invalid')
        if j.get('model')!=judge.get('model') or j.get('provenance')!='model-inference':
            raise ValidationError('Saved judge identity differs')
        if j.get('prompt_sha256')!=digest(build_judge_messages(case,rubric,candidate)):
            raise ValidationError('Saved judge prompt differs from the current evaluator; start a new evaluation')
        parsed=parse_judgement(canonical({'case_id':case['id'],'criteria':j.get('criteria'),'critical_errors':j.get('critical_errors')}),case,rubric,candidate)
        if any(j.get(f)!=parsed[f] or row.get(f)!=parsed[f] for f in ['score','passed']):
            raise ValidationError('Saved judgement score/pass differ from rubric')
        if row.get('response_provenance')!='model-inference' or row.get('response_status')!='ok':
            raise ValidationError('Saved judgement provenance mismatch')
        good_judge[k]=row
    return {'solver':good_solver,'judge':good_judge,'notes':notes,'deadlines':deadlines,'planned_attempts':len(expected)}


def _snapshot(source):
    source=Path(source).resolve()
    manifest=strict_json((source/'experiment-manifest.json').read_text())
    if manifest.get('state') not in {'running','interrupted','completed'}:raise ValidationError('Invalid experiment resume state')
    inputs=source/'inputs'
    hashes=manifest.get('input_hashes')
    if not isinstance(hashes,dict) or not hashes:raise ValidationError('Resume requires saved input fingerprints')
    for name,expected in hashes.items():
        if Path(name).name!=name or (inputs/name).is_symlink():raise ValidationError('Invalid saved input path')
        if not (inputs/name).is_file() or sha(inputs/name)!=expected:raise ValidationError('Experiment input fingerprint mismatch')
    cfg=strict_json((inputs/'experiment.json').read_text())
    names=['experiment.json','cases.jsonl','rubrics.jsonl',cfg.get('judge'),*cfg.get('solvers',[])]
    if any(not isinstance(n,str) or Path(n).name!=n or n not in hashes for n in names):
        raise ValidationError('Experiment input snapshot is incomplete or has unsafe paths')
    from .business_selection import validate_business_policy
    validate_business_policy(cfg.get('policy'))
    if not cfg['solvers']:raise ValidationError('Resume requires at least one solver')
    planned=[Path(p).stem for p in cfg['solvers']]
    if len(planned)!=len(set(planned)) or manifest.get('planned_runs')!=planned:raise ValidationError('Experiment run plan mismatch')
    return source,manifest,inputs,cfg


def resume_plan(source):
    source,manifest,inputs,cfg=_snapshot(source)
    judge=load_judge_config(inputs/cfg['judge'])
    cases,_,_=import_inputs(inputs/'cases.jsonl',inputs/'rubrics.jsonl')
    totals={'planned_attempts':0,'reused_solver_answers':0,'reused_judgements':0,'pending_solver_answers':0,'pending_judgements':0,'retry_not_before_utc':None,'runs':[]}
    deadlines=[];identities=set()
    for name in cfg['solvers']:
        solver=load_solver_config(inputs/name);require_api_configs(solver,judge)
        if solver['model']==judge.get('model'):raise ValidationError('Judge must differ from solver')
        for configuration in solver['configurations']:
            identity=(solver['model'],configuration['name'])
            if identity in identities:raise ValidationError('Duplicate model/configuration identity')
            identities.add(identity)
        planned=len(cases)*len(solver['configurations'])*solver.get('repetitions',1)
        path=source/Path(name).stem
        reuse=load_reusable_run(path,inputs/'cases.jsonl',inputs/'rubrics.jsonl',solver,judge) if path.exists() else {'solver':{},'judge':{},'deadlines':[]}
        row={'run':path.name,'planned_attempts':planned,'reused_solver_answers':len(reuse['solver']),
             'reused_judgements':len(reuse['judge']),'pending_solver_answers':planned-len(reuse['solver']),
             'pending_judgements':planned-len(reuse['judge'])}
        for k in ['planned_attempts','reused_solver_answers','reused_judgements','pending_solver_answers','pending_judgements']:totals[k]+=row[k]
        totals['runs'].append(row)
        for deadline in reuse['deadlines']:
            try:dt=datetime.fromisoformat(deadline)
            except (TypeError,ValueError):raise ValidationError('Invalid saved retry deadline') from None
            if dt.tzinfo is None:raise ValidationError('Retry deadline must have a timezone')
            deadlines.append(dt)
    if deadlines:totals['retry_not_before_utc']=max(deadlines).isoformat()
    return totals


def resume_experiment(source,output,allow_remote=False):
    """Resume a snapshot into a fresh directory; original files never change."""
    from .experiment import compare_runs, report, write
    source,old_manifest,inputs,cfg=_snapshot(source)
    output=Path(output).resolve()
    if output==source or output.is_relative_to(source):raise ValidationError('Resume output must be outside the source experiment')
    plan=resume_plan(source)
    if plan['retry_not_before_utc'] and datetime.fromisoformat(plan['retry_not_before_utc'])>datetime.now(timezone.utc):
        raise ValidationError('API retry deadline has not elapsed: '+plan['retry_not_before_utc'])
    judge=load_judge_config(inputs/cfg['judge'])
    # Construct only to validate endpoint/auth/transport before writing or calling anything.
    from .providers import make_provider
    from .business.providers import make_judge
    for row,name in zip(plan['runs'],cfg['solvers']):
        solver=load_solver_config(inputs/name)
        if row['pending_solver_answers']>solver.get('max_calls',100):raise ValidationError('Pending solver calls exceed budget')
        if row['pending_judgements']>judge.get('max_calls',100):raise ValidationError('Pending judge calls exceed budget')
        if row['pending_solver_answers']:make_provider(solver['provider'],allow_remote)
    if plan['pending_judgements']:make_judge(judge,allow_remote)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(inputs,output/'inputs')
    # Keep complete original evidence, including failed responses and raw invalid judgements.
    shutil.copytree(source,output/'history'/'source')
    manifest={'state':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
              'planned_runs':old_manifest['planned_runs'],'runs':[],
              'input_hashes':old_manifest['input_hashes'],
              'resume':{'source_manifest_sha256':sha(source/'experiment-manifest.json'),'source_name':source.name,'plan':plan}}
    write(output/'experiment-manifest.json',manifest)
    try:
        for name in cfg['solvers']:
            original=source/Path(name).stem
            target=output/Path(name).stem
            run_business(output/'inputs/cases.jsonl',output/'inputs/rubrics.jsonl',output/'inputs'/name,target,
                         judge_path=output/'inputs'/cfg['judge'],allow_remote_solver=allow_remote,
                         allow_remote_judge=allow_remote,save_raw_judge=True,
                         reuse_from=original if original.exists() else None)
            manifest['runs'].append(target.name);write(output/'experiment-manifest.json',manifest)
        result=compare_runs([output/p for p in manifest['runs']],cfg['policy'])
        report(output,result)
        manifest.update(state='completed',completed_at_utc=datetime.now(timezone.utc).isoformat())
        write(output/'experiment-manifest.json',manifest)
        return result
    except BaseException:
        manifest['state']='interrupted';write(output/'experiment-manifest.json',manifest)
        raise
