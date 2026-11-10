"""Constraint-based selection from complete, hash-verified saved comparisons.

No inference, price guessing, LLM judging, or conversion between timer scopes.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from .datasets import file_sha256
from .reporting import wilson_interval
from .runner import fingerprint, read_records


def finite(value, field, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{field} must be a finite number')
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{field} is outside the allowed range')
    return value


def validate_policy(policy):
    allowed={'min_accuracy','max_error_rate','max_mean_latency_seconds','max_mean_cost_usd',
             'quality_basis','objective','allowed_runs','min_tasks'}
    if not isinstance(policy,dict) or set(policy)-allowed:
        raise ValueError('Policy contains unknown fields')
    p={'min_accuracy':0.0,'max_error_rate':1.0,'quality_basis':'observed',
       'objective':'quality','min_tasks':1,**policy}
    for k in ['min_accuracy','max_error_rate']:finite(p[k],k,maximum=1)
    for k in ['max_mean_latency_seconds','max_mean_cost_usd']:
        if p.get(k) is not None:finite(p[k],k)
    if p['quality_basis'] not in {'observed','wilson_lower'}:
        raise ValueError('quality_basis must be observed or wilson_lower')
    if p['objective'] not in {'quality','latency'}:
        raise ValueError('objective must be quality or latency')
    if type(p['min_tasks']) is not int or p['min_tasks']<1:
        raise ValueError('min_tasks must be a positive integer')
    if 'allowed_runs' in p and (not isinstance(p['allowed_runs'],list) or not p['allowed_runs']
            or any(not isinstance(x,str) for x in p['allowed_runs'])):
        raise ValueError('allowed_runs must be a nonempty list of source run names')
    return p


def load_evidence(comparison):
    comparison=Path(comparison).resolve()
    manifest=json.loads((comparison/'manifest.json').read_text())
    if manifest.get('status')!='completed' or manifest.get('provenance')!='model-inference' or not manifest.get('sources'):
        raise ValueError('Selection requires a completed comparison of real model runs')
    groups=[];all_keys=set();all_records=[];scope=None
    for source in manifest['sources']:
        name=source['name']
        if not isinstance(name,str) or Path(name).name!=name or name in {'.','..'}:
            raise ValueError('Unsafe source run name')
        root=comparison.parent/name
        if root.resolve().parent!=comparison.parent:
            raise ValueError('Source run path escapes comparison parent')
        for filename,hash_key in [('manifest.json','manifest_sha256'),('records.jsonl','records_sha256')]:
            if file_sha256(root/filename)!=source[hash_key]:
                raise ValueError(f'Source {name}/{filename} changed after comparison')
        m=json.loads((root/'manifest.json').read_text())
        if m.get('status')!='completed' or m.get('provenance')!='model-inference':
            raise ValueError('Source is not a completed model-inference run')
        raw=m.get('source_manifest',m.get('config',{}))
        ids=m.get('case_ids',[])
        if not ids or len(ids)!=len(set(ids)):raise ValueError('Missing or duplicate case IDs')
        version=m.get('prompt_version',raw.get('prompt_version'))
        current=(m.get('dataset_sha256'),tuple(sorted(ids)),version)
        if not current[0] or not current[2] or (scope is not None and current!=scope):
            raise ValueError('Datasets, selected IDs or prompt versions are not comparable')
        if m.get('dataset_sha256')!=manifest.get('dataset_sha256') or set(ids)!=set(manifest.get('case_ids',[])):
            raise ValueError('Source scope differs from comparison manifest')
        scope=current
        protocols=raw.get('protocols') or [c['name'] for c in raw.get('configurations',[])]
        reps=raw.get('repetitions',1)
        if not protocols or len(protocols)!=len(set(protocols)) or type(reps) is not int or reps<1:
            raise ValueError('Source lacks an exact protocol/repetition plan')
        expected={(i,c,r) for i in ids for c in protocols for r in range(reps)}
        rows=read_records(root/'records.jsonl')
        keys=[(r['id'],r['config'],r['repetition']) for r in rows]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:
            raise ValueError('Incomplete, duplicate or unexpected coverage in source run')
        if len(rows)!=m.get('planned_attempts') or len(rows)!=m.get('completed_attempts'):
            raise ValueError('Source counts differ from its plan')
        by_config=defaultdict(list)
        for row in rows:
            if type(row.get('correct')) is not bool or row.get('provenance')!='model-inference':
                raise ValueError('Invalid correctness/provenance value')
            if row['correct'] and (row.get('error') or row.get('parse_error')):
                raise ValueError('A failed response cannot also be marked correct')
            if type(row['repetition']) is not int:raise ValueError('Invalid repetition index')
            key=(row['provider'],row['model'],row['config'],row['id'],row['repetition'])
            if key in all_keys:raise ValueError('Duplicate attempt across source runs')
            all_keys.add(key);by_config[(row['provider'],row['model'],row['config'])].append(row)
            all_records.append({**row,'source_run':name})
        latency_scope=raw.get('latency_measurement')
        # A duration without an explicit timer boundary is not interchangeable with wall time.
        runtime_keys=['engine','llama_cpp_python_version','device','platform','threads','n_ctx',
                      'n_batch','n_ubatch','model_quantization','model_revision']
        runtime={k:raw.get(k) for k in runtime_keys}
        latency_group=(fingerprint({'scope':latency_scope,'runtime':runtime})
                       if isinstance(latency_scope,str) and latency_scope.strip() and
                       all(runtime.get(k) is not None for k in ['engine','device','platform','threads']) else None)
        for (provider,model,config),rs in sorted(by_config.items()):
            first=[r for r in rs if r['repetition']==0]
            correct=sum(r['correct'] for r in rs)
            errors=sum(bool(r.get('error') or r.get('parse_error')) for r in rs)
            latency=[r.get('latency_seconds') for r in rs]
            valid_latency=all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in latency)
            cost_contract=raw.get('cost_measurement',{})
            costs=[r.get('cost_usd') for r in rs]
            valid_cost=(cost_contract=={'currency':'USD','scope':'per_inference','basis':'provider_reported'} and
                        all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in costs))
            groups.append({'run':name,'provider':provider,'model':model,'config':config,
                           'attempts':len(rs),'tasks':len(ids),'repetitions':reps,'coverage':1.0,
                           'accuracy':correct/len(rs),'wilson_lower':wilson_interval(sum(r['correct'] for r in first),len(first))[0],
                           'error_rate':errors/len(rs),'mean_latency_seconds':statistics.fmean(latency) if valid_latency else None,
                           'latency_scope':latency_scope,'latency_comparability_group':latency_group,
                           'mean_cost_usd':statistics.fmean(costs) if valid_cost else None,
                           'cost_measurement':cost_contract if valid_cost else None})
    combined=read_records(comparison/'records.jsonl')
    if len(combined)!=manifest.get('planned_attempts') or len(combined)!=manifest.get('completed_attempts'):
        raise ValueError('Comparison coverage differs from its manifest')
    if sorted(map(fingerprint,combined))!=sorted(map(fingerprint,all_records)):
        raise ValueError('Comparison records differ from hash-verified sources')
    if len({g['repetitions'] for g in groups})!=1:
        raise ValueError('Selection requires the same repetition count in all configurations')
    return groups,manifest


def select_configuration(comparison,policy):
    p=validate_policy(policy)
    candidates,manifest=load_evidence(comparison)
    requested=set(p.get('allowed_runs',[]))
    if requested and requested-{c['run'] for c in candidates}:
        raise ValueError('Policy names an unknown source run')
    require_latency=p['objective']=='latency' or p.get('max_mean_latency_seconds') is not None
    eligible=[]
    for c in candidates:
        reasons=[]
        quality=c['accuracy'] if p['quality_basis']=='observed' else c['wilson_lower']
        c['quality_value_used']=quality
        if requested and c['run'] not in requested:reasons.append('run_not_allowed')
        if c['tasks']<p['min_tasks']:reasons.append('too_few_tasks')
        if quality<p['min_accuracy']:reasons.append('quality_below_threshold')
        if c['error_rate']>p['max_error_rate']:reasons.append('error_rate_above_limit')
        if p.get('max_mean_cost_usd') is not None:
            if c['mean_cost_usd'] is None:reasons.append('cost_unknown')
            elif c['mean_cost_usd']>p['max_mean_cost_usd']:reasons.append('cost_above_limit')
        if require_latency:
            if c['mean_latency_seconds'] is None:reasons.append('latency_missing_or_invalid')
            if c['latency_comparability_group'] is None:reasons.append('latency_scope_or_runtime_unknown')
        c['rejection_reasons']=reasons
        if not reasons:eligible.append(c)
    latency_groups={c['latency_comparability_group'] for c in eligible}
    result={'schema_version':'1.0','policy':p,'dataset_sha256':manifest['dataset_sha256'],
            'prompt_version':manifest.get('prompt_version'),'candidates':candidates,'selected':None,
            'model_inference_performed':False,'llm_judge_used':False,
            'warnings':['Selection describes observed saved data, not guaranteed future quality or an SLA.',
                        'Small dev pilot and annotation issues limit generalization. No hidden model judge is used.',
                        'Unknown cost stays unknown; no dollar price is inferred from zero paid API calls.',
                        'Latency limits refer to mean recorded time in its declared scope, not a maximum response deadline.',
                        'Wilson lower bound uses first repetition per task; observed accuracy uses all planned attempts.']}
    if require_latency and len(latency_groups)>1:
        result.update(status='incomparable_latency_scopes',message='No selection: timer/runtime groups differ. Restrict allowed_runs to a comparable group.')
        return result
    if p.get('max_mean_latency_seconds') is not None:
        for c in eligible:
            if c['mean_latency_seconds']>p['max_mean_latency_seconds']:
                c['rejection_reasons'].append('mean_latency_above_limit')
        eligible=[c for c in eligible if not c['rejection_reasons']]
    if not eligible:
        result.update(status='no_eligible_configuration',message='No configuration satisfies all constraints with trustworthy available metrics.')
        return result
    if p['objective']=='quality':
        key=lambda c:(-c['quality_value_used'],c['error_rate'])
        result['ranking_rule']='Highest chosen quality metric, then lowest provider/parse error rate; no latency or cost tiebreak.'
    else:
        key=lambda c:(c['mean_latency_seconds'],-c['quality_value_used'],c['error_rate'])
        result['ranking_rule']='Lowest mean latency within one declared timer/runtime group, then quality and error rate.'
    best=min(key(c) for c in eligible)
    winners=[c for c in eligible if key(c)==best]
    result['eligible_count']=len(eligible)
    if len(winners)>1:
        result.update(status='tied_configurations',message='Several configurations tie; no arbitrary winner selected.',tied=winners)
    else:
        result.update(status='selected',message='One configuration satisfies the constraints and ranking rule.',selected=winners[0])
    return result
