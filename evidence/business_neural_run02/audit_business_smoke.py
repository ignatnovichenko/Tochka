#!/usr/bin/env python3
"""Read-only integrity audit of the actual native business smoke, no inference."""
from pathlib import Path
import datetime
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
MVP = HERE.parents[1] / 'mvp'
RUN = MVP / 'runs' / 'business_neural_smoke_run02'
sys.path.insert(0, str(HERE / 'source_snapshot'))
from nir_eval.business.schema import import_inputs, digest
from nir_eval.business.evaluation import build_solver_messages, build_judge_messages, parse_judgement


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    manifest = json.loads((RUN / 'manifest.json').read_text())
    assert manifest['state'] == 'completed'
    assert manifest['target_quality_method'] == 'neural_judge_only'
    assert manifest['planned_attempts'] == manifest['evaluated_attempts'] == 2
    source = json.loads((HERE / 'source_manifest.json').read_text())['files']
    assert source == manifest['code_sha256']
    for relative, expected in source.items():
        assert sha(HERE / 'source_snapshot' / 'nir_eval' / relative) == expected
    for key, name in [('cases_sha256', 'business_smoke_cases.jsonl'), ('rubrics_sha256', 'business_rubrics.jsonl')]:
        assert sha(HERE / name) == manifest[key]
    for key, name in [('solver_records_sha256', 'solver_records.jsonl'), ('responses_sha256', 'responses.jsonl'), ('records_sha256', 'records.jsonl')]:
        assert sha(RUN / name) == manifest[key]
    cases, rubrics, responses = import_inputs(HERE / 'business_smoke_cases.jsonl', HERE / 'business_rubrics.jsonl', RUN / 'responses.jsonl')
    solver = {r['id']: r for r in readlines(RUN / 'solver_records.jsonl')}
    judge = {r['id']: r for r in readlines(RUN / 'judge_raw.jsonl')}
    records = {r['id']: r for r in readlines(RUN / 'records.jsonl')}
    responses = {r['id']: r for r in responses}
    assert len(solver) == len(judge) == len(records) == len(responses) == 2
    assert set(solver) == set(judge) == set(records) == set(responses) == {c['id'] for c in cases}
    per_case = []
    for case in cases:
        cid = case['id']; s = solver[cid]; j = judge[cid]; r = records[cid]
        assert s['error'] is None and s['provenance'] == 'model-inference'
        assert responses[cid]['status'] == 'ok' and responses[cid]['response'] == s['raw_output']
        assert s['messages'] == build_solver_messages(case)
        # Exact reconstruction above is the leakage check: only task/context enter solver.
        assert s['prompt_sha256'] == digest(s['messages'])
        assert s['provider_metadata']['model_revision'] == '7ae557604adf67be50417f59c2c2f167def9a775'
        assert s['provider_metadata']['torch_version'].startswith('2.6.0')
        assert s['provider_metadata']['one_request_per_process'] is True
        assert s['provider_metadata']['retries'] == s['provider_metadata']['repair_calls'] == 0
        messages = build_judge_messages(case, rubrics[case['rubric_id']], s['raw_output'])
        assert j['prompt_sha256'] == r['judge']['prompt_sha256'] == digest(messages)
        assert r['judge']['output_sha256'] == digest(j['raw_output'])
        assert r['judge']['usage'] == j['usage']
        meta = j['metadata']
        assert meta['engine_version'] == '0.3.16'
        assert meta['model_revision'] == 'bb5d59e06d9551d752d08b292a50eb208b07ab1f'
        assert meta['n_gpu_layers'] == 0 and meta['n_threads'] == 4 and meta['n_ctx'] == 4096
        assert meta['effective_generation_config']['max_tokens'] == 512
        assert meta['effective_generation_config']['temperature'] == 0
        assert meta['inputs_truncated'] is False and meta['retries'] == meta['repair_calls'] == 0
        rendered = ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n'
        assert meta['prompt_sha256'] == hashlib.sha256(rendered.encode()).hexdigest()
        validated = False
        try:
            parsed = parse_judgement(j['raw_output'], case, rubrics[case['rubric_id']], s['raw_output'])
            validated = meta['finish_reason'] not in {'length', 'content_filter'}
        except ValueError:
            parsed = None
        if validated:
            assert r['judge']['status'] == 'evaluated'
            assert all(r['judge'][k] == parsed[k] for k in ['criteria','critical_errors','score','passed'])
            assert r['score'] == parsed['score'] and r['passed'] == parsed['passed']
        else:
            assert r['judge']['status'] == 'error' and r['score'] is None and r['passed'] is None
        per_case.append({'id':cid,'solver_tokens':s['token_usage'],'solver_seconds':s['latency_seconds'],
            'judge_tokens':j['usage'],'judge_seconds':r['judge']['latency_seconds'],
            'judge_finish_reason':meta['finish_reason'],'judge_status':r['judge']['status'],
            'judge_reason_code':r['judge']['reason_code'],'score':r['score'],'passed':r['passed']})
    result = {'audit_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'audit_passed':True,'actual_solver_calls':2,'actual_judge_calls':2,
              'no_import_replay_or_control':True,'solver_reference_rubric_leakage_check':'exact messages reconstruction passed',
              'quality_decision':'neural judge only; audit revalidates schema/evidence/aggregation, not answer correctness',
              'source_and_record_hashes_verified':True,'per_case':per_case,
              'limitations':['Two synthetic cases are a technical integration smoke, not validation of judge reliability.',
                  'No blind human reference study, adversarial study, repeated judging or statistical quality conclusion.',
                  'Command latency includes model loading for each subprocess; not comparable to persistent-model throughput.',
                  'Source and evidence are snapshotted. Original results are retained unchanged.']}
    (HERE / 'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
