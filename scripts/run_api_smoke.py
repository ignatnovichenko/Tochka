#!/usr/bin/env python3
"""Opt-in live API smoke: Native Go models on synthetic business cases.

Never runs on import or in CI. Maximum 30 provider attempts, no automatic
transport retry or fallback endpoint. Explicit judge repairs count toward the cap. Credentials are read by Tochka, never saved here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--solver', action='append', help='name=model; repeatable')
    parser.add_argument('--case-id', action='append')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--judge-max-tokens', type=int, default=3072)
    parser.add_argument('--skip-calibration', action='store_true')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--structured-retries', type=int, choices=[0,1,2], default=0)
    args = parser.parse_args()
    solvers = [x.split('=',1) for x in args.solver] if args.solver else [('luna','gpt-6-luna'),('qwen','qwen3.7-plus'),('glmflash','glm-5.3-flash')]
    if any(len(x)!=2 or not all(x) for x in solvers): parser.error('--solver requires name=model')
    from nir_eval.experiment import main as cli, read, jsonl
    from nir_eval.credentials import GO_BASE_URL, GO_KEY_ENV
    from nir_eval.providers import ProviderUnavailable
    from nir_eval import native_backend
    args.output.mkdir(parents=True, exist_ok=False)
    ledger = args.output / 'requests.jsonl'
    state = {'attempts': 0, 'stop': False}
    state['reserved'] = 0
    original = native_backend.evaluate_jobs
    batch_number = 0

    def guarded(jobs, max_concurrency=4, on_result=None, *, on_started=None):
        nonlocal batch_number
        if state['stop'] or state['reserved'] + len(jobs) > 30:
            raise ProviderUnavailable('live_smoke_budget_guard')
        state['reserved'] += len(jobs)
        batch_number += 1
        number = batch_number
        lookup = {j['job_id']:j for j in jobs}
        starts = {}
        def started(job_id, at):
            starts[job_id] = at
            state['attempts'] += 1
            if on_started: on_started(job_id, at)
        def completed(job_id, result, seconds):
            from nir_eval.providers import Generation
            row={'batch':number,'job_id':job_id,'model':lookup[job_id]['model'],
                 'started_at_utc':starts.get(job_id),'engine':'native.threadpool','wall_seconds':seconds}
            if isinstance(result,Generation):
                row.update(status='success',usage=result.usage,metadata=result.metadata)
                if result.metadata.get('billing_source')=='balance': state['stop']=True
            else:
                row.update(status='error',error=str(result),error_type=type(result).__name__,
                           request_attempted=getattr(result,'request_attempted',True))
            with ledger.open('a') as stream: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            if on_result: on_result(job_id,result,seconds)
            print(json.dumps(row,ensure_ascii=False),flush=True)
            if state['stop']: raise ProviderUnavailable('live_smoke_balance_billing_guard')
        return original(jobs,max_concurrency,completed,on_started=started)

    def command(argv):
        if cli([str(x) for x in argv]):
            raise RuntimeError('CLI smoke command failed')

    native_backend.evaluate_jobs = guarded
    try:
        for engine in ('native',):
            experiment = args.output / (engine + '-experiment')
            command(['init', experiment, '--demo'])
            cases = [json.loads(x) for x in (experiment / 'cases.jsonl').read_text().splitlines()]
            cases = [c for c in cases if c['id'] in args.case_id] if args.case_id else [c for c in cases if c['id'] != 'meeting_summary_01']
            jsonl(experiment / 'cases.jsonl', cases)
            connection = ['--base-url', GO_BASE_URL, '--api-key-env', GO_KEY_ENV,
                          '--temperature', '0', '--timeout', '120']
            for name, model in solvers:
                command(['solver-add', experiment, '--name', name, '--model', model,
                         '--max-tokens', args.max_tokens, '--max-calls', str(len(cases)), *connection])
            command(['judge-set', experiment, '--model', 'glm-5.3', '--max-tokens', args.judge_max_tokens,
                     '--max-calls', '12', '--structured-retries', str(args.structured_retries), *connection])
            command(['policy-set', experiment, '--min-quality', '0.8', '--min-coverage', '1'])
            command(['plan', experiment])
            command(['run', experiment, '--output', args.output / (engine + '-run'), '--allow-remote','--concurrency',args.concurrency])
        if not args.skip_calibration:
            experiment = args.output / 'native-experiment'
            labels = [json.loads(x) for x in (experiment / 'calibration.jsonl').read_text().splitlines()]
            labels = [x for x in labels if x['case_id'] in {c['id'] for c in cases}]
            jsonl(experiment / 'labels-selected.jsonl', labels)
            command(['calibrate', experiment, '--labels', experiment / 'labels-selected.jsonl',
                     '--output', args.output / 'calibration', '--allow-remote','--concurrency',args.concurrency])
        (args.output / 'smoke-summary.json').write_text(json.dumps({**state, 'max_provider_attempts':30,
            'models':[model for _,model in solvers], 'judge':'glm-5.3',
            'max_concurrency':args.concurrency,'cases_per_engine':len(cases), 'calibration_labels':'synthetic author labels; not independent certification',
            'purpose':'integration smoke; insufficient sample to rank models or engines'}, indent=2)+'\n')
    finally:
        native_backend.evaluate_jobs = original


if __name__ == '__main__': main()
