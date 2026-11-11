#!/usr/bin/env python3
"""Opt-in live API smoke: Promptfoo Go models on synthetic business cases.

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
    parser.add_argument('--structured-retries', type=int, choices=[0,1,2], default=0)
    args = parser.parse_args()
    solvers = [x.split('=',1) for x in args.solver] if args.solver else [('luna','gpt-6-luna'),('qwen','qwen3.7-plus'),('glmflash','glm-5.3-flash')]
    if any(len(x)!=2 or not all(x) for x in solvers): parser.error('--solver requires name=model')
    from nir_eval.experiment import main as cli, read, jsonl
    from nir_eval.credentials import GO_BASE_URL, GO_KEY_ENV
    from nir_eval.providers import ProviderUnavailable
    from nir_eval.promptfoo_backend import PromptfooProvider, require_runtime
    require_runtime()
    args.output.mkdir(parents=True, exist_ok=False)
    ledger = args.output / 'requests.jsonl'
    state = {'attempts': 0, 'stop': False}
    originals = {}

    def wrap(cls):
        original = cls.generate
        originals[cls] = original
        def generate(self, messages, model, parameters, context):
            if state['stop'] or state['attempts'] >= 30:
                raise ProviderUnavailable('live_smoke_budget_guard')
            state['attempts'] += 1
            row = {'attempt': state['attempts'], 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                   'engine': 'promptfoo', 'model': model}
            start = time.monotonic()
            try:
                result = original(self, messages, model, parameters, context)
                row.update(status='success', usage=result.usage, metadata=result.metadata)
                # Stop the next call if provider reports a balance charge, not Go usage.
                if result.metadata.get('billing_source') == 'balance':
                    state['stop'] = True
                    row['guard'] = 'stopped_after_reported_balance_billing'
                return result
            except Exception as exc:
                row.update(status='error', error_type=type(exc).__name__)
                # Provider errors contain safe codes; no exception body/URL is persisted.
                from nir_eval.providers import ProviderError
                if isinstance(exc, ProviderError): row['error'] = str(exc)
                raise
            finally:
                row['wall_seconds'] = time.monotonic() - start
                with ledger.open('a') as stream: stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(json.dumps(row, ensure_ascii=False), flush=True)
        cls.generate = generate

    def command(argv):
        if cli([str(x) for x in argv]):
            raise RuntimeError('CLI smoke command failed')

    wrap(PromptfooProvider)
    try:
        for engine in ('promptfoo',):
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
            command(['run', experiment, '--output', args.output / (engine + '-run'), '--allow-remote'])
        if not args.skip_calibration:
            experiment = args.output / 'promptfoo-experiment'
            labels = [json.loads(x) for x in (experiment / 'calibration.jsonl').read_text().splitlines()]
            labels = [x for x in labels if x['case_id'] in {c['id'] for c in cases}]
            jsonl(experiment / 'labels-selected.jsonl', labels)
            command(['calibrate', experiment, '--labels', experiment / 'labels-selected.jsonl',
                     '--output', args.output / 'calibration', '--allow-remote'])
        (args.output / 'smoke-summary.json').write_text(json.dumps({**state, 'max_provider_attempts':30,
            'models':[model for _,model in solvers], 'judge':'glm-5.3',
            'cases_per_engine':len(cases), 'calibration_labels':'synthetic author labels; not independent certification',
            'purpose':'integration smoke; insufficient sample to rank models or engines'}, indent=2)+'\n')
    finally:
        for cls, original in originals.items(): cls.generate = original


if __name__ == '__main__': main()
