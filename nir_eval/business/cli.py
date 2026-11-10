"""CLI integration: `python -m nir_eval.business ...` or main(argv)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from .schema import VERSION, ValidationError, canonical, digest, strict_json, import_inputs, load_jsonl, validate_response
from .providers import make_judge
from .evaluation import evaluate_matrix, build_solver_messages
from .reporting import write_report


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_records(rows):
    """Import existing solver records; never pass gold or rubric fields onward."""
    converted = []
    for row in rows:
        try:
            record = {'id': row['id'], 'response': row.get('raw_output') or '', 'model': row['model'],
                      'configuration': row['config'], 'repetition': row.get('repetition', 0),
                      'status': 'error' if row.get('error') else 'ok',
                      'provenance': row.get('provenance', 'imported-unverified')}
            validate_response(record)
            converted.append(record)
        except (KeyError, TypeError, ValidationError):
            raise ValidationError('Invalid solver record for conversion') from None
    return converted


def _write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(canonical(row) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def parser():
    root = argparse.ArgumentParser(prog='nir-eval business', description='Private business-task evaluation. Judge disabled by default.')
    subs = root.add_subparsers(dest='command', required=True)
    for name in ('validate', 'evaluate', 'export-inputs'):
        sub = subs.add_parser(name)
        sub.add_argument('--cases', required=True)
        sub.add_argument('--rubrics', required=True)
        if name != 'export-inputs':
            sub.add_argument('--responses', required=name == 'evaluate')
        if name in {'evaluate', 'export-inputs'}:
            sub.add_argument('--output', required=True)
        if name == 'evaluate':
            sub.add_argument('--judge-config')
            sub.add_argument('--allow-remote-judge', action='store_true', help='Allow remote transport only after permissions, terms and costs have been checked.')
    sub = subs.add_parser('import-records')
    sub.add_argument('--input', required=True)
    sub.add_argument('--output', required=True)
    sub = subs.add_parser('report')
    sub.add_argument('--records', required=True)
    sub.add_argument('--output', required=True)
    sel = subs.add_parser('select', help='Read-only selection using saved neural-judge scores and coverage')
    sel.add_argument('--run',required=True)
    sel.add_argument('--policy',required=True)
    sel.add_argument('--output',required=True)
    sub = subs.add_parser('run', help='Cases → actual solver → neural judge → report')
    sub.add_argument('--cases', required=True)
    sub.add_argument('--rubrics', required=True)
    sub.add_argument('--solver-config', required=True)
    sub.add_argument('--judge-config')
    sub.add_argument('--output', required=True)
    sub.add_argument('--allow-remote-solver', action='store_true')
    sub.add_argument('--allow-remote-judge', action='store_true')
    sub.add_argument('--save-raw-judge', action='store_true', help='Save raw judge text locally; may contain user data')
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'select':
            from ..business_selection import select_business_run
            result=select_business_run(args.run,strict_json(Path(args.policy).read_text()))
            with Path(args.output).open('x',encoding='utf-8') as stream:
                stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(canonical({'status':result['status'],'eligible_count':result['eligible_count']}))
            return 0
        if args.command == 'run':
            from .running import run_business
            result = run_business(args.cases,args.rubrics,args.solver_config,args.output,
                                  judge_path=args.judge_config,allow_remote_solver=args.allow_remote_solver,
                                  allow_remote_judge=args.allow_remote_judge,save_raw_judge=args.save_raw_judge)
            print(canonical(result['summary']))
            return 0
        if args.command == 'report':
            result = write_report(load_jsonl(args.records), args.output)
            print(canonical(result['summary']))
            return 0
        if args.command == 'import-records':
            records = convert_records(load_jsonl(args.input))
            _write_jsonl(args.output, records)
            print(canonical({'imported': len(records), 'new_inference': False}))
            return 0
        cases, rubrics, responses = import_inputs(args.cases, args.rubrics, getattr(args, 'responses', None))
        if args.command == 'validate':
            print(canonical({'valid': True, 'cases': len(cases), 'rubrics': len(rubrics), 'responses': len(responses)}))
            return 0
        if args.command == 'export-inputs':
            _write_jsonl(args.output, ({'id': case['id'], 'messages': build_solver_messages(case)} for case in cases))
            print(canonical({'exported': len(cases), 'reference_and_rubric_excluded': True}))
            return 0
        config = strict_json(Path(args.judge_config).read_text('utf-8')) if args.judge_config else {'type': 'disabled'}
        judge = make_judge(config, allow_remote=args.allow_remote_judge)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        manifest = {'schema_version': VERSION, 'state': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                    'inputs': {'cases_sha256': _file_hash(args.cases), 'rubrics_sha256': _file_hash(args.rubrics),
                               'responses_sha256': _file_hash(args.responses)},
                    'judge': {'type': config['type'], 'model': judge.model, 'provenance': judge.kind,
                              'config_sha256': digest(config), 'allow_remote': args.allow_remote_judge},
                    'code_sha256': {p.name: _file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                    'notes': ['No paid API called by default. An explicit transport flag is not permission or a guarantee of free service.',
                              'Original user inputs remain in their source files; this manifest stores hashes only.']}
        manifest_path = output / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        rows = []
        with (output / 'records.jsonl').open('x', encoding='utf-8') as stream:
            for row in evaluate_matrix(cases, rubrics, responses, judge, max_calls=config.get('max_calls', 100)):
                stream.write(canonical(row) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                rows.append(row)
        summary = write_report(rows, output)
        manifest.update(state='completed', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        record_count=len(rows), records_sha256=_file_hash(output / 'records.jsonl'),
                        summary_sha256=_file_hash(output / 'summary.json'))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(canonical(summary['summary']))
        return 0
    except (ValidationError, FileExistsError, FileNotFoundError, PermissionError, KeyError, TypeError, ImportError):
        # Do not expose filenames, content, credentials or provider exception bodies.
        print('Business evaluation stopped: invalid inputs/configuration, unavailable dependency, or output path already exists. Validate inputs and use a fresh output directory.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
