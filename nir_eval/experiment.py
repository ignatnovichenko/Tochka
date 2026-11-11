"""User-facing experiment workflow. No code editing required."""
import argparse
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from .business.schema import (ValidationError, canonical, strict_json, load_jsonl,
    import_inputs, validate_case, validate_rubric, identifier, digest)
from .business.running import run_business, load_judge_config, sha
from .business.providers import make_judge
from .business.evaluation import evaluate
from .business_selection import select_business_run, validate_business_policy
from .api_support import validate_api_config
from .business_report import write_business_report, terminal_summary
from .credentials import interactive_credentials
from .provenance import source_hashes


def read(path):
    return strict_json(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def jsonl(path, rows):
    Path(path).write_text(''.join(canonical(r) + '\n' for r in rows), encoding='utf-8')


def config(root):
    return read(root / 'experiment.json')


def existing_rows(path):
    return load_jsonl(path) if path.exists() and path.stat().st_size else []


def configure_provider(args):
    if args.provider not in {'api', 'local'}:
        raise ValidationError('Подключение моделей доступно только через API. Используйте --provider api.')
    provider = {'type': 'openai-compatible', 'base_url': args.base_url or '',
                'api_key_env': args.api_key_env, 'timeout_seconds': args.timeout}
    provider['engine'] = 'promptfoo'
    for key in ('input_price_per_million', 'output_price_per_million'):
        value = getattr(args, key, None)
        if value is not None: provider[key] = value
    local = validate_api_config(provider, allow_remote=True)
    if args.provider == 'local' and not local:
        raise ValidationError('Provider local requires a localhost endpoint; use --provider api for HTTPS')
    parameters = {'temperature': args.temperature if args.temperature is not None else 0,
                  'max_tokens': args.max_tokens if args.max_tokens is not None else (8192 if args.command == 'judge-set' else 2048)}
    if not math.isfinite(parameters['temperature']) or not 0 <= parameters['temperature'] <= 2 or not 1 <= parameters['max_tokens'] <= 131072:
        raise ValidationError('Invalid generation parameters')
    return provider, parameters


def experiment_plan(root, allow_remote=False, check_credentials=False):
    from .business.running import load_solver_config
    root = Path(root).resolve()
    cfg = config(root)
    validate_business_policy(cfg.get('policy'))
    cases, _, _ = import_inputs(root / 'cases.jsonl', root / 'rubrics.jsonl')
    if not cfg.get('solvers') or not cfg.get('judge'):
        raise ValidationError('Configure solvers and a separate judge first')
    names = [cfg['judge'], *cfg['solvers']]
    if any(not isinstance(n,str) or Path(n).name != n for n in names) or len(set(names)) != len(names):
        raise ValidationError('Configuration files must have unique plain filenames')
    judge = load_judge_config(root / cfg['judge'])
    if judge.get('type') != 'generation-provider' or judge.get('provider',{}).get('type') != 'openai-compatible':
        raise ValidationError('Требуется API-судья; старые CLI-конфигурации доступны только для чтения результатов.')
    validate_api_config(judge['provider'], allow_remote=allow_remote, require_key=check_credentials)
    solver_calls = 0
    configs = []
    identities = set()
    for filename in cfg['solvers']:
        sc = load_solver_config(root / filename)
        if sc['provider'].get('type') != 'openai-compatible':
            raise ValidationError('Новые прогоны используют только API. Перенастройте solver-add --provider api.')
        validate_api_config(sc['provider'], allow_remote=allow_remote, require_key=check_credentials)
        if sc['model'] == judge['model']:
            raise ValidationError('Judge must be a different model from every solver')
        for item in sc['configurations']:
            identity = (sc['model'], item['name'])
            if identity in identities: raise ValidationError('Duplicate model/configuration identity')
            identities.add(identity)
        n = len(cases) * len(sc['configurations']) * sc.get('repetitions',1)
        if n > sc.get('max_calls',100) or n > judge.get('max_calls',100):
            raise ValidationError('Planned calls exceed configured budget')
        solver_calls += n
        configs.append({'model':sc['model'],'calls':n,'config_file':filename})
    if check_credentials:
        from .promptfoo_backend import require_runtime
        require_runtime()
    return {'cases':len(cases),'configurations':configs,'solver_calls':solver_calls,
            'judge_calls':solver_calls,'maximum_judge_calls':sum(min(c['calls']*(1+judge.get('structured_retries',0)),judge.get('max_calls',100)) for c in configs),
            'estimated_total_cost_usd':None,'note':'Total cost unknown before actual usage; subscription fees are not allocated.'}


def compare_runs(paths, policy):
    if not paths:
        raise ValidationError('No completed runs')
    selections = [select_business_run(p, policy) for p in paths]
    for field in ('cases_sha256', 'rubrics_sha256', 'judge_config_sha256'):
        if len({s[field] for s in selections}) != 1:
            raise ValidationError('Cannot compare different tasks, rubrics or judges')
    candidates = []
    identities = set()
    for path, result in zip(paths, selections):
        for candidate in result['candidates']:
            identity = (candidate['model'], candidate['configuration'])
            if identity in identities:
                raise ValidationError('Duplicate configuration across runs')
            identities.add(identity)
            candidates.append({**candidate, 'run': Path(path).name})
    eligible = [c for c in candidates if not c['rejection_reasons']]
    if policy.get('max_mean_solver_latency_seconds') is not None or policy.get('max_mean_total_latency_seconds') is not None:
        if len({c['solver_latency_scope'] for c in eligible}) > 1:
            for c in eligible:
                c['rejection_reasons'].append('incomparable_solver_latency_scopes')
            eligible = []
    best = max((c['mean_judge_score'] for c in eligible), default=None)
    winners = [c for c in eligible if math.isclose(c['mean_judge_score'], best, rel_tol=0, abs_tol=1e-12)] if best is not None else []
    return {'schema_version': 'business-experiment-v1', 'policy': validate_business_policy(policy),
            'status': 'no_eligible_configuration' if not winners else ('selected' if len(winners) == 1 else 'tied_configurations'),
            'eligible_count': len(eligible), 'selected': winners[0] if len(winners) == 1 else None,
            'tied': winners if len(winners) > 1 else [], 'candidates': candidates,
            'sources': [{'run': Path(p).name, 'manifest_sha256': r['source_manifest_sha256']} for p, r in zip(paths, selections)],
            'warnings': selections[0]['warnings']}


def report(output, result, source_root=None):
    return write_business_report(Path(output), result, source_root=source_root)


def run_experiment(root, output, allow_remote=False):
    root, output = Path(root).resolve(), Path(output)
    experiment_plan(root, allow_remote=allow_remote, check_credentials=True)
    cfg = config(root)
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'inputs'; inputs.mkdir()
    for name in ['experiment.json', 'cases.jsonl', 'rubrics.jsonl', cfg['judge'], *cfg['solvers']]:
        shutil.copy2(root / name, inputs / Path(name).name)
    manifest = {'state': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'planned_runs': [Path(p).stem for p in cfg['solvers']], 'runs': [],
                'input_hashes': {p.name: sha(p) for p in inputs.iterdir()}}
    write(output / 'experiment-manifest.json', manifest)
    try:
        for solver in cfg['solvers']:
            target = output / Path(solver).stem
            run_business(inputs / 'cases.jsonl', inputs / 'rubrics.jsonl', inputs / solver, target,
                         judge_path=inputs / cfg['judge'], allow_remote_solver=allow_remote,
                         allow_remote_judge=allow_remote, save_raw_judge=True)
            manifest['runs'].append(target.name)
            write(output / 'experiment-manifest.json', manifest)
        result = compare_runs([output / x for x in manifest['runs']], cfg['policy'])
        report(output, result)
        manifest.update(state='completed', completed_at_utc=datetime.now(timezone.utc).isoformat())
        write(output / 'experiment-manifest.json', manifest)
        return result
    except BaseException:
        manifest['state'] = 'interrupted'; write(output / 'experiment-manifest.json', manifest)
        raise


def calibrate(root, labels_path, output, allow_remote=False):
    cfg = config(root)
    cases, rubrics, _ = import_inputs(root / 'cases.jsonl', root / 'rubrics.jsonl')
    cases = {c['id']: c for c in cases}
    labels = load_jsonl(labels_path)
    seen = set()
    for x in labels:
        if not isinstance(x.get('candidate'), str) or x.get('case_id') not in cases or type(x.get('expected_pass')) is not bool or type(x.get('expected_critical')) is not bool:
            raise ValidationError('Labels require case_id, candidate, expected_pass and expected_critical')
        identifier(x.get('id'))
        if x['id'] in seen:
            raise ValidationError('Duplicate label ID')
        seen.add(x['id'])
    jc = load_judge_config(root / cfg['judge'])
    if len(labels) > jc.get('max_calls', 100):
        raise ValidationError('Calibration exceeds judge budget')
    if jc.get('type') != 'generation-provider' or jc.get('provider',{}).get('type') != 'openai-compatible':
        raise ValidationError('Configure an API judge before calibration')
    validate_api_config(jc['provider'], allow_remote=allow_remote, require_key=True)
    actual_judge = make_judge(jc, allow_remote)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(labels_path, output / 'labels.jsonl')
    for source in [root / 'cases.jsonl', root / 'rubrics.jsonl', root / cfg['judge']]:
        shutil.copy2(source, output / source.name)
    write(output / 'manifest.json', {'state': 'running', 'judge_config_sha256': digest(jc),
          'started_at_utc': datetime.now(timezone.utc).isoformat(),
          'inputs': {p.name: sha(p) for p in output.iterdir() if p.is_file()},
          'code_sha256': source_hashes()})
    try:
        rows = []
        with (output / 'raw.jsonl').open('w', encoding='utf-8') as raw, (output / 'records.jsonl').open('w', encoding='utf-8') as records:
            class Recording:
                kind = actual_judge.kind
                model = actual_judge.model
                structured_retries = getattr(actual_judge, 'structured_retries', 0)
                def judge(self, messages, case_id):
                    reply = actual_judge.judge(messages, case_id)
                    raw.write(canonical({'case_id': case_id, 'raw_output': reply.text, 'metadata': reply.metadata}) + '\n'); raw.flush()
                    return reply
            for label in labels:
                case = cases[label['case_id']]
                row = evaluate(case, rubrics[case['rubric_id']], label['candidate'], Recording())
                evaluated = row['judge']['status'] == 'evaluated'
                critical = any(c['detected'] for c in row['judge'].get('critical_errors', [])) if evaluated else None
                row.update(label_id=label['id'], expected_pass=label['expected_pass'], expected_critical=label['expected_critical'],
                           pass_agreement=(row['passed'] == label['expected_pass']) if evaluated else None,
                           critical_agreement=(critical == label['expected_critical']) if evaluated else None)
                rows.append(row); records.write(canonical(row) + '\n'); records.flush()
                print(f'calibration {len(rows)}/{len(labels)}: {row["judge"]["status"]}', flush=True)
        known = [r for r in rows if r['pass_agreement'] is not None]
        summary = {'total': len(rows), 'evaluated': len(known), 'unknown': len(rows) - len(known),
                   'coverage': len(known) / len(rows),
                   'pass_agreement': sum(r['pass_agreement'] for r in known) / len(known) if known else None,
                   'critical_agreement': sum(r['critical_agreement'] for r in known) / len(known) if known else None,
                   'judge_model': actual_judge.model, 'judge_config_sha256': digest(jc), 'labels_sha256': sha(labels_path),
                   'limitation': 'Small synthetic author-labelled control set; not independent human certification.'}
        write(output / 'summary.json', summary)
        manifest = read(output / 'manifest.json'); manifest.update(state='completed', records_sha256=sha(output / 'records.jsonl'), summary_sha256=sha(output / 'summary.json'))
        write(output / 'manifest.json', manifest)
        return summary
    except BaseException as exc:
        manifest = read(output / 'manifest.json')
        manifest.update(state='interrupted', failure_type=type(exc).__name__, updated_at_utc=datetime.now(timezone.utc).isoformat())
        write(output / 'manifest.json', manifest)
        raise


def parser():
    p = argparse.ArgumentParser(prog='tochka', description='Полный цикл бизнес-эксперимента без редактирования кода')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('wizard', help='Пошаговый мастер настройки и запуска')
    sub.add_parser('setup', help='Установить Promptfoo; setup --check проверяет готовность')
    sub.add_parser('keys', help='Управление сохранёнными API-ключами; keys --help')
    init = sub.add_parser('init', help='Создать эксперимент'); init.add_argument('directory'); init.add_argument('--demo', action='store_true')
    models = sub.add_parser('models', help='Показать настроенные API-модели без сетевых запросов'); models.add_argument('--directory')
    plan = sub.add_parser('plan', help='План вызовов без inference'); plan.add_argument('directory')
    plan.add_argument('--allow-remote', action='store_true')
    resume = sub.add_parser('resume', help='Продолжить прогон в новой папке, сохранив старый'); resume.add_argument('run'); resume.add_argument('--output', required=True); resume.add_argument('--allow-remote', action='store_true'); resume.add_argument('--plan-only', action='store_true')
    reselect = sub.add_parser('reselect', help='Пересчитать выбор и отчёт без новых модельных запросов'); reselect.add_argument('run'); reselect.add_argument('--output', required=True)
    for flag in ['min-quality','min-coverage','min-pass-rate','max-latency','max-cost','max-critical-rate']:
        reselect.add_argument('--'+flag,type=float)
    reselect.add_argument('--policy')
    for name in ['import', 'task-add', 'rubric-add', 'solver-add', 'judge-set', 'policy-set', 'show', 'run', 'calibrate']:
        q = sub.add_parser(name); q.add_argument('directory')
        if name == 'import':
            q.add_argument('--cases', required=True); q.add_argument('--rubrics', required=True)
        if name == 'task-add':
            q.add_argument('--id', required=True); q.add_argument('--task', required=True); q.add_argument('--rubric', required=True)
            q.add_argument('--context', default=''); q.add_argument('--document', action='append', default=[], help='UTF-8 text/Markdown document, repeatable'); q.add_argument('--reference')
        if name == 'rubric-add':
            q.add_argument('--id', required=True); q.add_argument('--criterion', action='append', required=True, help='id=description; repeatable')
            q.add_argument('--critical', action='append', default=[], help='id=description; repeatable'); q.add_argument('--pass-threshold', type=float, default=.75)
        if name in ['solver-add', 'judge-set']:
            q.add_argument('--provider', choices=['api', 'local'], default='api'); q.add_argument('--model', required=True)
            q.add_argument('--base-url'); q.add_argument('--api-key-env', default='NIR_API_KEY'); q.add_argument('--input-price-per-million', type=float); q.add_argument('--output-price-per-million', type=float); q.add_argument('--temperature', type=float); q.add_argument('--max-tokens', type=int)
            q.add_argument('--timeout', type=float, default=180); q.add_argument('--max-calls', type=int, default=100)
            if name == 'judge-set':
                q.add_argument('--structured-retries', type=int, choices=[0,1,2], default=0, help='Explicit retries for invalid judge JSON/evidence only; count against max-calls')
            if name == 'solver-add':
                q.add_argument('--name', required=True); q.add_argument('--instruction', default=''); q.add_argument('--repetitions', type=int, default=1)
        if name == 'policy-set':
            q.add_argument('--min-quality', type=float, default=.75); q.add_argument('--min-coverage', type=float, default=1)
            q.add_argument('--min-pass-rate', type=float); q.add_argument('--max-latency', type=float)
            q.add_argument('--max-cost', type=float); q.add_argument('--max-critical-rate', type=float, default=0)
        if name in ['run', 'calibrate']:
            q.add_argument('--output', required=True); q.add_argument('--allow-remote', action='store_true')
        if name == 'calibrate': q.add_argument('--labels', required=True)
    q = sub.add_parser('results'); q.add_argument('run'); q.add_argument('--json', action='store_true')
    q = sub.add_parser('select'); q.add_argument('run'); q.add_argument('--policy', required=True); q.add_argument('--output', required=True)
    return p


@interactive_credentials
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'setup':
        from .setup_cli import main as setup_main
        return setup_main(argv[1:])
    if argv and argv[0] == 'keys':
        from .keys_cli import main as keys_main
        return keys_main(argv[1:])
    if argv and argv[0] == 'wizard':
        from .wizard import main as wizard_main
        return wizard_main(argv[1:])
    args = parser().parse_args(argv)
    try:
        if args.command == 'models':
            if args.directory:
                root = Path(args.directory)
                cfg = config(root)
                for filename in [*cfg.get('solvers',[]), *([cfg['judge']] if cfg.get('judge') else [])]:
                    print(read(root / filename)['model'])
            else:
                print('Укажите --directory EXPERIMENT для списка настроенных API-моделей. ID моделей берите у вашего API-провайдера.')
            return 0
        if args.command == 'plan':
            print(json.dumps(experiment_plan(args.directory,allow_remote=True,check_credentials=False),ensure_ascii=False,indent=2)); return 0
        if args.command == 'resume':
            from .resume import resume_experiment, resume_plan
            if args.plan_only:
                print(json.dumps(resume_plan(Path(args.run)),ensure_ascii=False,indent=2)); return 0
            result = resume_experiment(Path(args.run),Path(args.output),args.allow_remote)
            print(terminal_summary(result)); print('Отчёт: ' + str(Path(args.output)/'report.html')); return 0
        if args.command == 'reselect':
            source = Path(args.run).resolve()
            manifest = read(source/'experiment-manifest.json')
            if manifest.get('state') != 'completed': raise ValidationError('Use a completed run for selection; resume an interrupted run first')
            for name, expected in manifest.get('input_hashes',{}).items():
                if Path(name).name != name or sha(source/'inputs'/name) != expected:
                    raise ValidationError('Source input fingerprint mismatch')
            names=manifest['runs']
            if len(names)!=len(set(names)) or set(names)!=set(manifest['planned_runs']) or any(Path(x).name!=x for x in names):
                raise ValidationError('Experiment run plan is incomplete')
            policy = read(args.policy) if args.policy else read(source/'comparison.json')['policy'].copy()
            for arg,key in [('min_quality','min_mean_judge_score'),('min_coverage','min_judge_coverage'),('min_pass_rate','min_judge_pass_rate'),('max_latency','max_mean_solver_latency_seconds'),('max_cost','max_mean_cost_usd'),('max_critical_rate','max_judge_critical_error_rate')]:
                value = getattr(args,arg)
                if value is not None: policy[key]=value
            policy = validate_business_policy(policy)
            result=compare_runs([source/x for x in names],policy)
            output=Path(args.output).resolve()
            if output == source or output.is_relative_to(source):
                raise ValidationError('Новый отчёт должен находиться вне папки исходного прогона')
            output.mkdir(parents=True,exist_ok=False)
            report(output,result,source_root=source)
            write(output/'reselection.json',{'source_manifest_sha256':sha(source/'experiment-manifest.json'),'model_calls_performed':False,'policy':policy})
            print(terminal_summary(result)); print('Отчёт: '+str(output/'report.html')); return 0
        if args.command == 'results':
            result = read(Path(args.run) / 'comparison.json')
            print(json.dumps(result,ensure_ascii=False,indent=2) if args.json else terminal_summary(result)); return 0
        if args.command == 'select':
            run = Path(args.run); manifest = read(run / 'experiment-manifest.json')
            if manifest['state'] != 'completed': raise ValidationError('Experiment is not complete')
            result = compare_runs([run / x for x in manifest['runs']], read(args.policy))
            with Path(args.output).open('x', encoding='utf-8') as f: f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            print(result['status']); return 0
        root = Path(args.directory).resolve()
        if args.command == 'init':
            root.mkdir(parents=True, exist_ok=False)
            cfg = {'schema_version': 1, 'solvers': [], 'judge': None, 'policy': validate_business_policy({'min_mean_judge_score': .75})}
            if args.demo:
                source = Path(__file__).parent / 'demo'
                for name in ['cases.jsonl', 'rubrics.jsonl', 'calibration.jsonl']:
                    shutil.copy2(source / name, root / name)
            else:
                (root / 'cases.jsonl').touch(); (root / 'rubrics.jsonl').touch()
            write(root / 'experiment.json', cfg); print(f'Создан эксперимент: {root}'); return 0
        cfg = config(root)
        if args.command == 'show':
            print(json.dumps(cfg, ensure_ascii=False, indent=2)); return 0
        if args.command == 'import':
            import_inputs(args.cases, args.rubrics)
            shutil.copyfile(args.cases, root / 'cases.jsonl'); shutil.copyfile(args.rubrics, root / 'rubrics.jsonl')
        elif args.command == 'rubric-add':
            def items(values):
                out = []
                for v in values:
                    k, sep, description = v.partition('=')
                    if not sep: raise ValidationError('Use id=description')
                    out.append({'id': k, 'description': description})
                return out
            row = validate_rubric({'id': args.id, 'criteria': items(args.criterion), 'critical_errors': items(args.critical), 'pass_threshold': args.pass_threshold})
            rows = existing_rows(root / 'rubrics.jsonl')
            if any(r['id'] == args.id for r in rows): raise ValidationError('Rubric ID already exists')
            jsonl(root / 'rubrics.jsonl', rows + [row])
        elif args.command == 'task-add':
            rubrics = {r['id']: r for r in existing_rows(root / 'rubrics.jsonl')}
            context = args.context + ''.join('\n\n' + Path(p).read_text(encoding='utf-8') for p in args.document)
            row = {'id': args.id, 'task': args.task, 'context': context, 'rubric_id': args.rubric}
            if args.reference is not None: row['reference'] = args.reference
            validate_case(row)
            if args.rubric not in rubrics: raise ValidationError('Unknown rubric ID')
            rows = existing_rows(root / 'cases.jsonl')
            if any(r['id'] == args.id for r in rows): raise ValidationError('Task ID already exists')
            jsonl(root / 'cases.jsonl', rows + [row])
        elif args.command in ['solver-add', 'judge-set']:
            if not 1 <= args.max_calls <= 10000 or not 0 < args.timeout <= 3600: raise ValidationError('Invalid call budget or timeout')
            provider, parameters = configure_provider(args)
            if args.command == 'solver-add':
                identifier(args.name)
                if not 1 <= args.repetitions <= 10000: raise ValidationError('Invalid repetitions')
                path = 'solver-' + args.name + '.json'
                if path in cfg['solvers']: raise ValidationError('Configuration name already exists')
                value = {'provider': provider, 'model': args.model, 'configurations': [{'name': args.name, 'parameters': parameters}], 'repetitions': args.repetitions, 'max_calls': args.max_calls}
                if args.instruction: value['configurations'][0]['instruction'] = args.instruction
                cfg['solvers'].append(path)
            else:
                path = 'judge.json'
                value = {'type': 'generation-provider', 'provider': provider, 'model': args.model, 'parameters': parameters, 'max_calls': args.max_calls, 'structured_retries': args.structured_retries}
                cfg['judge'] = path
            write(root / path, value)
        elif args.command == 'policy-set':
            value = {'min_mean_judge_score': args.min_quality, 'min_judge_coverage': args.min_coverage, 'max_judge_critical_error_rate': args.max_critical_rate}
            for key, val in [('min_judge_pass_rate', args.min_pass_rate), ('max_mean_solver_latency_seconds', args.max_latency), ('max_mean_cost_usd', args.max_cost)]:
                if val is not None: value[key] = val
            cfg['policy'] = validate_business_policy(value)
            write(root / 'policy.json', cfg['policy'])
        elif args.command == 'run':
            result = run_experiment(root, Path(args.output), args.allow_remote); print(terminal_summary(result)); print('Отчёт: '+str(Path(args.output)/'report.html')); return 0
        elif args.command == 'calibrate':
            result = calibrate(root, args.labels, Path(args.output), args.allow_remote); print(json.dumps(result, ensure_ascii=False)); return 0
        write(root / 'experiment.json', cfg)
        print('Готово'); return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        # Input validation messages do not contain payloads or credentials.
        message = str(exc) if isinstance(exc, ValidationError) else type(exc).__name__
        print('Ошибка: ' + message, file=sys.stderr); return 2
