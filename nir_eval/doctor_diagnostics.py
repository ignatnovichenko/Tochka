"""Bounded offline diagnostics. Output contains no user content or credentials."""
import ast
import os
from pathlib import Path
import stat
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent
# Fixed filenames only: never walk an experiment or execute its Python files.
SOURCE_FILES = (
    '__init__.py', '__main__.py', 'app.py', 'experiment.py', 'api_support.py',
    'providers.py', 'credentials.py', 'key_store.py', 'keys_cli.py', 'wizard.py',
    'resume.py', 'business_report.py', 'business_selection.py',
    'business/schema.py', 'business/running.py', 'business/providers.py',
    'business/evaluation.py', 'business/reporting.py',
)
MAX_SOURCE_BYTES = 512_000
MAX_JSON_BYTES = 1_000_000
MAX_INPUT_BYTES = 20_000_000
MAX_SOLVERS = 100


def _read(path, limit):
    """Bounded regular-file read; refuse final symlinks and special files."""
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError('unsafe or oversized file')
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError('oversized file')
        return data, info


def _json(path, limit=MAX_JSON_BYTES):
    from .business.schema import strict_json
    data, _ = _read(path, limit)
    value = strict_json(data)
    if not isinstance(value, dict):
        raise ValueError('expected object')
    return value


def _finding(result, code, summary, repair=None):
    item = {'code': code, 'summary': summary}
    if repair:
        item['repair'] = repair
    result['findings'].append(item)


def _keys(result):
    path = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config') / 'tochka' / 'credentials.json'
    try:
        data, info = _read(path, MAX_JSON_BYTES)
        if os.name == 'posix' and (info.st_mode & 0o077 or info.st_uid != os.getuid()):
            result['key_store'] = {'status': 'unsafe_permissions'}
            _finding(result, 'KEY_STORE_PERMISSIONS', 'Хранилище ключей имеет небезопасные права или владельца.', 'REVIEW_KEY_STORE_PERMISSIONS')
            return
        from .business.schema import strict_json
        from .key_store import _validate_entry, identity
        import re
        store = strict_json(data)
        if not isinstance(store, dict) or store.get('version') != 1 or not isinstance(store.get('keys'), dict):
            raise ValueError()
        identities = set()
        for name, entry in store['keys'].items():
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name):
                raise ValueError()
            _validate_entry(entry)
            ident = identity(entry)
            if ident in identities:
                raise ValueError()
            identities.add(ident)
        result['key_store'] = {'status': 'ok', 'stored_entries': len(identities)}
    except FileNotFoundError:
        result['key_store'] = {'status': 'absent'}
        # An environment variable or local server can supply authentication; absence isn't failure.
    except Exception:
        result['key_store'] = {'status': 'unreadable_or_invalid'}
        _finding(result, 'KEY_STORE_INVALID', 'Хранилище ключей недоступно, повреждено или имеет небезопасный тип файла. Оно не изменено.', 'CONFIGURE_KEYS')


def _safe_filename(value):
    return isinstance(value, str) and value not in {'', '.', '..'} and Path(value).name == value and '/' not in value and '\\' not in value


def _experiment(result, root):
    section = result['experiment'] = {'status': 'unrecognized'}
    if not root.is_dir() or root.is_symlink():
        _finding(result, 'EXPERIMENT_MISSING', 'Выбранная папка эксперимента отсутствует или недоступна.', 'RECREATE_EXPERIMENT')
        return
    manifest_path = root / 'experiment-manifest.json'
    business_manifest_path = root / 'manifest.json'
    if manifest_path.exists() or business_manifest_path.exists():
        try:
            manifest = _json(manifest_path if manifest_path.exists() else business_manifest_path)
            state = manifest.get('state')
            if state not in {'running', 'interrupted', 'completed'}:
                raise ValueError()
            section.update(status='saved_run', state=state, integrity_checked=False)
            for name in ('planned_attempts', 'solver_completed_attempts', 'evaluated_attempts', 'judge_evaluated'):
                number = manifest.get(name)
                if type(number) is int and 0 <= number <= 10**9:
                    section[name] = number
            if manifest_path.exists():
                for name in ('planned_runs', 'runs'):
                    values = manifest.get(name)
                    if isinstance(values, list) and len(values) <= MAX_SOLVERS and all(_safe_filename(v) for v in values):
                        section[name + '_count'] = len(values)
                required = ['comparison.json', 'report.html'] if state == 'completed' else []
            else:
                required = ['responses.jsonl', 'solver_records.jsonl', 'records.jsonl'] if state == 'completed' else []
            missing = [name for name in required if not (root / name).is_file()]
            if missing:
                section['missing_artifacts'] = missing
                _finding(result, 'RUN_ARTIFACTS_MISSING', 'В завершённом прогоне отсутствуют ожидаемые файлы результатов.', 'REVIEW_SAVED_RUN')
            if state == 'interrupted':
                _finding(result, 'RUN_INTERRUPTED', 'Прогон был прерван. Продолжение должно сохранять результат в новую папку.', 'RESUME_TO_NEW_DIRECTORY')
            elif state == 'running':
                _finding(result, 'RUN_NOT_FINISHED', 'Прогон не помечен завершённым; он может ещё выполняться. Не удаляйте блокировки.', 'REVIEW_SAVED_RUN')
        except Exception:
            section['status'] = 'invalid_manifest'
            _finding(result, 'RUN_MANIFEST_INVALID', 'Манифест прогона недоступен или имеет неверный формат.', 'REVIEW_SAVED_RUN')
        return
    try:
        cfg = _json(root / 'experiment.json')
        solvers = cfg.get('solvers')
        if not isinstance(solvers, list) or not 1 <= len(solvers) <= MAX_SOLVERS:
            raise ValueError()
        names = [cfg.get('judge'), *solvers]
        if not all(_safe_filename(n) for n in names):
            raise ValueError()
        for name in ['cases.jsonl', 'rubrics.jsonl', *names]:
            _read(root / name, MAX_INPUT_BYTES if name.endswith('.jsonl') else MAX_JSON_BYTES)
        from .experiment import experiment_plan
        plan = experiment_plan(root, allow_remote=True, check_credentials=False)
        section.update(status='valid_plan', credentials_checked=False)
        # Never return configuration filenames, model IDs, endpoint URLs or plan text.
        for name in ('cases', 'solver_calls', 'judge_calls', 'maximum_judge_calls'):
            number = plan.get(name)
            if type(number) is int and 0 <= number <= 10**9:
                section[name] = number
        section['solver_configuration_files'] = len(solvers)
    except FileNotFoundError:
        section['status'] = 'missing_inputs'
        _finding(result, 'EXPERIMENT_INPUT_MISSING', 'Отсутствует обязательный файл настроек, задач или критериев.', 'CONFIGURE_EXPERIMENT')
    except Exception:
        section['status'] = 'invalid_plan'
        _finding(result, 'EXPERIMENT_PLAN_INVALID', 'План не прошёл локальную проверку задач, критериев, моделей, судьи или ограничений.', 'CONFIGURE_EXPERIMENT')


def collect_diagnostics(experiment: Path | None = None) -> dict:
    """Read-only local checks; safe bounded metadata suitable for a model prompt.

    This does not certify API connectivity, model quality or historical run integrity.
    Raw exception messages, source lines, arbitrary paths and user content never leave
    this function. Repair values are suggestions only, never executed actions.
    """
    result = {'schema_version': 'tochka-doctor-diagnostics-v1',
              'python': {'version': '.'.join(map(str, sys.version_info[:3])), 'supported': sys.version_info >= (3, 10)},
              'source_syntax': [], 'key_store': {}, 'experiment': None, 'findings': [],
              'network_calls': 0, 'model_calls_performed': False, 'files_modified': False}
    if not result['python']['supported']:
        _finding(result, 'PYTHON_UNSUPPORTED', 'Требуется Python 3.10 или новее.', 'REINSTALL_PACKAGE')
    for name in SOURCE_FILES:
        check = {'module': name, 'status': 'ok'}
        try:
            data, _ = _read(PACKAGE_ROOT / name, MAX_SOURCE_BYTES)
            ast.parse(data, filename=name)
        except FileNotFoundError:
            check['status'] = 'missing'
        except SyntaxError:
            check['status'] = 'syntax_error'
        except Exception:
            check['status'] = 'unreadable'
        result['source_syntax'].append(check)
    if any(check['status'] != 'ok' for check in result['source_syntax']):
        _finding(result, 'PACKAGE_SOURCE_INVALID', 'Часть файлов пакета отсутствует, недоступна или содержит синтаксическую ошибку.', 'REINSTALL_PACKAGE')
    _keys(result)
    if experiment is not None:
        try:
            _experiment(result, Path(experiment))
        except Exception:
            result['experiment'] = {'status': 'unreadable'}
            _finding(result, 'EXPERIMENT_UNREADABLE', 'Выбранный эксперимент недоступен для локальной проверки.', 'CONFIGURE_EXPERIMENT')
    return result
