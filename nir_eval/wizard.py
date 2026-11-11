"""Interactive API-only setup. Saving a plan never starts inference implicitly."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shlex
from urllib.parse import urlsplit

from .business.schema import ValidationError, import_inputs, load_jsonl, validate_case
from .credentials import interactive_credentials, request_api_key


@dataclass
class ModelSettings:
    name: str
    model: str
    endpoint: str
    key_env: str
    temperature: float
    max_tokens: int
    input_price: float | None = None
    output_price: float | None = None
    instruction: str = ''


def is_remote(endpoint):
    return urlsplit(endpoint).hostname not in {'localhost', '127.0.0.1', '::1'}


def valid_endpoint(value):
    u = urlsplit(value)
    if u.scheme not in {'http', 'https'} or not u.hostname:
        raise ValueError('Укажите HTTP(S)-адрес API, например http://127.0.0.1:8000/v1.')
    if u.username or u.password or u.query or u.fragment:
        raise ValueError('В адресе API не должно быть ключей, пароля, query или fragment.')
    if is_remote(value) and u.scheme != 'https':
        raise ValueError('Для внешнего API требуется HTTPS.')
    return value.rstrip('/')


class Questions:
    def __init__(self, input_fn, print_fn):
        self.input = input_fn
        self.print = print_fn

    def ask(self, label, default=None, validate=None, optional=False):
        while True:
            suffix = f' [{default}]' if default is not None else ''
            answer = self.input(label + suffix + ': ').strip()
            if not answer:
                answer = str(default) if default is not None else ''
            if not answer and optional:
                return None
            try:
                if not answer:
                    raise ValueError('Введите значение.')
                return validate(answer) if validate else answer
            except (ValueError, OSError) as exc:
                self.print(str(exc) if isinstance(exc, ValueError) else 'Файл недоступен. Проверьте путь.')

    def number(self, label, default=None, low=0, high=None, integer=False, optional=False):
        def check(text):
            value = int(text) if integer else float(text.replace(',', '.'))
            if not math.isfinite(value) or value < low or (high is not None and value > high):
                raise ValueError(f'Введите число от {low}' + (f' до {high}.' if high is not None else '.'))
            return value
        return self.ask(label, default, check, optional)


def _new_path(value):
    path = Path(value).expanduser().resolve()
    if path.exists():
        raise ValueError('Эта папка уже существует. Выберите новую, чтобы сохранить прежние результаты.')
    return path


def _output_path(value, root):
    path = _new_path(value)
    if path == root or path.is_relative_to(root) or root.is_relative_to(path):
        raise ValueError('Папки эксперимента и результатов должны быть отдельными и не вложенными друг в друга.')
    return path


def _file(value):
    p = Path(value).expanduser().resolve()
    if not p.is_file():
        raise ValueError('Файл не найден. Укажите путь к существующему файлу.')
    return p


def _env_name(value):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,119}', value):
        raise ValueError('Введите имя переменной окружения (например NIR_API_KEY), а не сам ключ.')
    if value.lower().startswith(('sk-', 'ghp_', 'github_pat_')):
        raise ValueError('Сам ключ вводить нельзя: требуется только имя переменной окружения.')
    return value


def _model(value):
    if len(value) > 200 or any(c.isspace() for c in value):
        raise ValueError('Нужен ID модели из каталога API, без пробелов.')
    if value.startswith(('sk-', 'ghp_', 'github_pat_')):
        raise ValueError('Это похоже на секрет. Укажите ID модели, не ключ.')
    return value


def _model_questions(q, name, endpoint, key_env, judge=False, forbidden=()):
    q.print('\n' + ('Отдельная модель-судья' if judge else 'Конфигурация ' + name))
    def distinct(value):
        value = _model(value)
        if value in forbidden:
            raise ValueError('Судья должен отличаться от каждой решающей модели.')
        return value
    model = q.ask('ID модели', validate=distinct)
    endpoint = q.ask('Адрес API', endpoint, valid_endpoint)
    key_env = q.ask('Имя переменной с ключом (сам ключ не вводите)', key_env, _env_name)
    temperature = q.number('Температура', 0, low=0, high=2)
    max_tokens = q.number('Максимум выходных токенов', 8192 if judge else 512, low=64, high=32768, integer=True)
    instruction = '' if judge else q.ask('Дополнительная инструкция (Enter — без неё)', optional=True) or ''
    while True:
        input_price = q.number('Цена входа, USD за 1 млн токенов (Enter — неизвестна)', optional=True)
        output_price = q.number('Цена выхода, USD за 1 млн токенов (Enter — неизвестна)', optional=True)
        if (input_price is None) == (output_price is None):
            break
        q.print('Укажите оба тарифа или оставьте оба неизвестными. Повторите ввод тарифов.')
    return ModelSettings(name, model, endpoint, key_env, temperature, max_tokens, input_price, output_price, instruction)


def model_command(command, directory, settings, max_calls, retries=0):
    args = [command, str(directory), '--provider', 'api', '--base-url', settings.endpoint,
            '--api-key-env', settings.key_env, '--model', settings.model,
            '--temperature', str(settings.temperature), '--max-tokens', str(settings.max_tokens),
            '--max-calls', str(max_calls)]
    if command == 'solver-add':
        args += ['--name', settings.name]
        if settings.instruction:
            args += ['--instruction', settings.instruction]
    else:
        args += ['--structured-retries', str(retries)]
    for flag, value in [('--input-price-per-million', settings.input_price),
                        ('--output-price-per-million', settings.output_price)]:
        if value is not None:
            args += [flag, str(value)]
    return args


def estimated_cost(settings, judge, cases, retries, solver_input=None, judge_input=None):
    """Scenario estimate, never a promised bill or an inferred zero for missing prices."""
    all_models = [*settings, judge]
    if any(m.input_price is None or m.output_price is None for m in all_models):
        return None
    # Exact configured zero token tariffs imply zero token-cost estimate without counts.
    if all(m.input_price == 0 and m.output_price == 0 for m in all_models):
        return 0.0
    if solver_input is None or judge_input is None:
        return None
    solver_cost = sum((solver_input * m.input_price + m.max_tokens * m.output_price) / 1_000_000 for m in settings) * cases
    judge_cost = (judge_input * judge.input_price + judge.max_tokens * judge.output_price) / 1_000_000 * cases * len(settings) * (1 + retries)
    return solver_cost + judge_cost


def run_wizard(directory=None, output=None, *, input_fn=None, print_fn=None, command_fn=None):
    input_fn = input_fn or input
    print_fn = print_fn or print
    if command_fn is None:
        from .experiment import main as command_fn
    q = Questions(input_fn, print_fn)
    try:
        q.print('Tochka · Мастер сравнения ИИ на задачах бизнеса\nНастройки сохраняются локально. До отдельного подтверждения запросов к моделям не будет.')
        root = _new_path(str(directory)) if directory is not None else q.ask('Новая папка эксперимента', 'experiments/my-test', _new_path)
        requested_output = _output_path(str(output), root) if output is not None else None
        mode = q.ask('Набор задач: 1 — демонстрация, 2 — свои JSONL', '1', lambda x: x if x in {'1', '2'} else (_ for _ in ()).throw(ValueError('Выберите 1 или 2.')))
        if mode == '1':
            source = Path(__file__).parent / 'demo'
            case_path, rubric_path = source / 'cases.jsonl', source / 'rubrics.jsonl'
        else:
            while True:
                case_path = q.ask('Файл задач JSONL', validate=_file)
                rubric_path = q.ask('Файл критериев JSONL', validate=_file)
                try:
                    import_inputs(case_path, rubric_path)
                    break
                except ValidationError as exc:
                    q.print('Набор не прошёл проверку: ' + str(exc))
        cases, _, _ = import_inputs(case_path, rubric_path)
        docs = []
        q.print('Общие документы будут добавлены к контексту каждой задачи. Поддерживаются UTF-8 текст и Markdown.')
        while True:
            path = q.ask('Путь к общему документу (Enter — закончить)', validate=_file, optional=True)
            if path is None:
                break
            try:
                text = path.read_text(encoding='utf-8')
                updated = [{**c, 'context': c['context'] + '\n\n' + text} for c in cases]
                for c in updated:
                    validate_case(c)
            except (UnicodeError, OSError, ValidationError):
                q.print('Документ не добавлен: нужен UTF-8 текст допустимого размера.')
                continue
            cases = updated
            docs.append(path)
        q.print(f'Загружено задач: {len(cases)}. Эталоны и критерии не передаются решающим моделям.')
        from .credentials import GO_BASE_URL, GO_KEY_ENV, api_key, saved_go_key
        has_go = bool(api_key({'base_url': GO_BASE_URL, 'api_key_env': GO_KEY_ENV}))
        if has_go:
            q.print('Найден ключ OpenCode Go в доступном локальном источнике; повторный ввод не нужен.')
        endpoint = q.ask('Общий адрес OpenAI-совместимого API', GO_BASE_URL if has_go else 'http://127.0.0.1:8000/v1', valid_endpoint)
        key_env = q.ask('Имя переменной с API-ключом (сам ключ не вводите)', GO_KEY_ENV if has_go and endpoint.rstrip('/') == GO_BASE_URL else 'NIR_API_KEY', _env_name)
        count = q.number('Сколько конфигураций сравнить', 2, low=2, high=20, integer=True)
        solvers = [_model_questions(q, f'model-{i + 1}', endpoint, key_env) for i in range(count)]
        judge = _model_questions(q, 'judge', endpoint, key_env, judge=True, forbidden={m.model for m in solvers})
        retries = q.number('Дополнительные вызовы судьи при невалидном JSON/цитатах', 0, low=0, high=2, integer=True)
        q.print('\nТребования к результату. Неполные оценки по умолчанию исключаются из выбора.')
        quality = q.number('Минимальное качество, %', 80, low=0, high=100)
        pass_rate = q.number('Минимальная доля прошедших задач, %', 100, low=0, high=100)
        latency = q.number('Максимальное среднее время solver, секунд (Enter — без лимита)', low=0, optional=True)
        cost = q.number('Максимальная средняя стоимость solver + judge, USD (Enter — без лимита)', low=0, optional=True)
        concurrency = q.number('Максимум одновременных запросов к моделям', 4, low=1, high=32, integer=True)
        target = requested_output if requested_output is not None else q.ask('Папка результатов', str(Path('runs') / root.name), lambda value: _output_path(value, root))
        solver_input = judge_input = None
        if all(m.input_price is not None and m.output_price is not None for m in [*solvers, judge]) and any(m.input_price != 0 or m.output_price != 0 for m in [*solvers, judge]):
            q.print('Для приблизительной стоимости нужны ваши оценки входных токенов. Это сценарий расходов, не денежный лимит.')
            solver_input = q.number('Оценка входных токенов на один solver-запрос (Enter — неизвестно)', integer=True, optional=True)
            judge_input = q.number('Оценка входных токенов на один judge-запрос (Enter — неизвестно)', integer=True, optional=True)
        estimate = estimated_cost(solvers, judge, len(cases), retries, solver_input, judge_input)
        policy = ['policy-set', str(root), '--min-quality', str(quality / 100), '--min-pass-rate', str(pass_rate / 100), '--min-coverage', '1', '--max-critical-rate', '0']
        if latency is not None:
            policy += ['--max-latency', str(latency)]
        if cost is not None:
            policy += ['--max-cost', str(cost)]
        commands = [['init', str(root)] + (['--demo'] if mode == '1' else [])]
        if mode == '2':
            commands += [['import', str(root), '--cases', str(case_path), '--rubrics', str(rubric_path)]]
        commands += [model_command('solver-add', root, m, len(cases)) for m in solvers]
        commands += [model_command('judge-set', root, judge, len(cases) * (1 + retries), retries), policy]
        for command in commands:
            if command_fn(command) != 0:
                q.print('Настройка остановлена. Уже сохранённые файлы оставлены для проверки; запросов к моделям не было.')
                return 2
        if docs:
            from .experiment import jsonl
            jsonl(root / 'cases.jsonl', cases)
        q.print('\nПлан запуска')
        q.print(f'Задач: {len(cases)} · Конфигураций: {count} · Solver-запросов: {len(cases) * count}')
        q.print(f'Judge-запросов: {len(cases) * count}; максимум с исправлениями: {len(cases) * count * (1 + retries)}')
        q.print(f'Одновременных запросов: до {concurrency}. Параллельность не увеличивает число задач.')
        for m in solvers:
            q.print(f'  {m.name}: {m.model} — {m.endpoint}')
        q.print(f'Судья: {judge.model} — {judge.endpoint}')
        q.print(f'Минимальное качество: {quality:g}% · Полнота проверки: 100% · Критические ошибки: 0')
        if estimate is None:
            q.print('Оценка стоимости: неизвестна (не хватает тарифов или оценки входных токенов).')
        else:
            q.print(f'Оценка стоимости по указанным тарифам и токенам: ${estimate:.6f}. Это не гарантия и не денежный бюджет.')
        q.print('Ограничение стоимости при отборе применяется после выполнения запросов, а не останавливает расход денег.')
        remote = any(is_remote(m.endpoint) for m in [*solvers, judge])
        if remote:
            q.print('При запуске задачи и контекст уйдут во внешние API; судье также передаются эталоны и критерии.')
        absent = sorted({m.key_env for m in [*solvers, judge] if is_remote(m.endpoint) and not api_key({'base_url': m.endpoint, 'api_key_env': m.key_env})})
        if absent:
            q.print('Для ' + ', '.join(absent) + ' ключ будет запрошен скрытым вводом после подтверждения и сохранён локально для следующих запусков. Управление: tochka keys.')
        q.print('Настройки сохранены: ' + str(root))
        q.print('Результаты будут сохранены: ' + str(target))
        if q.ask('Запустить сейчас? Только «да» запускает API-вызовы; Enter — сохранить без запуска', optional=True) not in {'да', 'ДА', 'Да'}:
            q.print('Запуск не выполнялся. Для запуска позже:')
            q.print(f'tochka run {shlex.quote(str(root))} --output {shlex.quote(str(target))} --concurrency {concurrency}' + (' --allow-remote' if remote else ''))
            return 0
        for m in [*solvers, judge]:
            if is_remote(m.endpoint) and not request_api_key({'base_url': m.endpoint, 'api_key_env': m.key_env}):
                q.print('Запуск не выполнен: для скрытого ввода откройте CLI в терминале или задайте переменную ' + m.key_env + '. Настройки сохранены.')
                return 2
        args = ['run', str(root), '--output', str(target), '--concurrency', str(concurrency)] + (['--allow-remote'] if remote else [])
        return command_fn(args)
    except (EOFError, KeyboardInterrupt):
        q.print('\nМастер остановлен. Не подтверждённый запуск не выполнялся.')
        return 130
    except (ValueError, OSError) as exc:
        q.print('Мастер остановлен: ' + (str(exc) if isinstance(exc, ValueError) else type(exc).__name__))
        return 2


@interactive_credentials
def main(argv=None, *, input_fn=None, print_fn=None, command_fn=None):
    parser = argparse.ArgumentParser(prog='tochka wizard', description='Пошаговая настройка API-эксперимента без редактирования кода')
    parser.add_argument('--directory', help='Новая папка настроек')
    parser.add_argument('--output', help='Новая папка результатов')
    args = parser.parse_args(argv)
    return run_wizard(args.directory, args.output, input_fn=input_fn, print_fn=print_fn, command_fn=command_fn)


if __name__ == '__main__':
    raise SystemExit(main())
