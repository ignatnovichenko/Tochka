"""Persistent, shell-free interactive console for Tochka."""
import argparse
from pathlib import Path
import shlex

DESCRIPTIONS = {
    'setup': 'Установить Promptfoo или проверить готовность (--check)',
    'wizard': 'Пошагово создать и запустить эксперимент',
    'keys': 'Настроить API-ключи безопасным вводом',
    'init': 'Создать эксперимент (можно --demo)',
    'import': 'Загрузить задачи и критерии',
    'task-add': 'Добавить задачу и документы',
    'rubric-add': 'Добавить критерии оценки',
    'solver-add': 'Настроить проверяемую API-модель',
    'judge-set': 'Настроить отдельную модель-судью',
    'policy-set': 'Задать ограничения качества, времени и стоимости',
    'models': 'Посмотреть модели эксперимента',
    'plan': 'Показать план вызовов без запросов к моделям',
    'show': 'Посмотреть настройки эксперимента',
    'run': 'Запустить проверку (--concurrency N задаёт параллельность)',
    'calibrate': 'Проверить судью на размеченных примерах (--concurrency N)',
    'results': 'Посмотреть результаты прогона',
    'select': 'Выбрать конфигурацию по файлу ограничений',
    'reselect': 'Изменить ограничения и отчёт без запросов к моделям',
    'resume': 'Продолжить незавершённый прогон в новой папке (--concurrency N)',
    'use': 'Выбрать активную папку эксперимента',
    'model': 'Посмотреть или изменить модель помощника /doctor',
    'doctor': 'Диагностика и помощь с отдельным подтверждением отправки',
    'help': 'Все команды или /help КОМАНДА',
    'exit': 'Выйти',
}


def command_parsers():
    from .experiment import parser
    return next(a.choices for a in parser()._actions if isinstance(a, argparse._SubParsersAction))


def dispatch(args, input_fn, print_fn):
    if args[0] == 'setup':
        from .setup_cli import main
        return main(args[1:], input_fn=input_fn, print_fn=print_fn)
    if args[0] == 'wizard':
        from .wizard import main
        return main(args[1:], input_fn=input_fn, print_fn=print_fn)
    from .experiment import main
    return main(args)


def run_shell(*, input_fn=None, print_fn=None, command_fn=None):
    input_fn, print_fn = input_fn or input, print_fn or print
    command_fn = command_fn or (lambda args: dispatch(args, input_fn, print_fn))
    parsers = command_parsers()
    commands = list(parsers) + ['use', 'model', 'doctor', 'help', 'exit']
    active = None
    last_error = None
    print_fn('Tochka — интерактивная консоль. / — меню, /help — все команды, /exit — выход.')
    print_fn('Аргументы с пробелами заключайте в кавычки. Команды выполняются без системной оболочки.')
    while True:
        try:
            line = input_fn('tochka> ').strip()
            if not line:
                continue
            if line == '/':
                for i, name in enumerate(commands, 1):
                    print_fn(f'{i}. /{name} — {DESCRIPTIONS.get(name, name)}')
                choice = input_fn('Номер команды (Enter — назад): ').strip()
                if not choice:
                    continue
                if not choice.isdigit() or not 1 <= int(choice) <= len(commands):
                    print_fn('Выберите номер из меню.')
                    continue
                name = commands[int(choice) - 1]
                if name in parsers:
                    print_fn(parsers[name].format_usage().strip())
                    if active:
                        print_fn(f'Активный эксперимент: {active}. Для команды без аргументов папка подставляется автоматически.')
                    tail = input_fn('Аргументы команды (или Enter): ')
                    line = name + ' ' + tail
                else:
                    line = name
            parts = shlex.split(line[1:] if line.startswith('/') else line)
            if not parts:
                continue
            name, args = parts[0], parts[1:]
            if name in {'exit', 'quit'}:
                return 0
            if name == 'help':
                if not args:
                    for item in commands:
                        print_fn(f'/{item} — {DESCRIPTIONS.get(item, item)}')
                elif len(args) == 1 and args[0].lstrip('/') in commands:
                    target = args[0].lstrip('/')
                    if target in parsers:
                        command_fn([target, '--help'])
                    elif target in {'model', 'doctor'}:
                        if target == 'model':
                            from .doctor_settings import main
                        else:
                            from .doctor import main
                        main(['--help'])
                    else:
                        print_fn(f'/{target} — {DESCRIPTIONS[target]}')
                else:
                    print_fn('Неизвестная команда. /help показывает список.')
                continue
            if name == 'use':
                raw = ' '.join(args) if args else input_fn('Папка эксперимента (Enter — показать текущую): ').strip()
                if raw:
                    candidate = Path(raw).expanduser().resolve()
                    if not (candidate / 'experiment.json').is_file():
                        print_fn('В этой папке нет experiment.json. Создайте эксперимент через /wizard или /init.')
                        continue
                    active = candidate
                print_fn(f'Активный эксперимент: {active or "не выбран"}')
                continue
            if name == 'model':
                from .doctor_settings import main
                status = main(args, input_fn=input_fn, print_fn=print_fn)
            elif name == 'doctor':
                if args:
                    print_fn('/doctor работает с текущей моделью и активным экспериментом; используйте /model и /use.')
                    continue
                from .doctor import run_doctor
                from .doctor_settings import load_settings
                try:
                    profile = load_settings()
                except ValueError:
                    profile = None
                status = run_doctor(profile=profile, experiment=active,
                                    input_fn=input_fn, print_fn=print_fn, last_error=last_error)
            elif name in parsers:
                if not args and active and name != 'init':
                    if any(a.dest == 'directory' and not a.option_strings for a in parsers[name]._actions):
                        args = [str(active)]
                    elif name == 'models':
                        args = ['--directory', str(active)]
                status = command_fn([name, *args])
            else:
                print_fn('Неизвестная команда. Введите / или /help.')
                continue
            if isinstance(status, int) and status:
                last_error = f'Команда {name} завершилась с кодом {status}.'
                print_fn(last_error + ' /doctor поможет с диагностикой.')
        except EOFError:
            print_fn('\nДо встречи!')
            return 0
        except KeyboardInterrupt:
            print_fn('\nКоманда прервана. /exit — выход из консоли.')
            last_error = 'Пользователь прервал команду.'
        except SystemExit as exc:
            if exc.code:
                last_error = 'Ошибка аргументов команды.'
                print_fn(last_error + ' Используйте /help КОМАНДА.')
        except Exception as exc:
            # Provider exceptions can contain credentials or business data.
            last_error = f'Ошибка выполнения ({type(exc).__name__}).'
            print_fn(last_error + ' Консоль работает; /doctor — диагностика.')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='tochka shell', description='Интерактивная консоль Tochka')
    parser.parse_args(argv)
    return run_shell()
