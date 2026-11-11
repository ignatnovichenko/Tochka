"""Guided product actions over the existing, scriptable CLI commands."""
from datetime import datetime
from pathlib import Path


class ConsoleActions:
    def __init__(self, input_fn, print_fn, command_fn):
        self.ask, self.print, self.command = input_fn, print_fn, command_fn
        self.active = None
        self.last_run = None

    def choice(self, title, options):
        self.print(title)
        for i, label in enumerate(options, 1):
            self.print(f'{i}. {label}')
        while True:
            answer = self.ask('Номер (Enter — назад): ').strip()
            if not answer:
                return None
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return int(answer)
            self.print('Выберите номер из меню.')

    def value(self, label, default=None, required=False):
        while True:
            answer = self.ask(label + (f' [{default}]' if default is not None else '') + ': ').strip()
            if answer:
                return answer
            if default is not None:
                return str(default)
            if not required:
                return None
            self.print('Введите значение.')

    def folder(self, label):
        raw = self.value(label)
        return Path(raw).expanduser().resolve() if raw else None

    def open_experiment(self, path):
        if not (path / 'experiment.json').is_file():
            self.print('В этой папке нет experiment.json. Создайте эксперимент через /experiment.')
            return 2
        self.active = path
        self.print(f'Активный эксперимент: {path}')
        return 0

    def require_active(self):
        if self.active:
            return True
        path = self.folder('Папка существующего эксперимента (Enter — назад)')
        return path is not None and self.open_experiment(path) == 0

    def experiment(self, args):
        if args:
            return self.open_experiment(Path(args[0]).expanduser().resolve())
        selected = self.choice('Эксперимент', ['Создать через пошаговый мастер', 'Создать основу с демонстрационными задачами', 'Открыть существующий'])
        if selected is None:
            return 0
        root = self.folder('Папка эксперимента (Enter — назад)')
        if root is None:
            return 0
        if selected == 3:
            return self.open_experiment(root)
        if selected == 1:
            from .wizard import main
            status = main(['--directory', str(root)], input_fn=self.ask, print_fn=self.print, command_fn=self.command)
        else:
            status = self.command(['init', str(root), '--demo'])
        if status == 0:
            return self.open_experiment(root)
        return status

    def form(self, name, fields, prefix=None):
        args = [name, *(prefix if prefix is not None else [str(self.active)])]
        for flag, label, default, required in fields:
            value = self.value(label, default, required)
            if value is not None:
                args += [flag, value]
        return self.command(args)

    def settings(self):
        selected = self.choice('Настройки', ['Задачи, документы и критерии', 'Проверяемые модели', 'Модель-судья', 'Ограничения качества, времени и стоимости', 'API-ключи', 'Посмотреть все настройки', 'Модель помощника диагностики'])
        if selected is None:
            return 0
        if selected == 5:
            return self.command(['keys'])
        if selected == 7:
            from .doctor_settings import main
            return main([], input_fn=self.ask, print_fn=self.print)
        if not self.require_active():
            return 0
        if selected == 6:
            return self.command(['show', str(self.active)])
        if selected == 1:
            mode = self.choice('Задачи и критерии', ['Загрузить набор из JSONL', 'Добавить задачу и документ', 'Добавить критерий оценки'])
            if mode == 1:
                return self.form('import', [('--cases', 'Файл задач JSONL', None, True), ('--rubrics', 'Файл критериев JSONL', None, True)])
            if mode == 2:
                return self.form('task-add', [('--id', 'Уникальный ID задачи', None, True), ('--task', 'Задание', None, True), ('--rubric', 'ID набора критериев', None, True), ('--context', 'Контекст (Enter — пропустить)', None, False), ('--document', 'Файл документа UTF-8 (Enter — пропустить)', None, False), ('--reference', 'Эталонный ответ (Enter — пропустить)', None, False)])
            if mode == 3:
                return self.form('rubric-add', [('--id', 'ID набора критериев', None, True), ('--criterion', 'Критерий: id=описание', None, True), ('--critical', 'Критическая ошибка: id=описание (Enter — пропустить)', None, False), ('--pass-threshold', 'Порог прохождения, от 0 до 1', '0.75', True)])
            return 0
        if selected in {2, 3}:
            if selected == 2:
                mode = self.choice('Проверяемые модели', ['Показать модели', 'Добавить конфигурацию модели'])
                if mode is None:
                    return 0
                if mode == 1:
                    return self.command(['models', '--directory', str(self.active)])
            from .credentials import GO_BASE_URL, GO_KEY_ENV, api_key
            has_go = bool(api_key({'base_url': GO_BASE_URL, 'api_key_env': GO_KEY_ENV}))
            fields = [] if selected == 3 else [('--name', 'Имя конфигурации', None, True)]
            fields += [('--model', 'ID модели', None, True), ('--base-url', 'Адрес API', GO_BASE_URL if has_go else 'http://127.0.0.1:8000/v1', True), ('--api-key-env', 'Имя переменной ключа (сам ключ вводится в разделе API-ключи)', GO_KEY_ENV if has_go else 'NIR_API_KEY', True), ('--temperature', 'Temperature', '0', True), ('--max-tokens', 'Максимум выходных токенов', '4096', True), ('--max-calls', 'Максимум вызовов', '100', True)]
            if selected == 3:
                fields += [('--structured-retries', 'Исправления невалидной оценки, от 0 до 2', '1', True)]
            return self.form('judge-set' if selected == 3 else 'solver-add', fields)
        from .experiment import config
        policy = config(self.active).get('policy', {})
        return self.form('policy-set', self.policy_fields(policy))

    @staticmethod
    def policy_fields(policy):
        return [(flag, label, policy.get(key, default), required) for flag, label, key, default, required in [
            ('--min-quality', 'Минимальное качество, от 0 до 1', 'min_mean_judge_score', .75, True),
            ('--min-coverage', 'Минимальная полнота оценки, от 0 до 1', 'min_judge_coverage', 1, True),
            ('--min-pass-rate', 'Минимальная доля прошедших задач (Enter — оставить)', 'min_judge_pass_rate', None, False),
            ('--max-latency', 'Максимальное среднее время ответа, секунд (Enter — оставить)', 'max_mean_solver_latency_seconds', None, False),
            ('--max-cost', 'Максимальная средняя стоимость задачи, USD (Enter — оставить)', 'max_mean_cost_usd', None, False),
            ('--max-critical-rate', 'Максимальная доля критических ошибок', 'max_judge_critical_error_rate', 0, True)]]

    def output(self, label, stem):
        default = Path.cwd() / 'runs' / f'{stem}-{datetime.now():%Y%m%d-%H%M%S-%f}'
        return str(Path(self.value(label, default)).expanduser().resolve())

    def run(self):
        mode = self.choice('Запуск', ['Новый прогон', 'Продолжить прежний прогон', 'Только показать план без API-вызовов'])
        if mode is None:
            return 0
        if mode == 2:
            source = self.folder('Папка прежнего прогона (Enter — назад)')
            if source is None:
                return 0
            output = self.output('Новая папка результатов', 'resume')
            args = ['resume', str(source), '--output', output]
            status = self.command([*args, '--plan-only'])
        else:
            if not self.require_active():
                return 0
            status = self.command(['plan', str(self.active)])
            if status or mode == 3:
                return status
            output = self.output('Новая папка результатов', self.active.name)
            args = ['run', str(self.active), '--output', output]
        if status:
            return status
        return self.execute(args, output)

    def execute(self, args, output):
        from .wizard import Questions
        count = Questions(self.ask, self.print).number('Параллельность', 4, low=1, high=32, integer=True)
        self.print('API-вызовы используют лимиты провайдера. Во внешние API уйдут задачи и контекст; судье также эталоны и критерии.')
        if (self.value('Запустить? Введите «да», Enter — отмена') or '').lower() != 'да':
            self.print('Запуск отменён; запросов к моделям не было.')
            return 0
        if args[0] in {'run', 'resume'}:
            # Keep the path even after interruption/error so continuation is easy.
            self.last_run = Path(output)
        return self.command([*args, '--concurrency', str(count), '--allow-remote'])

    def results(self):
        source = self.value('Папка результатов', self.last_run)
        if not source:
            return 0
        source = str(Path(source).expanduser().resolve())
        status = self.command(['results', source])
        if status:
            return status
        self.last_run = Path(source)
        self.print('HTML-отчёт: ' + str(Path(source) / 'report.html'))
        if (self.value('Изменить ограничения и пересчитать выбор без API? Введите «да», Enter — назад') or '').lower() != 'да':
            return 0
        from .experiment import read
        policy = read(Path(source) / 'comparison.json')['policy']
        output = self.output('Новая папка отчёта', 'reselection')
        return self.form('reselect', self.policy_fields(policy), [source, '--output', output])

    def calibrate(self):
        if not self.require_active():
            return 0
        labels = self.value('Файл размеченных примеров JSONL', self.active / 'calibration.jsonl')
        output = self.output('Новая папка калибровки', 'calibration')
        return self.execute(['calibrate', str(self.active), '--labels', str(Path(labels).expanduser().resolve()), '--output', output], output)
