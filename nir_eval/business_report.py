"""Decision-first, coverage-aware business reports from saved evidence only."""
from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
import statistics

from .business.schema import ValidationError, load_jsonl

UNKNOWN = 'неизвестно'
REASONS = {
    'too_few_tasks': 'Недостаточно задач для выбора',
    'judge_coverage_below_threshold': 'Судья оценил недостаточно ответов',
    'judge_quality_unknown': 'Качество неизвестно: нет валидных оценок судьи',
    'judge_score_below_threshold': 'Среднее качество ниже вашего порога',
    'generation_error_rate_above_limit': 'Слишком много ошибок получения ответа',
    'neural_critical_error_rate_above_limit': 'Обнаружены критические ошибки',
    'judge_pass_rate_unknown': 'Доля успешно выполненных задач неизвестна',
    'judge_pass_rate_below_threshold': 'Недостаточно задач прошли проверку',
    'solver_latency_unknown': 'Время выполнения неизвестно',
    'solver_latency_above_limit': 'Выполнение задач медленнее допустимого',
    'total_latency_unknown': 'Общее время выполнения и проверки неизвестно',
    'total_latency_above_limit': 'Общее время выполнения и проверки превышает лимит',
    'cost_unknown': 'Стоимость неизвестна, поэтому лимит цены проверить нельзя',
    'cost_above_limit': 'Стоимость превышает ваш лимит',
    'incomparable_solver_latency_scopes': 'Время измерено разными способами',
}
JUDGE_REASONS = {
    'judge_invalid_structured_output': 'Судья нарушил формат ответа или требования к цитатам',
    'judge_call_budget_exhausted': 'Исчерпан лимит запросов к судье',
    'judge_connection_unavailable': 'Сервис судьи недоступен',
    'judge_provider_error': 'Ошибка сервиса или ответа судьи',
    'judge_not_configured': 'Судья не настроен',
    'candidate_generation_error': 'Решающая модель не вернула ответ',
    'candidate_missing': 'Ответ решающей модели отсутствует',
    'judge_opencode_free_limit': 'Исчерпан лимит провайдера',
    'judge_opencode_cooldown_active': 'Ожидание восстановления лимита провайдера',
    'judge_http_429': 'Провайдер ограничил частоту запросов или исчерпан лимит',
    'judge_http_401': 'Провайдер не принял авторизацию',
    'judge_http_403': 'У провайдера нет доступа к выбранной модели',
}
POLICY_LABELS = {
    'min_mean_judge_score': ('Минимальное среднее качество', 'percent'),
    'min_judge_coverage': ('Минимальная полнота оценки судьёй', 'percent'),
    'min_judge_pass_rate': ('Минимальная доля прошедших проверку', 'percent'),
    'max_generation_error_rate': ('Допустимая доля ошибок генерации', 'percent'),
    'max_judge_critical_error_rate': ('Допустимая доля критических ошибок', 'percent'),
    'min_tasks': ('Минимум задач', 'number'),
    'max_mean_solver_latency_seconds': ('Максимальное среднее время ответа', 'seconds'),
    'max_mean_total_latency_seconds': ('Максимальное среднее время с проверкой', 'seconds'),
    'max_mean_cost_usd': ('Максимальная средняя стоимость с проверкой', 'money'),
}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _percent(value):
    return f'{value * 100:.1f}%' if _number(value) else UNKNOWN


def _seconds(value):
    return f'{value:.2f} с' if _number(value) else UNKNOWN


def _money(value):
    if not _number(value):
        return UNKNOWN
    if value == 0:
        return '$0'
    return f'${value:.3g}' if abs(value) < .000001 else f'${value:.6f}'.rstrip('0').rstrip('.')


def _plain(value):
    # Prevent stored model/config strings from injecting terminal control sequences.
    return ''.join(c if c.isprintable() else ' ' for c in str(value))


def _name(candidate):
    return f'{candidate.get("configuration", "без названия")} · {candidate.get("model", UNKNOWN)}'


def _coverage(candidate):
    evaluated = candidate.get('judge_evaluated', 0)
    planned = candidate.get('attempts', 0)
    return f'{evaluated} из {planned}'


def _incomplete(candidate):
    return candidate.get('judge_evaluated', 0) < candidate.get('attempts', 0)


def _reason_text(reason):
    return REASONS.get(reason, 'Ограничение не выполнено: ' + str(reason))


def _decision(result):
    status = result.get('status')
    if status == 'selected' and result.get('selected'):
        return 'Под ваши требования подходит', _name(result['selected'])
    if status == 'tied_configurations' and result.get('tied'):
        return 'Несколько подходящих вариантов с одинаковым качеством', '; '.join(_name(c) for c in result['tied'])
    return 'Подходящих конфигураций нет', 'Смотрите причины исключения. Измените ограничения или завершите недостающие проверки.'


def terminal_summary(result):
    """Return plain Russian text with coverage kept next to any quality percentage."""
    candidates = result.get('candidates', [])
    title, decision = _decision(result)
    planned = sum(c.get('attempts', 0) for c in candidates)
    evaluated = sum(c.get('judge_evaluated', 0) for c in candidates)
    lines = ['Tochka · Результаты эксперимента', f'{title}: {_plain(decision)}',
             f'Конфигураций: {len(candidates)} · Оценено ответов: {evaluated} из {planned}', '']
    for c in candidates:
        lines += [_plain(_name(c)),
                  f'  Качество: {_percent(c.get("mean_judge_score"))} по {_coverage(c)} оценённым ответам; '
                  f'прошли проверку: {_percent(c.get("judge_pass_rate_among_evaluated"))} среди оценённых.',
                  f'  Время ответа: {_seconds(c.get("mean_solver_latency_seconds"))}; '
                  f'с проверкой: {_seconds(c.get("mean_total_latency_seconds"))}; '
                  f'стоимость с проверкой: {_money(c.get("mean_cost_usd"))} / задачу.']
        if _incomplete(c):
            lines.append('  НЕПОЛНАЯ ПРОВЕРКА: неизвестные оценки не равны нулю; среднее относится только к оценённым ответам.')
        reasons = c.get('rejection_reasons', [])
        lines.append('  ' + ('Не подходит: ' + '; '.join(_plain(_reason_text(r)) for r in reasons) if reasons else 'Соответствует заданным ограничениям.'))
    lines += ['', 'Оценки описывают эту выборку и мнение судьи, а не гарантируют качество в эксплуатации.',
              'Неизвестная стоимость не считается нулевой. Цена / задача включает выполнение и оценку.']
    return '\n'.join(lines)


def _safe_path(root, relative):
    relative = Path(str(relative))
    if relative.is_absolute() or '..' in relative.parts:
        raise ValidationError('Report source must stay inside the saved experiment')
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValidationError('Report source must stay inside the saved experiment')
    return result


def _rows(root, relative):
    path = _safe_path(root, relative)
    return load_jsonl(path) if path.is_file() and path.stat().st_size else []


def _identity(row):
    return (row.get('id'), row.get('model'), row.get('configuration'), row.get('repetition', 0))


def _evidence(root, candidate):
    run = candidate.get('run')
    if not run:
        return [], {}, {}, {}
    folder = _safe_path(root, run)
    records = [r for r in _rows(folder, 'records.jsonl')
               if r.get('model') == candidate.get('model') and r.get('configuration') == candidate.get('configuration')]
    responses = {_identity(r): r for r in _rows(folder, 'responses.jsonl')}
    solver = {_identity(r): r for r in _rows(folder, 'solver_records.jsonl')}
    cases = _rows(root, 'inputs/cases.jsonl') or _rows(root, 'cases.jsonl')
    return records, responses, solver, {c['id']: c for c in cases}


def _known_average(values):
    return statistics.fmean(values) if values and all(_number(x) and x >= 0 for x in values) else None


def _costs(candidate, records, solvers):
    sc = candidate.get('mean_solver_cost_usd')
    jc = candidate.get('mean_judge_cost_usd')
    if sc is None and len(records) == candidate.get('attempts'):
        sc = _known_average([solvers.get(_identity(r), {}).get('provider_metadata', {}).get('cost_usd') for r in records])
    if jc is None and len(records) == candidate.get('attempts'):
        jc = _known_average([r.get('judge', {}).get('cost_usd') for r in records])
    return sc, jc


def _csv_safe(value):
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def _esc(value):
    return html.escape(str(value), quote=True)


def _answer_details(candidate, evidence):
    records, responses, solvers, cases = evidence
    if not records:
        return '<p class="muted">Подробные записи задач недоступны в исходной папке. Сводные метрики сохранены.</p>'
    result = []
    statuses = {'passed': 'Проверка пройдена', 'failed': 'Проверка не пройдена',
                'evaluation_error': 'Оценка неизвестна: ошибка судьи', 'not_evaluated': 'Оценка неизвестна'}
    for row in records:
        judge = row.get('judge', {})
        known = judge.get('status') == 'evaluated'
        key = _identity(row)
        response = responses.get(key, {})
        case = cases.get(row.get('id'), {})
        result.append('<details class="task"><summary><span>' + _esc(row.get('id', 'Задача')) +
                      ' · повтор ' + _esc(row.get('repetition', 0) + 1) + '</span><span class="tag">' +
                      _esc(statuses.get(row.get('status'), 'Оценка неизвестна')) + '</span></summary>')
        if case:
            result.append('<h4>Задание</h4><pre>' + _esc(case.get('task', '')) + '</pre>')
            result.append('<details><summary>Контекст и эталон</summary><h4>Контекст</h4><pre>' + _esc(case.get('context', '')) +
                          '</pre><h4>Эталон (не передавался решающей модели)</h4><pre>' + _esc(case.get('reference', 'Не задан')) + '</pre></details>')
        result.append('<h4>Ответ модели</h4><pre>' + _esc(response.get('response') or 'Ответ не получен') + '</pre>')
        result.append('<p><b>Оценка судьи:</b> ' + _esc(_percent(judge.get('score')) if known else UNKNOWN) + '</p>')
        if not known:
            result.append('<p class="notice">Этот ответ не включён в среднее качество. Причина: ' + _esc(JUDGE_REASONS.get(judge.get('reason_code'), 'Не удалось получить и проверить вердикт судьи')) + '.</p>')
        else:
            for criterion in judge.get('criteria', []):
                result.append('<h4>' + _esc(criterion.get('id', 'Критерий')) + ' — ' + _esc(criterion.get('score', UNKNOWN)) +
                              '/4</h4><p>' + _esc(criterion.get('rationale', '')) + '</p>')
                for quote in criterion.get('evidence', []):
                    result.append('<blockquote>' + _esc(quote.get('quote', '')) + '<footer>Источник: ' + _esc(quote.get('source', '')) + '</footer></blockquote>')
            errors = judge.get('critical_errors', [])
            if errors:
                result.append('<h4>Критические ошибки</h4><ul>')
                for error in errors:
                    result.append('<li><b>' + _esc(error.get('id', '')) + ': ' + ('обнаружена' if error.get('detected') else 'не обнаружена') + '</b> — ' + _esc(error.get('rationale', '')) + '</li>')
                result.append('</ul>')
        generation = solvers.get(key, {})
        result.append('<p class="muted">Выполнение: ' + _esc(_seconds(generation.get('latency_seconds'))) +
                      ' · Проверка: ' + _esc(_seconds(judge.get('latency_seconds'))) +
                      ' · Стоимость вместе: ' + _esc(_money(row.get('total_cost_usd'))) + '</p></details>')
    return ''.join(result)


STYLE = '''
:root{color-scheme:light;--ink:#172c3d;--muted:#526679;--line:#dce5ec;--paper:#fff;--bg:#f3f6fa;--accent:#145e53;--warning:#805300}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}main{max-width:1240px;margin:auto;padding:36px 24px 64px}header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:16px}.brand{font-weight:800;font-size:24px;letter-spacing:-.5px}.muted,small{color:var(--muted)}h1{font-size:clamp(25px,4vw,38px);line-height:1.2;margin:10px 0 14px}h2{font-size:23px;margin-top:32px}h3{font-size:18px}h4{margin-bottom:6px}.hero{padding:26px;margin:24px 0;background:var(--paper);border:1px solid var(--line);border-left:6px solid var(--accent);border-radius:12px}.hero.warning{border-left-color:var(--warning)}.eyebrow{font-size:12px;font-weight:750;text-transform:uppercase;letter-spacing:1px;color:var(--accent)}.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.kpi{padding:16px 20px;background:var(--paper);border:1px solid var(--line);border-radius:10px}.kpi b{display:block;font-size:28px}.notice{padding:12px 16px;background:#fff4d9;border-left:3px solid #cb941f;border-radius:4px}.table-wrap{overflow:auto;background:var(--paper);border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:14px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#eaf0f5;font-size:12px}td strong{display:block}td small{display:block;max-width:260px}.tag{display:inline-block;border-radius:5px;background:#e5f1ed;color:#174b42;padding:3px 7px;font-size:12px}.tag.warn{background:#fff1cf;color:#705000}.reasons{margin:8px 0;padding-left:18px}.task{background:var(--paper);border:1px solid var(--line);border-radius:9px;padding:0 18px;margin:12px 0}.task[open]{padding-bottom:16px}summary{cursor:pointer;font-weight:650;padding:13px 0}summary .tag{margin-left:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace;background:#f3f6f8;padding:14px;border-radius:6px}blockquote{border-left:3px solid #9bb5c6;margin:12px 0;padding:4px 16px;background:#f7fafc}blockquote footer{font-size:12px;color:var(--muted)}.policy{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 24px}.policy div{display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-bottom:1px solid var(--line)}.downloads a{color:#145e53;margin-right:20px}a:focus-visible,summary:focus-visible{outline:3px solid #386ba5;outline-offset:4px}.footer{border-top:1px solid var(--line);margin-top:32px;padding-top:16px;font-size:13px}@media(max-width:680px){main{padding:20px 14px}.kpis,.policy{grid-template-columns:1fr}.hero{padding:20px}header{align-items:flex-start;gap:10px}header small{text-align:right}}@media print{body{background:white}main{max-width:none;padding:0}.table-wrap{overflow:visible}details{break-inside:avoid}.downloads{display:none}}
'''


def write_business_report(output: Path, result: dict, source_root: Path | None = None):
    """Write self-contained HTML/CSV/JSON. Read raw evidence; never mutate it."""
    output = Path(output)
    root = Path(source_root) if source_root is not None else output
    candidates = result.get('candidates', [])
    # Resolve and read all optional evidence before writing any report files.
    details = [_evidence(root, c) for c in candidates]
    costs = [_costs(c, e[0], e[2]) for c, e in zip(candidates, details)]
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    fields = ['model', 'configuration', 'attempts', 'judge_evaluated', 'judge_unknown', 'judge_coverage',
              'mean_judge_score', 'judge_pass_rate_among_evaluated', 'mean_solver_latency_seconds',
              'mean_total_latency_seconds', 'mean_solver_cost_usd', 'mean_judge_cost_usd', 'mean_cost_usd',
              'cost_basis', 'rejection_reasons']
    with (output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for candidate, (sc, jc) in zip(candidates, costs):
            row = {k: candidate.get(k) for k in fields}
            row.update(mean_solver_cost_usd=sc, mean_judge_cost_usd=jc)
            writer.writerow({k: _csv_safe(v) for k, v in row.items()})
    title, decision = _decision(result)
    evaluated = sum(c.get('judge_evaluated', 0) for c in candidates)
    planned = sum(c.get('attempts', 0) for c in candidates)
    unknown = planned - evaluated
    body = ['<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
            '<meta name="color-scheme" content="light"><title>Tochka · Результаты выбора модели</title><style>', STYLE,
            '</style></head><body><main><header><div class="brand">Tochka<span class="muted"> / Точка выбора</span></div><small>Выбор ИИ на задачах бизнеса<br>Отчёт по сохранённым результатам</small></header>',
            '<section class="hero' + (' warning' if result.get('status') == 'no_eligible_configuration' else '') + '"><div class="eyebrow">Результат отбора</div><h1>' + _esc(title) + '</h1><p>' + _esc(decision) + '</p>',
            '<p class="muted">Выбор основан на ваших ограничениях ниже. Среди подходящих конфигураций сравнивается среднее качество; одинаковые результаты показаны как ничья.</p></section>',
            '<section class="kpis" aria-label="Полнота эксперимента"><div class="kpi"><b>' + _esc(len(candidates)) + '</b>конфигураций сравнивается</div><div class="kpi"><b>' + _esc(evaluated) + ' / ' + _esc(planned) + '</b>ответов оценено судьёй</div><div class="kpi"><b>' + _esc(unknown) + '</b>оценок остаются неизвестными</div></section>']
    if unknown:
        body.append('<p class="notice"><b>Проверка неполная.</b> Средние оценки и доли успешных ответов рассчитаны только по валидным вердиктам судьи. Пропущенные ответы не получили ни ноль, ни положительную оценку.</p>')
    body += ['<h2>Сравнение вариантов</h2><p class="muted">Качество — средний балл судьи по критериям. Полнота — сколько ответов удалось проверить. Стоимость указана в USD в среднем на одну задачу.</p>',
             '<div class="table-wrap" role="region" aria-label="Сравнение конфигураций" tabindex="0"><table><thead><tr><th scope="col">Конфигурация</th><th scope="col">Качество и полнота</th><th scope="col">Прошли проверку</th><th scope="col">Время / задачу</th><th scope="col">Стоимость / задачу</th><th scope="col">Соответствие требованиям</th></tr></thead><tbody>']
    for c, (sc, jc) in zip(candidates, costs):
        incomplete = _incomplete(c)
        reasons = c.get('rejection_reasons', [])
        cost_basis = {'provider_reported': 'по данным провайдера', 'configured_rates': 'расчёт по заданным тарифам', 'estimated': 'оценка по тарифам'}.get(c.get('cost_basis'), 'согласно сохранённым данным')
        body.append('<tr><td><strong>' + _esc(c.get('configuration', '')) + '</strong><small>' + _esc(c.get('model', '')) + '</small></td><td><strong>' + _esc(_percent(c.get('mean_judge_score'))) + '</strong><small>Оценено ' + _esc(_coverage(c)) + '</small>' + ('<span class="tag warn">Неполная проверка</span>' if incomplete else '<span class="tag">Полная проверка</span>') + '</td><td>' + _esc(_percent(c.get('judge_pass_rate_among_evaluated'))) + '<small>среди оценённых</small><small>Критические ошибки: ' + _esc(_percent(c.get('judge_critical_error_rate_among_evaluated'))) + '</small></td><td><strong>' + _esc(_seconds(c.get('mean_solver_latency_seconds'))) + '</strong><small>С проверкой: ' + _esc(_seconds(c.get('mean_total_latency_seconds'))) + '</small></td><td><strong>' + _esc(_money(c.get('mean_cost_usd'))) + '</strong><small>Выполнение: ' + _esc(_money(sc)) + '<br>Судья: ' + _esc(_money(jc)) + '<br>' + _esc(cost_basis) + '</small></td><td><span class="tag' + (' warn' if reasons else '') + '">' + ('Не подходит' if reasons else 'Подходит') + '</span>' + ('<ul class="reasons">' + ''.join('<li>' + _esc(_reason_text(r)) + '</li>' for r in reasons) + '</ul>' if reasons else '<small>Все заданные ограничения выполнены</small>') + '</td></tr>')
    body += ['</tbody></table></div><p class="muted">Неизвестная цена не считается нулевой. Время включает вызов провайдера, а время с проверкой — также судью. Результаты по этой выборке не гарантируют качество или скорость в эксплуатации.</p>',
             '<h2>Ваши ограничения</h2><div class="policy">']
    for key, value in result.get('policy', {}).items():
        if key not in POLICY_LABELS or value is None:
            continue
        label, fmt = POLICY_LABELS[key]
        rendered = {'percent': _percent, 'seconds': _seconds, 'money': _money, 'number': str}[fmt](value)
        body.append('<div><span>' + _esc(label) + '</span><b>' + _esc(rendered) + '</b></div>')
    body += ['</div><p class="muted">Пороги можно изменить и пересчитать выбор по этим данным без повторных запросов моделям.</p>',
             '<h2>Ответы и объяснения судьи</h2>']
    for candidate, evidence in zip(candidates, details):
        body.append('<h3>' + _esc(_name(candidate)) + '</h3>' + _answer_details(candidate, evidence))
    body += ['<section class="downloads"><h2>Файлы для дальнейшей работы</h2><a href="comparison.csv" download>Таблица для Excel · CSV</a><a href="comparison.json" download>Полные метрики · JSON</a></section>',
             '<footer class="footer"><b>Границы выводов.</b> Судья — отдельная модель, а не независимый эталон истины. Высокая полнота означает наличие валидных оценок, но не доказывает их содержательную точность. Проверяйте калибровку и спорные ответы перед применением к задачам компании. Этот отчёт создан из сохранённых данных без новых модельных запросов.</footer></main></body></html>']
    (output / 'report.html').write_text(''.join(body), encoding='utf-8')
    return {'html': output / 'report.html', 'csv': output / 'comparison.csv', 'json': output / 'comparison.json'}
