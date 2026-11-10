"""Honest coverage-aware reports. Missing judges never become quality scores."""
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path
from .schema import VERSION, ValidationError


def _aggregate(rows):
    counts = Counter(row['status'] for row in rows)
    judge_status = Counter(row['judge']['status'] for row in rows)
    decided = [row for row in rows if row['passed'] is not None]
    scored = [row for row in rows if row['judge']['status'] == 'evaluated']
    passed = sum(row['passed'] is True for row in rows)
    return {'attempts': len(rows), 'passed': passed, 'failed': counts['failed'],
            'not_evaluated': counts['not_evaluated'], 'evaluation_errors': counts['evaluation_error'],
            'decided': len(decided), 'judge_evaluated': len(scored),
            'judge_coverage': len(scored)/len(rows) if rows else 0,
            'judge_status_counts': dict(sorted(judge_status.items())),
            'pass_rate_among_decided': passed/len(decided) if decided else None,
            'all_attempts_pass_lower_bound': passed/len(rows) if decided else None,
            'mean_judge_score': sum(row['score'] for row in scored)/len(scored) if scored else None,
            'candidate_generation_errors': sum(row['response_status'] == 'error' for row in rows),
            'candidate_missing': sum(row['response_status'] == 'missing' for row in rows),
            'diagnostic_check_failures': sum(row['deterministic_status'] == 'failed' for row in rows),
            'judge_provenance_counts': dict(Counter(row['judge']['provenance'] for row in rows))}


def summarize(rows):
    if not rows:
        raise ValidationError('No evaluation records')
    keys = [(r['id'], r['model'], r['configuration'], r['repetition']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValidationError('Duplicate evaluation record')
    if any(r.get('schema_version') != VERSION for r in rows):
        raise ValidationError('Unsupported evaluation schema')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['model'], row['configuration'])].append(row)
    return {'schema_version': VERSION, 'summary': _aggregate(rows),
            'groups': [{'model': model, 'configuration': configuration, **_aggregate(items)}
                       for (model, configuration), items in sorted(groups.items())],
            'limitations': [
                'Rubric score requires a connected judge and is not an objective ground truth.',
                'Unavailable, invalid and missing judgements have null scores, not zero quality.',
                'Programmatic checks are diagnostics only. Only a valid judge judgement determines quality pass/fail.',
                'All-attempts pass lower bound is a coverage bound, not imputed quality.',
                'Multiple repetitions share cases; these descriptive rates have no independence claim.',
                'Prompt isolation and schema validation mitigate but do not eliminate prompt injection.',
                'A model-inference label describes transport execution, not independent verification of the model or its quality.',
            ]}


def _csv_safe(value):
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def write_report(rows, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    fields = ['id', 'model', 'configuration', 'repetition', 'response_status', 'status',
              'passed', 'score', 'deterministic_status', 'judge_status', 'judge_model', 'judge_provenance', 'judge_reason']
    with (output / 'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = {key: row.get(key) for key in fields}
            flattened.update(judge_status=row['judge']['status'], judge_model=row['judge']['model'],
                             judge_provenance=row['judge']['provenance'], judge_reason=row['judge']['reason_code'])
            writer.writerow({key: _csv_safe(value) for key, value in flattened.items()})
    esc = lambda text: html.escape(str(text), quote=True)
    lines = ['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>НИР · Оценка бизнес-текстов</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#183047;background:#f7fafc}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d5e0e8;padding:10px;text-align:left}th{background:#e6eef4}.note{padding:18px;background:#fff3cf}code{overflow-wrap:anywhere}</style>',
             '<h1>Оценка бизнес-текстов</h1><p class="note">Судья не является эталоном истины. Пропуски и ошибки оценки не равны нулевому качеству. Рубрика требует независимой калибровки человеком.</p>',
             '<h2>Сводка</h2><table><tr><th>Модель / конфигурация</th><th>Попытки</th><th>Оценено судьёй</th><th>Passed / failed</th><th>Средний score</th></tr>']
    for group in summary['groups']:
        score = 'нет оценки' if group['mean_judge_score'] is None else f"{group['mean_judge_score']:.3f}"
        lines.append(f"<tr><td>{esc(group['model'])}<br>{esc(group['configuration'])}</td><td>{group['attempts']}</td><td>{group['judge_evaluated']}</td><td>{group['passed']} / {group['failed']}</td><td>{score}</td></tr>")
    lines += ['</table><h2>Покрытие и ограничения</h2><ul>']
    lines.extend('<li>' + esc(text) + '</li>' for text in summary['limitations'])
    lines += ['</ul><h2>Отдельные попытки</h2><table><tr><th>Задача / конфигурация</th><th>Статус</th><th>Диагностика (не оценка)</th><th>Судья</th><th>Причина</th></tr>']
    for row in rows:
        lines.append('<tr>' + ''.join('<td>' + esc(value) + '</td>' for value in [
            row['id'] + ' / ' + row['configuration'] + ' / ' + str(row['repetition']), row['status'],
            row['deterministic_status'], row['judge']['status'], row['judge']['reason_code'] or '']) + '</tr>')
    lines += ['</table><h2>Обоснования судьи</h2>']
    for row in rows:
        lines.append('<details><summary>' + esc(row['id'] + ' / ' + row['configuration']) + '</summary>')
        lines.append('<p>Оценка: ' + esc(row['score'] if row['score'] is not None else 'неизвестно') + '</p>')
        for criterion in row['judge'].get('criteria', []):
            lines.append('<p><b>' + esc(criterion['id']) + ': ' + esc(criterion['score']) + '/4</b> — ' + esc(criterion['rationale']) + '</p>')
            for evidence in criterion['evidence']:
                lines.append('<blockquote>' + esc(evidence['source']) + ': ' + esc(evidence['quote']) + '</blockquote>')
        for critical in row['judge'].get('critical_errors', []):
            lines.append('<p><b>Критическая ошибка ' + esc(critical['id']) + ': ' + esc(critical['detected']) + '</b> — ' + esc(critical['rationale']) + '</p>')
        lines.append('</details>')
    lines += ['<p>Полные ответы: responses.jsonl; критерии и хеши: records.jsonl; время, токены и метаданные генератора: solver_records.jsonl.</p></html>']
    (output / 'report.html').write_text('\n'.join(lines), encoding='utf-8')
    return summary
