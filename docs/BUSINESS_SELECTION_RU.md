> Историческое описание исходного прототипа. Актуальная поставка: `README.md`, `RESULTS_RU.md` и `CHANGELOG_RU.md` в корне.

# Выбор конфигурации по оценкам нейросудьи

`business_selection.py` содержит read-only функцию `select_business_run(run_dir, policy)`.
Она не вызывает модели, не использует качество FinQA и не пишет в каталог прогона.
CLI: `python -m nir_eval business select --run runs/NAME --policy configs/business_select_quality.json --output runs/my_selection.json`.

## Вход

Только завершенный `business-pipeline-v1` со `state=completed` и `target_quality_method=neural_judge_only`.
Три файла должны совпадать с SHA-256 в manifest: `responses.jsonl`, `solver_records.jsonl`, `records.jsonl`.
Проверяются одинаковые ключи `id/model/configuration/repetition`, уникальность, точное покрытие всех задач,
непрерывные повторы от нуля, counts, происхождение настоящего inference, связи ответа/prompt/оценки.
Нужны `case_ids`, `solver_model`, `planned_attempts`, `solver_completed_attempts`, `evaluated_attempts`,
`judge_evaluated`, fingerprints данных/config. Новые manifests должны иметь `configuration_names` и `repetitions`.
Старый manifest без двух последних полей принимается с предупреждением: план восстанавливается из групп и planned count.
Хеши не являются цифровой подписью и не доказывают достоверность источника сами по себе.

## Политика

- `min_judge_coverage`: доля всех попыток с действительной оценкой; по умолчанию **1.0**.
- `min_mean_judge_score`: минимум среднего neural score 0–1 по оцененным ответам; по умолчанию 0.
- `max_generation_error_rate`: максимум доли ошибок генератора среди всех попыток; по умолчанию **0**.
- `max_judge_critical_error_rate`: максимум доли оцененных ответов с критической ошибкой, обнаруженной судьей; по умолчанию **0**. Это нейрооценка, не проверка кодом.
- `min_judge_pass_rate`: необязательный минимум доли прошедших ответов среди оцененных судьей.
- `min_tasks`: минимум разных задач; по умолчанию 1.
- `max_mean_solver_latency_seconds`, `max_mean_total_latency_seconds`: необязательные ограничения времени.
  Короткие aliases `max_mean_solver_latency` / `max_mean_total_latency` тоже поддержаны, единицы секунды.
- `max_mean_cost_usd`: необязательная верхняя граница среднего полного расхода на попытку.
- `objective`: только `quality`. Скрытого tiebreak по скорости или цене нет.

Рекомендуемое начало: `{"min_judge_coverage":1,"min_mean_judge_score":0.75,"max_generation_error_rate":0}`.
Порог 0.75 учебный, не утвержденный бизнес-SLA. Требования и рубрики пользователь задает заранее.

## Пропуски и ограничения

Состояния judge `unavailable`, `error`, `not_evaluated` имеют `score=null`, `passed=null`.
Они не становятся нулевым качеством. Даже при пороге coverage=0 без хотя бы одной оценки выбор невозможен.
Средний score вычисляется только по оцененным ответам, и рядом всегда выводятся coverage и число неизвестных.
`checks`/`deterministic_status` не участвуют в качестве, фильтрации или рейтинге.
Настоящий neural score=0, напротив, является валидной оценкой.

Затраты неизвестны, пока manifest не содержит точный контракт
`cost_measurement={"currency":"USD","scope":"solver_plus_judge_per_attempt","basis":"provider_reported"}`,
а каждая строка evaluations — конечное неотрицательное `total_cost_usd`.
Нулевое число платных API-вызовов не превращает неизвестную стоимость в 0.

Для solver latency нужны finite `solver_records.latency_seconds` и единый непустой `latency_scope`.
Для total нужны все solver/judge latency, непустой manifest `judge_latency_scope` и
`total_latency_scope="solver_plus_judge_call_seconds"`. Это **сумма времени двух вызовов**,
а не пользовательское end-to-end время с загрузкой, очередью и подготовкой отчетов.
При отсутствующих метриках ограничение отклоняет конфигурацию с reason, а не угадывает число.

## Выход

`selected`, `tied_configurations` либо `no_eligible_configuration`.
В каждом кандидате все метрики и `rejection_reasons`. При равенстве score до 1e-12 возвращаются все победители без произвольного выбора.
Невалидные/незавершенные/измененные evidence вызывают ValidationError, а не маскируются под плохое качество.
Вывод описывает только сохраненную выборку. Надежность судьи независимо не установлена; это не SLA и не статистическое доказательство превосходства.

## Проверки

`python -m unittest discover -s tests -p test_business_selection.py -v`

Fixtures синтетические и не являются измерениями моделей. Тесты не вызывают API, модели или интернет.
