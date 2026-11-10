# Настоящий сквозной запуск через MVP CLI

Этот тест специально закрывает отличие между внешним экспериментом с последующим импортом и собственным запуском программы. Здесь **не использовались import-results, replay или controls** для получения сравниваемых ответов.

## Выполненный путь

1. `python -m nir_eval run --config configs/e2e_qwen_direct.json --output runs/e2e_qwen_direct`
2. `python -m nir_eval run --config configs/e2e_qwen_cot.json --output runs/e2e_qwen_cot_recovered`
3. `python -m nir_eval compare runs/e2e_qwen_direct runs/e2e_qwen_cot_recovered --output runs/e2e_qwen_verified_comparison`

Команда run сама загружает модель через встроенный transformers-local, вызывает настоящий model.generate, сохраняет raw outputs, вызывает score_output для каждого ответа и создаёт JSON/CSV/HTML/Markdown-отчёты. Команда compare читает два завершённых native runs и строит сводку; сама она новых inference не делает. Отдельной CLI-команды score в этой версии нет: оценка входит в run.

Модель одна: Qwen2.5-0.5B-Instruct, revision 7ae557604adf67be50417f59c2c2f167def9a775, 494032768 параметров, float32 CPU. Два prompt-протокола: direct и cot. Torch 2.6.0+cpu / Transformers 4.48.3, четыре потока, greedy, max_new_tokens=192, repetition_penalty=1.1. Модель загружается с локального диска, local_files_only=True, trust_remote_code=False. Платных API и сетевого inference нет.

## Проверенный итог

Оба runs завершены: по 12 реальных вызовов на тех же 12 ID исходного пилота, всего 24 завершённых попытки в сравнении.

| Показатель | direct | cot |
|---|---:|---:|
| Canonical match | 0/12 | 0/12 |
| Разбираемые ответы | 10/12 | 6/12 |
| Runtime errors | 0 | 0 |
| Среднее время provider.generate | 2.746 с | 9.164 с |
| Выходные токены | 177 | 822 |

Это доказательство работы пути run→scoring→compare, а не положительного качества маленькой модели и не сравнение двух разных моделей. Повтор того же набора с greedy не создаёт независимую новую выборку для научной оценки качества. Предыдущие импортированные результаты остаются отдельными артефактами.

## Независимый аудит

`audit_native_e2e.py` повторно проверил:
- native manifests имеют status=completed, provenance=model-inference, provider=transformers-local и не являются imported;
- у каждого ровно 12 уникальных запланированных ID, совпадающих с исходным 12-case набором;
- prompt hashes совпадают с whitelisted recipes без gold;
- backend metadata подтверждают реальные Torch/Transformers, model revision и parameter count;
- все 24 сохранённые оценки точно совпадают с независимым повторным вычислением score_output;
- compare содержит ровно 24 неизменённых исходных записи и две конфигурации;
- JSON/CSV/HTML/Markdown отчёты созданы.

Frozen source snapshot обоих runs сохранён. Code SHA-256 обоих: d0b4fb834a03e716c4750cd94ad73d791997a070513553a61c8dfc7b272524d6. Аудит импортирует эту копию, чтобы последующие изменения MVP не изменяли проверку исторического запуска.

Результат проверки — `audit.json`; реальные команды — `completed_commands.sh`; stdout и исходные configs рядом. Финальный HTML находится в `mvp/runs/e2e_qwen_verified_comparison/report.html`.

## Честно сохранённый технический сбой

Первый cot-run в `mvp/runs/e2e_qwen_cot` успел сохранить 7 ответов. После обрыва транспорта исполнитель перестал узнавать session ID. Сохранились manifest=active и lock; доказать состояние старого процесса из нового изолированного namespace нельзя. Lock и данные не удалялись, manifest вручную не менялся.

Вместо слепого resume или слияния частичных файлов выполнен новый полный native cot-run в **новой папке**. Частичный старый run не входит в итоговую таблицу, не импортировался и не подменяет отсутствующие ответы. Это не тест успешного resume. Все 7 сохранённых записей и логи остаются для аудита.
