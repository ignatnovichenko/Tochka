# Точка выбора — исследовательский CLI-прототип

Версия 0.2.0, 10 ноября 2026. Основан на `nir_mvp_judge_final_20261110.zip` (SHA-256 `f81badfdc95135638676234ed035f13c4da6f889bf88e8b249ab7b75214eadae`).

Программа сравнивает конфигурации моделей на задачах компании: ответы → отдельная модель-судья → качество, время, доступные токены и стоимость → отбор по ограничениям. Численные ответы тоже оценивает судья. При сбое судьи оценка **неизвестна**, а не равна нулю. Неизвестная стоимость не считается бесплатной.

## Установка

Требуется Python 3.10+ (проверено на 3.11), для OpenCode — установленный и работающий официальный CLI (проверено на 2.2.6).

```bash
cd mvp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[spreadsheet]'
python -m nir_eval experiment --help
```

Ядро бизнес-сценария не требует сторонних Python-библиотек. Без pip можно сразу использовать `python3 -m nir_eval ...` из этой папки. Дополнение `spreadsheet` нужно для старого табличного модуля и полного набора тестов. Через uv: `uv venv .venv && uv pip install --python .venv/bin/python -e '.[spreadsheet]'`.

## Полный демонстрационный цикл

Используйте новую папку для каждого запуска. Запросы передаются внешнему сервису только с `--allow-remote`.

```bash
python -m nir_eval experiment init my-test --demo
python -m nir_eval experiment models --installed
python -m nir_eval experiment solver-add my-test --provider opencode --model opencode/big-pickle --name pickle-direct --instruction 'Ответьте кратко, без вводных рассуждений.'
python -m nir_eval experiment solver-add my-test --provider opencode --model opencode/minimax-m3.1-flash-free --name minimax-checked --instruction 'Проверьте факты и вычисления перед ответом. Дайте готовый деловой ответ.'
python -m nir_eval experiment judge-set my-test --provider opencode --model opencode/glm-5.3-flash-free --max-calls 40 --structured-retries 1
python -m nir_eval experiment policy-set my-test --min-quality 0.8 --min-pass-rate 1 --max-latency 60 --max-cost 0
python -m nir_eval experiment calibrate my-test --labels my-test/calibration.jsonl --output runs/my-calibration --allow-remote
python -m nir_eval experiment run my-test --output runs/my-business --allow-remote
python -m nir_eval experiment results runs/my-business
```

Откройте `runs/my-business/report.html` в браузере. Сводка также находится в `comparison.json` и `comparison.csv`; ссылки в HTML ведут к подробным оценкам судьи. `selected` означает один лучший подходящий вариант, `tied_configurations` — несколько равных лидеров, `no_eligible_configuration` — подходящих нет. Это описательный выбор на проверенных задачах, не гарантия будущего качества.

Команда `run` запрещает одинаковую модель у solver и judge. Несколько конфигураций одной модели задаются несколькими `solver-add` с разными `--name` и `--instruction`. Параметры температуры и лимита токенов OpenCode CLI не предоставляет: программа явно отклоняет такие параметры вместо притворной настройки.

## Свои задачи, документы и критерии — без изменения кода

```bash
python -m nir_eval experiment init company-test
python -m nir_eval experiment rubric-add company-test --id returns --criterion 'facts=Ответ опирается только на правила возврата' --criterion 'action=Понятно следующее действие клиента' --critical 'false_promise=Выдуманы обещания или условия' --pass-threshold 0.75
python -m nir_eval experiment task-add company-test --id return-001 --rubric returns --task 'Ответьте клиенту на запрос возврата' --context 'Покупка 5 дней назад; товар не использован; чек есть.' --document return-policy.txt --reference 'Возврат допустим при выполнении условий правил.'
```

`--document` принимает UTF-8 текст или Markdown, можно указать несколько файлов. Содержимое сохраняется в контексте задачи; PDF/DOCX сначала нужно экспортировать в текст. Эталон и рубрика скрыты от решающей модели. Можно импортировать готовые JSONL:

```bash
python -m nir_eval experiment import company-test --cases my-cases.jsonl --rubrics my-rubrics.jsonl
python -m nir_eval business validate --cases company-test/cases.jsonl --rubrics company-test/rubrics.jsonl
```

Образцы формата: `nir_eval/demo/cases.jsonl` и `nir_eval/demo/rubrics.jsonl`. `init --demo` создаёт 4 синтетические задачи: письмо поставщику, резюме встречи, рост выручки и возврат товара. Это компактная демонстрация, не репрезентативная выборка компании.

## Локальные модели

Запустите свой сервер с OpenAI-совместимым `/v1/chat/completions` (например, llama.cpp или Ollama с уже загруженными весами), затем:

```bash
python -m nir_eval experiment solver-add company-test --provider local --base-url http://127.0.0.1:8080/v1 --model local-solver --name local-t0 --temperature 0 --max-tokens 512
python -m nir_eval experiment judge-set company-test --provider local --base-url http://127.0.0.1:8081/v1 --model local-judge --temperature 0 --max-tokens 2048
```

Имена должны соответствовать моделям серверов. Локальный транспорт проверен автоматическими тестами с реальным localhost HTTP-сервером; локальный нейросетевой inference на этом ПК в данной поставке не выполнялся. Веса и серверы в архив не включены. Продвинутые адаптеры command/Transformers и прежний FinQA CLI сохранены.

## Ограничения, повторный отбор и диагностика

Изменить ограничения можно без повторных модельных вызовов:

```bash
python -m nir_eval experiment policy-set my-test --min-quality 0.95 --max-latency 10
python -m nir_eval experiment select runs/my-business --policy my-test/policy.json --output runs/my-strict-selection.json
```

Среднее качество считается только по валидным оценкам; по умолчанию отбор требует покрытия судьёй 100%, отсутствия ошибок генерации и критических ошибок. При заданном лимите стоимости её неизвестное значение исключает конфигурацию. Стоимость solver+judge складывается только при наличии обоих значений. Для OpenCode сохраняется нативная структура токенов; она не объявляется эквивалентом OpenAI `prompt_tokens`.

Каждый run содержит неизменяемые входные копии, план, хеши, ответы, сырые судейские ответы, метаданные и времена. Время — длительность вызова провайдера, включая запуск CLI; total — solver+judge. Это не чистая скорость декодирования. Повторный запуск в существующую выходную папку запрещён. Сетевых повторов, resume и платного fallback нет. Для нарушений JSON/цитат судьи можно явно включить `--structured-retries 1` (по умолчанию 0; максимум 2). Каждая попытка сохраняется, входит в `max_calls`, общее время и стоимость. Успешные оценки по качеству не перезапрашиваются; повтор нужен только при невалидной структуре. `max_calls` задаётся на один solver run или одну калибровку. При 429 адаптер сохраняет локальный срок ожидания и не продолжает сетевые обращения до него.

**OpenCode:** используйте обычный установленный CLI. В проверенной среде обычный `run` работал, а отдельная конфигурация `opencode.json` с custom eval-agent/deny permissions возвращала 429. Адаптер не меняет ваш профиль, не читает ключи и не обращается напрямую к API. Подробнее: `docs/OPENCODE_CLI_RU.md`.

Набор calibration содержит заранее заданные положительные/отрицательные ответы и ожидаемые pass/critical. Это авторская синтетическая разметка, не независимая человеческая экспертиза. Подробности фактических прогонов: `RESULTS_RU.md`.

Тесты:

```bash
python scripts/check_all.py
```

Результаты `tests/` — проверки программы, не качество моделей. Исторические FinQA, локальные smoke и прежние попытки OpenCode сохранены в `evidence/` и старых каталогах `runs/`; они не смешиваются с текущим бизнес-прогоном. Презентация не изменялась.

## Границы прототипа

Нет GUI, многопользовательского доступа, распределённой очереди, независимой сертификации судьи и статистической гарантии качества. Инструкции «не использовать инструменты» и пустая рабочая папка OpenCode не являются OS sandbox. Демонстрационные данные синтетические; перед работой с конфиденциальными данными проверьте политику провайдера. Для бесплатных моделей доступность и тарифы могут меняться; текущий явный allowlist не обещает вечную доступность. Ключи не включайте в JSON/архив и не присылайте в чат.

## Продуктовое направление

[Концепция Tochka](docs/PRODUCT_VISION_RU.md): универсальный подбор конфигураций и маршрутизация моделей для AI-агентов в разных бизнес-процессах. Новый тип процесса инициирует сравнительную оценку, а её результаты становятся основой маршрута для последующих задач. Это направление развития, а не описание готовых возможностей MVP.
