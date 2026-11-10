# Локальные модели и нулевой бюджет API

## Проверенная конфигурация

- Qwen/Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Python 3.12.14; PyTorch 2.6.0+cpu; Transformers 4.48.3; CPU, float32.
- Около 494 млн параметров; веса ~954 МБ на диске. Загрузка занимает дополнительное время и RAM.
- Проверенный пилот использовал greedy decoding, `max_new_tokens=192`, `num_beams=1`, `use_cache=true`; upstream `repetition_penalty=1.1` сохранён. `temperature/top_p/top_k` не влияют на greedy decoding.
- Базовая конфигурация генерации и lock зависимостей сохранены в `evidence/qwen_0_5b/`. Пилот выполнен отдельным batch runner; встроенный `transformers-local` дополнительно проверен настоящим одиночным интеграционным вызовом.

## Подготовка

Создайте отдельное виртуальное окружение. Для CPU используйте официальный индекс PyTorch, чтобы не устанавливать многогигабайтные CUDA зависимости:

```bash
python -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install transformers==4.48.3
```

Скачайте модель из официального репозитория Hugging Face на фиксированной revision стандартным клиентом Hugging Face. Не запускайте код из model repository. Нужны tokenizer/config/generation_config и safetensors. Проверьте контрольные суммы по источнику и сохраните manifest. В текущей рабочей среде модель уже находится в `../experiments/model`; её веса не входят в архив MVP.

Параметры `provider.model_path`, `provider.revision`, `model` в `configs/qwen_local_example.json` необходимо привести к вашей локальной копии. Все относительные пути разрешаются от каталога config-файла.

```bash
.venv/bin/python -m nir_eval run --config configs/qwen_local_example.json --output runs/qwen_new_run
```

Это новый настоящий запуск, поэтому его время и ответы могут отличаться на другом железе и с другим числом потоков. `--resume` принимает только неизменные данные, код и конфиг. Веса также должны оставаться теми же; runner фиксирует указанную revision, но не гарантирует подлинность произвольного локального model_path. Сверьте независимый manifest файлов.

## Защита и ограничения

- `local_files_only=True`, `trust_remote_code=False`; загрузчик не получает разрешения выполнять пользовательский Python из модели.
- Только CPU и явно выбранный dtype. Автоматического GPU/offloading/квантования нет.
- Модель сохраняется в памяти между попытками. Загрузка вынесена в `provider_metadata.model_load_seconds`; per-attempt latency включает токенизацию/генерацию/декодирование встроенного adapter.
- В исходном пилоте latency measured around `model.generate`, поэтому небольшое отличие границ измерения указано в methodology. В сравнении новых конфигураций используйте одинаковый runner.
- Нет молчаливого усечения контекста: превышение окна даёт явную ошибку. Значения стоимости электроэнергии, аренды CPU и TTFT не выдумываются.
- Путь command-адаптера позволяет выполнить доверенную локальную программу, а не sandbox произвольного модельного кода. Никогда не задавайте как команду текст, сгенерированный моделью.
