# Приложение: реальная скорость CPU и Colab Tesla T4

**Это отдельная проверка скорости, не новые результаты качества на 12 задачах.** Использованы два одинаковых prompts одного задания FinQA, direct и CoT, на CPU и GPU. Основная таблица 60 benchmark-попыток не изменяется.

| Метрика | CPU | Colab T4 |
|---|---:|---:|
| Native decode, токенов/с | 5.096 | 27.149 |
| Native prefill, токенов/с | 20.825 | 720.208 |
| Выходные токены / полное время вызовов, токенов/с | 0.813 | 14.100 |
| Сумма полного времени двух вызовов, с | 121.748 | 7.092 |
| Выходных токенов | 99 | 100 |

Native decode: описательно **5.33×**, prefill **34.58×**. Каждая скорость рассчитана как сумма соответствующих native counters / сумма времени, не как среднее двух скоростей.

## Что именно сравнивается

Одна pinned Qwen2.5-7B-Instruct-GGUF Q4_K_M, llama-cpp-python 0.3.16, одинаковые model revision, prompts, seed, greedy parameters, max 192, n_ctx 4096 и batch 512. CPU: Xeon Platinum 8573C, 9 видимых vCPU, 4 потока. GPU: Colab Tesla T4; native log подтверждает offload 29/29 слоёв, CUDA buffers. GPU использует официальный cu124 wheel, CPU — native CPU build. Точный runtime и аппаратные ограничения приведены в `../../evidence/performance/colab_gpu_speed_probe/independent_review/README_RU.md`.

Выходы GPU отличаются от CPU, включая 100 против 99 токенов. Поэтому одинаковые входы не означают одинаковую вычислительную траекторию. Отношение полного wall time 17.17× — описательное, не строгий коэффициент одинаковой генерации.

## Проверка и ограничения

- Повторно проверены SHA исходного GPU ZIP, CRC, совпадение распакованных файлов, SHA raw records и prompts.
- Оба prompts побайтно совпадают между CPU и GPU; model revision, квантование и параметры совпадают.
- Native offload log подтверждает 29/29 слоёв на GPU; одного наличия torch.cuda для такого вывода недостаточно.
- Счётчики положительны, конечны и пересчитаны из raw данных. Их нативные token counts не всегда тождественны API completion token count.
- Только два prompts одного задания, без повторений, отдельного warm-up или доверительного интервала скорости. Меняется машина и CPU/CUDA build. Colab host имеет 2 видимых vCPU при n_threads=4.
- Native decode исключает prefill и часть host/sampling overhead. End-to-end throughput — отдельная метрика; загрузка модели исключена.
- Полного 12-task GPU quality-run в этом приложении нет. Бесплатность и доступность T4 в будущих сессиях не гарантируются.

Проверка без модели и сети: `python scripts/audit_speed_probe.py`.

Доказательства сохранены в `evidence/performance/`: оригинальный маленький Colab ZIP, raw records, manifests, native model-load log, исходники probe и независимая проверка. Веса, wheel_check, .so, .venv и CUDA wheels не включены. Исходники измерений сохранены для аудита; повторный inference потребует отдельно подготовить модели и подходящую вычислительную среду.
