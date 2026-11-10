# Источники, лицензии и атрибуция

## SpreadsheetBench

- Авторы: Zeyao Ma, Bohan Zhang, Jing Zhang, Jifan Yu, Xiaokang Zhang, Xiaohan Zhang, Sijia Luo, Xi Wang, Jie Tang.
- Статья: *SpreadsheetBench: Towards Challenging Real World Spreadsheet Manipulation*, NeurIPS Datasets and Benchmarks 2024; https://arxiv.org/abs/2406.14991 .
- Репозиторий: https://github.com/RUCKBReasoning/SpreadsheetBench .
- Зафиксированный commit: `49b73a94775fb489063f60ca1865e3a650079a79`.
- README с заявлением лицензии: https://github.com/RUCKBReasoning/SpreadsheetBench/blob/49b73a94775fb489063f60ca1865e3a650079a79/README.md .
- Лицензия, указанная авторами проекта: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**, https://creativecommons.org/licenses/by-sa/4.0/ ; полный текст https://creativecommons.org/licenses/by-sa/4.0/legalcode .
- Официальный comparator: https://github.com/RUCKBReasoning/SpreadsheetBench/blob/49b73a94775fb489063f60ca1865e3a650079a79/evaluation/evaluation.py .

Адаптер следует правилам нормализации значений из этого comparator. Его собственные изменения описаны в README: явные predictions, реальные случаи, answer_sheet, диагностика, консервативные формулы и ограничения файлов. Это локальная адаптация, не официальный релиз и не одобрение авторов benchmark.

Локальный набор сохранён отдельно от адаптера; копии датасета и оригинального evaluator в этот каталог не включены. При распространении данных и адаптаций сохраняйте атрибуцию, ссылку на лицензию, отметку изменений и соблюдайте ShareAlike в применимой области. CC BY-SA допускает коммерческое использование при соблюдении условий; это не универсальное разрешение на все права исходных документов.

Не назначаем пользователю новую лицензию на его независимые компоненты автоматически. До публичного выпуска собственного кода следует определить применимость ShareAlike к заимствованным частям и выбрать совместимые условия распространения. Это уведомление не заменяет юридическую проверку.

## openpyxl

Используется установленный пакет openpyxl; в адаптер не включён код зависимости. Проект openpyxl распространяется по MIT License. Источник: https://openpyxl.readthedocs.io/ ; сведения об установленной версии доступны через `python -c "import openpyxl; print(openpyxl.__version__)"`.
