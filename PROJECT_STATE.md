# PROJECT_STATE.md — Состояние проекта между сессиями

## Текущий статус
**Дата обновления:** 2026-09-08
**Активная фаза:** Исправление критических нарушений (Data Leakage, архитектура)
**Следующий шаг:** Замена print() на logging, проверка импортов в src/model.py

## Выполненные решения
1. Принята архитектура "микросервисы для ML" — каждый модуль решает одну задачу
2. Определена структура репозитория с папками `data/`, `models/`, `reports/`, `src/`
3. Решено использовать parquet для обмена данными между модулями
4. Модули НЕ импортируют друг друга напрямую — только через артефакты
5. Единый `evaluation/cv_runner.py` вместо дублирования функций
6. Все модели должны обучаться через `Pipeline` для честной CV
7. Финальная модель обучается на train+val (80% данных)
8. Feature Engineering — sklearn Transformer (fit только на train, нет leakage)
9. `experiments/run_experiment.py` — feature engineering ПОСЛЕ train_test_split

## В процессе
- Замена print() на logging (частично выполнена)
- Разделение прямых импортов (src/model.py → tuning.search_spaces)

## Отложенные решения
- Stacking — добавить только если даст >2% прироста R²
- Генерация 173 признаков — если SHAP всё равно режет до 8, делать сразу 15-30 осмысленных

## Структура модулей
```
src/
├── common/
│   ├── config.py           ✅ создано
│   ├── artifacts.py        ✅ создано
│   ├── schemas.py          ✅ создано
│   └── logging.py          ✅ создано
├── evaluation/
│   ├── cv_runner.py        ✅ создано (единая CV-оценка)
│   └── metrics.py
├── feature_engineering.py  ✅ переписан (AdvancedFeatureTransformer)
├── preprocessing.py
├── model.py
├── data_loader.py
├── results_db.py
└── ...
```

## План задач
1. [x] Создать `src/common/config.py` с путями и константами
2. [x] Создать `src/common/artifacts.py` с save/load функциями
3. [x] Создать `src/common/schemas.py` с Pydantic-схемами
4. [x] Создать `src/common/logging.py` с единым логгером
5. [x] Создать `src/common/cv_runner.py` — универсальная CV-обёртка
6. [x] Исправить Data Leakage в `feature_engineering.py`
7. [x] Исправить `experiments/run_experiment.py` — FE после сплита
8. [ ] Заменить print() на logging в `src/evaluation.py`, `src/model.py`, `experiments/feature_pipeline.py`
9. [ ] Разделить прямые импорты (src/model.py → tuning.search_spaces)
10. [ ] Создать `src/07_final_evaluator.py`
11. [ ] Создать `orchestrator.py`

## Известные проблемы
- `src/model.py` импортирует из `tuning.search_spaces` — прямой импорт между модулями
- `src/evaluation.py` содержит print() вместо logging
- `n_jobs=1` захардкожен в evaluation.py

## Логи последних изменений
- 2026-09-08: Исправлен Data Leakage (feature_engineering.py переписан на Transformer)
- 2026-09-08: Создан src/common/ с конфигами, артефактами, схемами, логированием
- 2026-09-08: Создан src/common/cv_runner.py
- 2026-09-08: Исправлен experiments/run_experiment.py
- 2026-09-08: Создан PROJECT_STATE.md
