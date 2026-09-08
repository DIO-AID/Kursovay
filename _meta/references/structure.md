# Структура проекта

## Директории
```
_lessons/           # База знаний: уроки, найденные проблемы
_meta/              # Служебная папка AI-агента
app/                # Streamlit UI + пайплайн
├── analysis/       # 6 модулей аналитики (overview, model, errors, optuna, shap, results)
├── ui.py           # Главный UI
├── pipeline_adapter.py  # Оркестрация обучения
├── data_handler.py # Загрузка данных
├── main.py         # Точка входа
└── config.py       # (пустой)
src/                # Core ML модули
├── models_library.py    # 11 моделей
├── feature_engineering.py  # Инженерия признаков
├── feature_selection.py    # Отбор
├── stacking.py      # Стекинг
├── evaluation.py    # Оценка
├── shap_analysis.py # SHAP
├── results_db.py    # SQLite БД
├── safe_fit.py      # Безопасное обучение
├── preprocessing.py # Предобработка
├── gpu_utils.py     # GPU/CPU
└── visualization.py # Простые графики
tuning/             # Optuna
├── search_spaces.py     # Пространства гиперпараметров
└── run_optuna_model.py  # Запуск Optuna
experiments/        # Batch эксперименты
orchestration/      # Оркестрация запусков
configs/            # Конфиги
utils/              # Утилиты
Final/              # Честное сравнение Grid vs Optuna
└── results_fine/   # Финальные результаты
data/               # Давайте (housing.csv, zillow и др.)
статья/             # Статья Сысоева по SHAP-отбору
tuning/             # SQLite БД с Optuna trials
```

## Ключевые файлы-документы
- `ТЗ.txt` — техническое задание
- `НовоеТЗ.txt` — доп. задание
- `задание.txt` — задание агенту
- `промт и структура.txt` — 6 промтов для развития
- `развитие.txt` — компромиссы улучшений
- `ошибки.txt` — лог ошибок
- `ПАРАМЕТРЫ_МОДЕЛЕЙ.md` — справочник гиперпараметров
- `prompt-repos-questionnaire.md` — оценка репозиториев
