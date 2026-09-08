# Уроки и опыт проекта ML Pipeline

## 1. Grid Search vs Optuna — честное сравнение

### Проблема
При первом сравнении Grid Search и Optuna Optuna стабильно находила лучшие результаты, чем полный перебор. Это было невозможно — если перебрать ВСЕ комбинации, лучше найти нельзя.

### Причина
Optuna использовала `suggest_int`/`suggest_float` — непрерывные диапазоны, что позволяло ей попадать на значения, которых НЕТ в Grid Search. Grid Search имел дискретные значения (например, `learning_rate: [0.05, 0.1, 0.15]`), а Optuna могла выбрать `0.073`.

**Это нечестное сравнение.** Сравнивали яблоки с апельсинами.

### Решение
Для честного сравнения Optuna должна использовать ТОЛЬКО `suggest_categorical` с теми же значениями, что и Grid:
```python
# Grid и Optuna — ОДИНАКОВЫЕ пространства
params = {
    "learning_rate": trial.suggest_categorical("learning_rate", [0.05, 0.1, 0.15]),
    "max_depth": trial.suggest_categorical("max_depth", [4, 8, 12]),
    # ... все параметры через categorical
}
```

### Результат честного сравнения
- **XGBoost:** Grid=0.8452, Optuna=0.8452 — **одинаково**
- **LightGBM:** Grid=0.8411, Optuna=0.8392 — **Grid чуть лучше**
- **CatBoost:** Grid=0.8358, Optuna=0.8358 — **одинаково**

**Вывод:** Optuna НЕ находит лучше полного перебора при одинаковом пространстве. Её преимущество — в скорости (100 trials vs 1000+ комбинаций).

---

## 2. CatBoost — iterations это bottleneck

### Проблема
CatBoost с варьируемым `iterations` работал в 76x медленнее XGBoost/LightGBM при одинаковом числе комбинаций. Каждая комбинация с iterations=500 обучалась ~40 секунд.

### Причина
`iterations` — это количество деревьев в CatBoost. Чем больше, тем дольше обучение. При grid из 77 значений iterations (50..500) каждый trial был очень медленным.

### Решение
- Для Grid: зафиксировать `iterations = 200` (достаточно для сходимости)
- Для Optuna: тоже зафиксировать, если пространство должно быть одинаковым
- Или оставить варьируемым только в Optuna (но тогда это уже нечестное сравнение)

---

## 3. Чекпоинты обязательны для длительных запусков

### Проблема
Grid Search на 138K комбинаций стабильно падал или превышал таймауты. Без чекпоинтов весь прогресс терялся.

### Решение
Сохранять JSON-чекпоинт каждые 50 комбинаций:
```python
if (i + 1) % 50 == 0:
    with open(checkpoint_path, 'w') as f:
        json.dump({"last_idx": i+1, "all_r2": all_r2, "best_r2": best_r2, ...}, f)
```

При перезапуске скрипт загружает чекпоинт и продолжает с `last_idx`.

---

## 4. Баг: переменная `optuna` конфликтует с модулем

### Проблема
```python
import optuna
...
optuna = time.time() - start  # ПЕРЕЗАПИСАЛИ МОДУЛЬ!
print(f"Time: {optuna:.1f}s")  # TypeError: module.__format__
```

### Решение
Не использовать `optuna` как имя переменной. Использовать `optuna_time`.

---

## 5. Баг: trailing comma в return dict

### Проблема
В `app/pipeline_adapter.py` строка 246:
```python
return {
    ...
    "X_test_processed": X_test_proc,
    
    "selected_features": final_features
}
```
Пропущен `return` для режима `base` (не stack, не optuna) — функция возвращает `None`.

### Решение
Добавить fallback return для режима `base`.

---

## 6. Hardcoded пути в optuna_analysis.py

### Проблема
```python
db_path = os.path.join(os.getcwd(), "tuning/optuna.db")
```
Хардкод пути к tuning/optuna.db, которая была удалена. Нет fallback на Final/results_fine/optuna_fine.db.

### Решение
Искать БД в нескольких местах: сначала `Final/results_fine/`, потом `tuning/`, потом корень.

---

## 7. Структура проекта

### Что оставить
- `app/` — Streamlit UI + анализ
- `src/` — core модули (data_loader, model, preprocessing, stacking, etc.)
- `tuning/` — Optuna код (search_spaces, run_optuna_model, metrics)
- `configs/` — конфигурации
- `experiments/` — feature pipeline
- `orchestration/` — runner, leaderboard
- `utils/` — утилиты
- `Final/main_fair_comparison.py` — главный скрипт сравнения
- `Final/results_fine/` — результаты fair comparison

### Что удалить
- Старые `main_*.py` скрипты (1d, fine_grid, full_comparison, limited_resume)
- Старые `.ipynb` ноутбуки
- Старые `.db` файлы кроме завершённых
- `курсовая старые наработки/` — старый курсовой проект
- `результат/` — старые результаты
- `test/` — тестовые запуски
- Временные `check_*.py` файлы

---

## 8. Рекомендации на будущее

1. **ВСЕГДА** использовать `suggest_categorical` для честного сравнения с Grid
2. **ВСЕГДА** сохранять чекпоинты для длительных экспериментов
3. **НЕ** использовать имена модулей как переменные (`optuna`, `json`, `time`)
4. **НЕ** хардкодить пути — использовать `Path(__file__).parent`
5. Фиксировать `iterations` у CatBoost для ускорения Grid Search
6. Optuna — для быстрого поиска, Grid — для полного покрытия пространства
