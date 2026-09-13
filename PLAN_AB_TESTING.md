# PLAN_AB_TESTING.md — План A/B-тестирования 5 методов

> Научно-строгое сравнение: baseline vs 5 экспериментов, «одно изменение на ветку».
> Каждый результат — подтверждённый (5-fold CV + 3 сида), не мнимо-случайный.
> Отрицательные результаты — тоже результаты.

---

## 📐 Базовое правило

| Что | Правило |
|---|---|
| Сплит | train/val/test = 60/20/20, `random_state=42` (тот же, что в `Final/main_fair_comparison.py`) |
| Preprocessing | Сначала split, потом fit на train, transform на test (никакого leakage) |
| Гиперпараметры | Зафиксированы 1 раз на baseline, НЕ меняются в ветках |
| Один тест | В конце каждой ветки predict на ТОМ ЖЕ `X_test` — только 1 раз |
| Изменение | Ровно ОДНО на ветку |
| Повторы | Каждый эксперимент 3 сида (42/123/456), усредняем метрики |
| Метрики | R2, MAE, RMSE + 5-fold CV (mean±std) |

## Критерий значимости прироста
`exp_mean > baseline_mean + 2 * baseline_std` → улучшение считается значимым.
Дополнительно сравниваем 3-сидовые прогоны: если нет прироста в ≥2 сидах — отклоняем.

---

## 🌿 Структура веток

```
main (baseline) ──┬── exp1-huber-loss
                  ├── exp2-target-encoding
                  ├── exp3-residual-stacking
                  ├── exp4-cross-features  (ожидание: НЕ сработает — подтверждение гипотезы)
                  └── exp5-tabm-meta       (⚠️ под вопросом — см. Шаг 5)
```

Каждая ветка создаётся от `main`, живёт отдельно, мержится только при подтверждении.

---

## 📊 Таблица экспериментов

| Ветка | Что меняем | Ожидание | Критерий |
|---|---|---|---|
| `main` | — | R2 = baseline | фиксируем в `baseline_metrics.json` |
| `exp1-huber-loss` | `objective='reg:pseudohubererror'` | устойчивость к выбросам | R2 > baseline + 0.02 |
| `exp2-target-encoding` | SmoothedTargetEncoder (OOF) вместо OneHot для `ocean_proximity` | лучшее кодирование категорий | R2 > baseline + 0.03 |
| `exp3-residual-stacking` | LGBM → XGB на остатках → сумма | модели дополняют друг друга | R2 > baseline + 0.02 |
| `exp4-cross-features` | mul/div/sub топ-5 признаков + отбор | ❌ НЕ даст прироста (<0.01) | подтверждение отрицательной гипотезы |
| `exp5-nn-meta` | **MLPRegressor (sklearn)** prediction как доп. признак. ⚠️ Настоящий TabM/torch — ТОЛЬКО если MLP даст значимый прирост | нелинейные паттерны | R2 > baseline + 0.03 |

> **Решение по TabM:** torch НЕ установлен; на Intel N95 это медленно и тяжело.
> Сначала дешёвая проверка гипотезы через sklearn `MLPRegressor` (та же идея «НС → мета-признак»).
> Если значимый прирост на обеих baseline-моделях — ставим torch и делаем настоящий TabM.
> Если нет — фиксируем отрицательный результат, torch не ставим.

---

## 🪜 Шаги

### Шаг 0 — подготовка (main)
1. Создать `experiments/ab/harness_ab.py` — общий харнес:
   - загрузка `data/housing.csv`, сплит 60/20/20 (`random_state=42`)
   - `Preprocessor` (train fit → test transform)
   - единая функция метрик: `R2, MAE, RMSE`
   - `evaluate_ab(model, X, y, cv=5)` → 5-fold CV mean±std
   - сохранение результата в `Final/results_ab/<name>.json`
2. Зафиксировать **ДВЕ baseline-модели** — LightGBM и XGBoost (лучшие параметры Optuna). Их гиперпараметры замораживаются на весь A/B-цикл.
3. Прогнать по 3 сида на каждую → записать `baseline_metrics.json` (по модели).
4. Коммит: `exp: baseline A/B harness + метрики`.

### Шаг 1 — ветка `exp1-huber-loss`
- Создать ветку от `main`.
- Скрипт `experiments/ab/exp1_huber.py`: та же harness, но XGBoost/модель с `objective='reg:pseudohubererror'` (+`huber_slope=1.0`).
- Всё остальное — как в baseline. Прогнать 3 сида, записать JSON, сравнить.
- Вывод в отчёте: да/нет прирост, гипотеза подтверждена/опровергнута.
- Коммит: `exp: Exp1 Huber Loss ...` → ветка → push. Если прирост значим → мердж в main и запись в `DECISIONS_LOG.md`.

### Шаг 2 — ветка `exp2-target-encoding`
- Создать ветку от `main`.
- Заменить OneHot на SmoothedTargetEncoder с OOF (fit только на train, transform на test — без leakage).
- Один категориальный признак (`ocean_proximity`, 5 категорий). Числовые — не трогаем.
- Коммит: `exp: Exp2 Target Encoding ...`

### Шаг 3 — ветка `exp3-residual-stacking`
- Создать ветку от `main`.
- `ResidualStacking`:
  1. LGBM обучается на train → `pred_lgbm`
  2. остатки = y_train − pred_lgbm
  3. XGB обучается на остатках
  4. финал: `pred = pred_lgbm + pred_xgb_residual`
- Оба обучения на train, оценка на ТОМ ЖЕ test (без refit на test).
- Коммит: `exp: Exp3 Residual Stacking ...`

### Шаг 4 — ветка `exp4-cross-features`
- Создать ветку от `main`.
- Сгенерировать mul/div/sub для топ-5 наиболее важных признаков.
- Отбор (mRMR/correlation) — вернуть к рабочему набору.
- Прогнать, записать R2. **Ожидаем R2 ≈ baseline.** Записываем это как научный результат.
- Коммит: `exp: Exp4 Cross-Features (отрицательный результат) ...` → НЕ мержим (или мержим как документированное подтверждение гипотезы).

### Шаг 5 — ветка `exp5-nn-meta`
- Создать ветку от `main`.
- **MVP (этап 1 — sklearn MLP):**
  - Обучить `MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300)` на `X_train`.
  - Получить `pred_mlp_train/pred_mlp_test` → добавить как колонку `<nn_meta_pred>`.
  - Обучить baseline LGBM/XGB на `X + nn_meta_pred`.
  - Сравнить с baseline по R2 (5-fold CV + 3 сида).
- **Если значимый прирост на обеих моделях → этап 2:**
  - `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - Реализовать TabM: mini-batch norm → MLP head → мета-фича.
  - Сравнить с MLP surrogate.
- **Если нет значимого прироста → фиксируем отрицательный результат**, torch не ставим.

### Шаг 6 — финальный комбинированный эксперимент (main)
- Только подтверждённые методы объединяем в ONE pipeline.
- Прогон 3 сидов, сравнение с baseline.
- Собираем итоговую таблицу + `DECISIONS_LOG.md` + графики для диплома.

---

## 📁 Где что лежит

| Артефакт | Путь |
|---|---|
| Харнес | `experiments/ab/harness_ab.py` |
| Результаты | `Final/results_ab/*.json` (коммитятся, csv нет) |
| Baseline | `Final/results_ab/baseline_metrics.json` |
| Журнал решений | `DECISIONS_LOG.md` |
| Черновые данные | `data/housing.csv` (в git НЕ идёт) |

## 🚫 Чего НЕ делаем
- Не меняем гиперпараметры между ветками.
- Не сравниваем с собой: каждая ветка = только одно изменение.
- Не оцениваем модель на тестовой выборке больше одного раза за прогон.
- Не коммитим csv/db/models.