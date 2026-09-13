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
| `exp5-tabm-meta` | TabM prediction как доп. признак | нелинейные паттерны | R2 > baseline + 0.03 |

---

## 🪜 Шаги

### Шаг 0 — подготовка (main)
1. Создать `experiments/ab/harness_ab.py` — общий харнес:
   - загрузка `data/housing.csv`, сплит 60/20/20 (`random_state=42`)
   - `Preprocessor` (train fit → test transform)
   - единая функция метрик: `R2, MAE, RMSE`
   - `evaluate_ab(model, X, y, cv=5)` → 5-fold CV mean±std
   - сохранение результата в `Final/results_ab/<name>.json`
2. Зафиксировать baseline-модель (LightGBM, лучшие параметры Optuna) и гиперпараметры — они больше НЕ меняются.
3. Прогнать 3 сида → записать `baseline_metrics.json`.
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

### Шаг 5 — ветка `exp5-tabm-meta` — ⚠️ БЛОКЕР
- **Проблема:** torch НЕ установлен, а TabM — нейросеть. Установка torch на Intel N95 (CPU) — тяжело и медленно.
- **Варианты (нужно решение):**
  1. Установить torch CPU и обучить простой TabM (MLP-блок, ~несколько тыс. параметров).
  2. Заменить TabM на sklearn-замену: `MLPRegressor` (torch не нужен) или `HistGradientBoostingRegressor` как мета-модель.
  3. Убрать Exp5 из плана — оставить 4 эксперимента + честный вывод.
- Мета-подход не меняется: фича `<meta_pred>` добавляется к 10 признакам, обучаем baseline-модель.

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