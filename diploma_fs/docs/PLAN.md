# План работы

**Тема.** Автоматический выбор модификаций признаков (raw, log, sqrt, sq, inv) для
интерпретируемых моделей и сравнение с бустингом.

**Цель.** Диплом и, возможно, совместная статья с А. Сысоевым (SUMMA). Не продукт.

**Честная рамка.** Инвариантность деревьев к монотонным преобразованиям — известный факт,
подаём со ссылкой. Метод «форма зависимости» — сочетание известных идей (ACE, Бокс–Кокс,
GAM/EBM); наш вклад — аккуратная постановка, воспроизводимое сравнение, разбор алгоритма
из статьи и проверка на реальных данных.

## Этапы

| Этап | Что | Статус |
|---|---|---|
| 1 | Конвейер, синтетика + diabetes, отчёт, документы | цикл 1 сделан; `research/main` собран |
| 2 | Реальные данные, Optuna, приложение | ветки `research/real-data`, `research/optuna`, `research/app` |
| 3 | Сильные базовые линии и развитие метода | `research/baselines`, `research/catboost-shap`, `research/shape-fit` |
| 4 | АКИ Сысоева, теория, Docker | `research/aff-sysoev`, `docs/theory`; Docker-образ |

## Ветки (все от `research/main`, в каждой `TASK.md`)

| Ветка | Цель |
|---|---|
| `research/real-data` | `scripts/download_data.py` (ucimlrepo) → `data/*.csv`; 10 датасетов: California Housing, Concrete, Airfoil, Wine Quality, Abalone; Combined Cycle Power Plant, Gas Turbine CO/NOx, Steel Industry Energy, Naval Propulsion, Superconductivity. Выборка 10 000 строк для больших; разбиение по времени для Power Plant, Gas Turbine, Steel. |
| `research/optuna` | default / Grid / Random (обязательный контроль) / Optuna при равном бюджете; подбор только внутри train; повтор по сидам; CatBoost и MLP; 3–4 датасета; всё в `results/optuna/studies.db` (study `<датасет>__<модель>__<метод>__seed<N>`); кривые сходимости. |
| `research/app` | Streamlit из 2 страниц: запуск (встроенный датасет или CSV, выбор цели/времени, методы, прогресс) и открытие `studies.db` в Optuna Dashboard. Без своих графиков, LLM, стекинга. |
| `research/baselines` | Бокс–Кокс / Йео–Джонсон по фактору и ACE как базовые линии выбора формы. |
| `research/catboost-shap` | Перенос на CatBoost: важность CatBoost, `select_features`, SHAP dependence вместо PD. |
| `research/shape-fit` | Развитие метода: проверка аддитивности, критерий запаса (BIC / бутстрэп выбора формы). |
| `research/aff-sysoev` | Анализ конечных изменений (АКИ) Сысоева как мера важности групп; сравнение с групповой важностью и формой. |
| `docs/theory` | Теоретическая глава: инвариантность деревьев (со ссылками), постановка, обзор ACE/Бокс–Кокс/GAM/EBM. |

**Порядок:** real-data, optuna, app → baselines, catboost-shap, shape-fit → aff-sysoev, theory.

**Не делаем:** комбинации признаков (x1·x2 и т.п.) — пробовали, взрыв числа признаков без результата.

## Цикл по ветке

1. Claude пишет код в ветку (`TASK.md` — цель, что меняется, критерий успеха).
2. Прогон на своём компьютере: `scripts/run.py` → `scripts/report.py`.
3. Вывод записывается в `docs/DECISIONS.md` (по заранее заданному критерию).
4. Pull Request в `research/main`. Ветку `main` и папку `app/` курсовой не трогаем.
