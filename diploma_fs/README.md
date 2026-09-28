# diploma_fs — выбор модификаций признаков

Стенд к диплому: как автоматически выбрать модификацию каждого фактора
(raw, log, sqrt, sq, inv) для интерпретируемой модели и чем это отличается от бустинга.

| Документ | Зачем |
|---|---|
| [docs/МЕТОДИЧКА.md](docs/МЕТОДИЧКА.md) | простым языком: выводы, методы, как читать отчёт, словарик |
| [docs/REPORT.md](docs/REPORT.md) | таблицы и графики (создаётся `scripts/report.py`) |
| [docs/PIPELINE.md](docs/PIPELINE.md) | конвейер из 7 шагов, правила честности, формат результатов |
| [docs/DECISIONS.md](docs/DECISIONS.md) | критерий принятия и решения по каждому методу |
| [docs/PLAN.md](docs/PLAN.md) | этапы и ветки |

## Установка (один раз)

Из папки `diploma_fs`:

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1          # ядро
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Extra   # + catboost, shap, optuna, streamlit
.venv\Scripts\Activate.ps1
```

Ядро (`requirements.txt`) и тяжёлые пакеты веток (`requirements-extra.txt`) ставятся раздельно:
если CatBoost или SHAP не собираются под вашу версию Python, ядро всё равно работает.
Если установка раньше падала — удалите `requirements.lock.txt` и `.venv` и запустите заново.

```bash
# Linux / macOS
bash scripts/setup.sh            # или: bash scripts/setup.sh --extra
source .venv/bin/activate
```

Проверка: `python scripts/check_env.py`.

## Запуск

```bash
python scripts/run.py --list                               # методы и датасеты
python scripts/run.py --method all                         # все методы × встроенные датасеты
python scripts/run.py --method base_fe_all shape_fit --datasets synth_indep
python scripts/theory_check.py                             # проверка H0 -> results/theory_check.json
python scripts/report.py                                   # docs/REPORT.md, results/summary.csv, results/figures/*.svg
```

Реальные датасеты — через реестр `fsx/registry.py` (цель, время, утечки заданы там):

```bash
python scripts/download_data.py                 # скачать все из реестра в data/
python scripts/download_data.py --check         # проверить скачанные файлы
python scripts/run.py --list                    # что есть в реестре
python scripts/run.py --method all --dataset steel tetouan --select-sample 3000
```

Свой CSV (разово, без реестра):

```bash
python scripts/run.py --method all --csv data/concrete.csv --target strength
python scripts/run.py --method all --csv data/my_series.csv --target y --time-col date --sample 10000
```

CSV кладите в `diploma_fs/data/` (или в `data/` в корне репозитория; `housing.csv` оттуда
подключается как встроенный датасет автоматически).
`--time-col` — строки сортируются по времени, обучение только на прошлом;
`--sample N` — N строк (случайные; для временных — последние N подряд); дата в `--csv` должна быть в формате ISO (ГГГГ-ММ-ДД), другие форматы — через реестр (`time_format`); `--name` — имя в результатах.

Полный прогон `--method all` на трёх встроенных датасетах занимает порядка 10–20 минут
на слабом ноутбуке (дольше всех Boruta и RFE).

## Структура

```
diploma_fs/
  fsx/            код конвейера: data, registry (реестр датасетов), transforms, evaluate, selectors/
  scripts/        run.py, download_data.py, theory_check.py, report.py, check_env.py, setup.ps1/.sh
  results/        raw/ (не в git), summary.csv, ranking.csv, theory_check.json, figures/*.svg
  docs/           отчёт и документы
  data/           CSV (не в git)
```

Новый метод отбора = новый файл в `fsx/selectors/` с `METHODS = {"имя": функция}`;
он появится в `run.py --list` автоматически.
