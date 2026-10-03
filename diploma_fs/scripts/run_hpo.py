"""Прогон гипотез о настройке Optuna (ветки research/hpo-*).

  python scripts/run_hpo.py --list
  python scripts/run_hpo.py --experiment base                       # базовая настройка Optuna, 3 датасета
  python scripts/run_hpo.py --experiment h_trials                   # гипотеза «число попыток»
  python scripts/run_hpo.py --experiment h_early_stopping           # гипотеза «ранняя остановка»
  python scripts/run_hpo.py --experiment base --quick               # проверка за несколько минут
  python scripts/run_hpo.py --experiment h_trials --dataset tetouan --models catboost --folds 3
  python scripts/run_hpo.py --experiment base --val-scheme time --name base_time   # контроль: проверка по времени

Описания экспериментов — experiments/hpo/<имя>.json; всё, чего там нет, берётся из базовой настройки (fsx/hpo.BASE).
Результаты: results/hpo/<имя>/<датасет>/<модель>__<режим остановки>__fold<k>.json
Потом: python scripts/report_hpo.py --experiment <имя>

--quick: synth_factory + tetouan, 2 фолда, 8 попыток — только проверить, что всё работает.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsx.hpo import BASE, MODELS, _has, available_models, run_experiment  # noqa: E402

EXP = ROOT / "experiments" / "hpo"

p = argparse.ArgumentParser(description="Гипотезы о настройке Optuna")
p.add_argument("--list", action="store_true")
p.add_argument("--experiment", default="base", help="имя файла из experiments/hpo без .json")
p.add_argument("--name", help="имя папки результатов (по умолчанию = имя эксперимента)")
p.add_argument("--dataset", nargs="+")
p.add_argument("--models", nargs="+", choices=list(MODELS))
p.add_argument("--folds", type=int)
p.add_argument("--trials", type=int)
p.add_argument("--val-scheme", choices=["random", "time"])
p.add_argument("--seed", type=int)
p.add_argument("--threads", type=int, default=-1)
p.add_argument("--max-rows", type=int, help="взять только последние N строк (быстрая проверка, не для результатов)")
p.add_argument("--quick", action="store_true")
a = p.parse_args()

ok, missing = available_models(list(MODELS))
print(f"Модели: есть {ok}; нет пакета {missing or '—'}. "
      f"Optuna: {'есть' if _has('optuna') else 'НЕТ -> случайный поиск (random_fallback), результаты не для диплома'}",
      flush=True)
if a.list:
    print("Эксперименты:", ", ".join(sorted(f.stem for f in EXP.glob("*.json"))))
    print("Базовая настройка:", json.dumps(BASE, ensure_ascii=False))
    sys.exit(0)

f = EXP / f"{a.experiment}.json"
if not f.exists():
    sys.exit(f"Нет {f}. Есть: {sorted(x.stem for x in EXP.glob('*.json'))}")
cfg = {**json.loads(f.read_text(encoding='utf-8'))}
cfg.setdefault("name", a.experiment)
for key, val in (("datasets", a.dataset), ("models", a.models), ("folds", a.folds), ("val_scheme", a.val_scheme),
                 ("seed", a.seed), ("name", a.name)):
    if val is not None:
        cfg[key] = val
if a.max_rows:
    cfg["max_rows"] = a.max_rows
if a.trials:
    cfg["n_trials"] = a.trials
    cfg["budgets"] = [b for b in cfg.get("budgets", [a.trials]) if b <= a.trials] or [a.trials]
if a.quick:
    cfg.update(datasets=a.dataset or ["synth_factory", "tetouan"], folds=2, n_trials=8,
               budgets=[4, 8], max_rows=3000, name=cfg["name"] + "_quick")

t0 = time.time()


def prog(what, i, n):
    print(f"\r{what:44s} попытка {i:3d}/{n} [{time.time() - t0:6.0f}с]", end="", flush=True)


brief, missing = run_experiment(cfg, prog, a.threads)
print(f"\nГотово за {time.time() - t0:.0f} с. Пропущены (нет пакета): {missing or '—'}")
print(f"{'датасет':12s} {'модель':9s} {'остановка':9s} фолд  стандартная  настроенная  изменение   на проверке")
for r in brief:
    print(f"{r['dataset']:12s} {r['model']:9s} {r['es_mode']:9s} {r['fold']:4d}  {r['default']:11.3f}  "
          f"{r['tuned']:11.3f}  {100 * (r['tuned'] / r['default'] - 1):+8.1f}%  {r['val']:11.3f}")
print(f"Отчёт: python scripts/report_hpo.py --experiment {cfg['name']}")
