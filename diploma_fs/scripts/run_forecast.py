"""Прогон экспериментов E1–E4 (ADR-001): временные признаки + отбор для CatBoost.

Примеры:
  python scripts/run_forecast.py --list
  python scripts/run_forecast.py --dataset synth_factory --quick          # проверка за пару минут
  python scripts/run_forecast.py --dataset steel tetouan                  # основной прогон
  python scripts/run_forecast.py --dataset steel --mode virtual           # режим «виртуальный датчик»
  python scripts/run_forecast.py --dataset steel --tune 30                # + настройка CatBoost (Optuna)
  python scripts/run_forecast.py --dataset steel --select cb_shap sysoev_fixed --no-sets

Потом: python scripts/report_forecast.py  -> docs/FORECAST_REPORT.md

--quick: 5 блоков вместо 10, отбор на последних 2000 строках train, только 3 быстрых метода
         отбора — чтобы проверить, что всё работает, а не для диплома.
"""
import argparse
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsx.data import TIME_DATASETS, DataError, registry_dataset  # noqa: E402
from fsx.forecast import DEFAULT_SELECTORS, N_BLOCKS, run_forecast  # noqa: E402
from fsx.models import has_catboost, has_optuna      # noqa: E402
from fsx.registry import DATASETS as REG             # noqa: E402

TIME_REG = [k for k, r in REG.items() if r.get("horizon")]

p = argparse.ArgumentParser(description="Прогноз энергопотребления: E1–E4")
p.add_argument("--list", action="store_true")
p.add_argument("--dataset", nargs="+", help="synth_factory и/или из реестра: " + ", ".join(TIME_REG))
p.add_argument("--mode", choices=["forecast", "virtual"], default="forecast")
p.add_argument("--horizon", help="переопределить горизонт, например 1h, 1D")
p.add_argument("--select", nargs="+", help="методы отбора (по умолчанию все из DEFAULT_SELECTORS)")
p.add_argument("--no-sets", action="store_true", help="не считать наборы E1/E2 (только отбор)")
p.add_argument("--select-sample", type=int, default=5000, help="строк train для отбора (последние)")
p.add_argument("--tune", type=int, default=0, help="попыток Optuna для CatBoost (0 — без настройки)")
p.add_argument("--blocks", type=int, default=N_BLOCKS)
p.add_argument("--quick", action="store_true")
a = p.parse_args()

print(f"CatBoost: {'есть' if has_catboost() else 'НЕТ -> вместо него HistGradientBoosting (hgb_fallback)'};"
      f" Optuna: {'есть' if has_optuna() else 'нет'}", flush=True)
if a.list or not a.dataset:
    print("Временные датасеты:", ", ".join(list(TIME_DATASETS) + TIME_REG))
    print("Методы отбора по умолчанию:", ", ".join(DEFAULT_SELECTORS))
    sys.exit(0)

selectors = a.select or (("cb_importance", "cb_shap", "sysoev_fixed") if a.quick else DEFAULT_SELECTORS)
blocks = 5 if a.quick else a.blocks
sel_sample = 2000 if a.quick else a.select_sample

for d in a.dataset:
    try:
        ds = TIME_DATASETS[d]() if d in TIME_DATASETS else registry_dataset(d)
    except (DataError, FileNotFoundError, KeyError) as e:
        print(f"!! {d}: {e}", flush=True)
        continue
    t0 = time.time()

    def prog(i, n, what, d=d):
        print(f"\r{d:14s} {i:4d}/{n} {what:22s} [{time.time() - t0:5.0f}с]", end="", flush=True)

    summ, info = run_forecast(d, ds, a.mode, a.horizon, selectors, not a.no_sets, sel_sample,
                              a.tune, blocks, prog)
    print(f"\n{d}: шаг {info['step']}, горизонт {info['horizon_steps']} шаг., "
          f"разогрев {info['warmup_steps']} строк, {time.time() - t0:.0f} с")
    print(f"  {'метод':26s} " + "  ".join(f"MAE {m:10s}" for m in ("naive_last", "naive_day", "ridge", "catboost")))
    for m, v in summ.items():
        print(f"  {m:26s} " + "  ".join(f"{v[k]:14.3f}" if k in v else " " * 14
                                        for k in ("naive_last", "naive_day", "ridge", "catboost")))
