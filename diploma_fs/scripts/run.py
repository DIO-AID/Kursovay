"""Запуск экспериментов.

Примеры:
  python scripts/run.py --list
  python scripts/run.py --method base_raw base_fe_all shape_fit
  python scripts/run.py --method all --datasets synth_indep diabetes
  python scripts/run.py --method all --dataset steel tetouan          # из fsx/registry.py
  python scripts/run.py --method all --dataset all --select-sample 3000
  python scripts/run.py --method all --csv data/my.csv --target y --time-col date --sample 10000

--select-sample N: методам отбора дать случайные N строк train (модели учатся на всём train).
Нужно для медленных sysoev_fixed / boruta / rfe на больших данных; значение пишется в JSON.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

for _s in (sys.stdout, sys.stderr):          # Windows-консоль в cp1251
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
warnings.filterwarnings("ignore")             # ConvergenceWarning MLP и т.п. засоряли вывод

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsx.data import DATASETS, DataError, csv_dataset, registry_dataset  # noqa: E402
from fsx.evaluate import run_method                 # noqa: E402
from fsx.registry import DATASETS as REG            # noqa: E402
from fsx.selectors import REGISTRY                  # noqa: E402

p = argparse.ArgumentParser(description="Отбор признаков с учётом модификаций")
p.add_argument("--list", action="store_true", help="показать методы и датасеты")
p.add_argument("--method", nargs="+", help="методы отбора или 'all'")
p.add_argument("--datasets", nargs="+", help="встроенные датасеты (синтетика, diabetes)")
p.add_argument("--dataset", nargs="+", help="реальные датасеты из fsx/registry.py или 'all'")
p.add_argument("--csv", help="путь к своему CSV")
p.add_argument("--target", help="целевой столбец CSV")
p.add_argument("--time-col", help="столбец времени: разбиение 'прошлое -> будущее'")
p.add_argument("--sample", type=int, help="взять N строк (для времени — последние N)")
p.add_argument("--name", help="имя датасета в результатах (по умолчанию имя файла)")
p.add_argument("--select-sample", type=int, help="подвыборка train для методов отбора")
a = p.parse_args()

if a.list or not a.method:
    print("Методы:            ", ", ".join(sorted(REGISTRY)))
    print("Встроенные (--datasets):", ", ".join(DATASETS))
    print("Реестр (--dataset):")
    for k, r in REG.items():
        print(f"  {k:14s} {r['role']:8s} {r['status']:9s} data/{r['file']}  цель: {r['target']}")
    sys.exit(0)

methods = sorted(REGISTRY) if a.method == ["all"] else a.method
unknown = [m for m in methods if m not in REGISTRY]
if unknown:
    sys.exit(f"Неизвестные методы: {unknown}. Доступны: {sorted(REGISTRY)}")

if a.csv:
    if not a.target:
        sys.exit("Для --csv нужен --target (целевой столбец)")
    name = a.name or Path(a.csv).stem
    datasets = {name: lambda: csv_dataset(a.csv, a.target, a.time_col, a.sample, name=name)}
elif a.dataset:
    names = list(REG) if a.dataset == ["all"] else a.dataset
    missing = [d for d in names if d not in REG]
    if missing:
        sys.exit(f"Нет в fsx/registry.py: {missing}. Есть: {list(REG)}")
    datasets = {d: (lambda d=d: registry_dataset(d)) for d in names}
else:
    names = a.datasets or list(DATASETS)
    missing = [d for d in names if d not in DATASETS]
    if missing:
        sys.exit(f"Неизвестные датасеты: {missing}. Доступны: {list(DATASETS)}")
    datasets = {d: DATASETS[d] for d in names}

total = len(methods) * len(datasets)
k = 0
for d, loader in datasets.items():
    try:
        loader()                                   # проверка данных до долгого прогона
    except (DataError, FileNotFoundError) as e:
        print(f"!! {d}: {e}", flush=True)
        continue
    for m in methods:
        k += 1
        t = time.time()

        def prog(i, n, m=m, d=d):
            print(f"\r[{k}/{total}] {m:18s} {d:14s} фолд {i}/{n}", end="", flush=True)

        r2, nf = run_method(d, loader, m, REGISTRY[m], progress=prog, select_sample=a.select_sample)
        print(f"\r[{k}/{total}] {m:18s} {d:14s} призн.={nf:5.1f} " +
              " ".join(f"{kk}={v:.4f}" for kk, v in r2.items()) + f"  [{time.time() - t:.0f}с]",
              flush=True)
