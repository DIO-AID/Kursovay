"""Запуск экспериментов.

Примеры:
  python scripts/run.py --list
  python scripts/run.py --method base_raw base_fe_all shape_fit
  python scripts/run.py --method all --datasets synth_indep diabetes
  python scripts/run.py --method all --csv data/concrete.csv --target strength
  python scripts/run.py --method all --csv data/steel.csv --target Usage_kWh --time-col date --sample 10000
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsx.data import DATASETS, csv_dataset          # noqa: E402
from fsx.evaluate import run_method                 # noqa: E402
from fsx.selectors import REGISTRY                  # noqa: E402

p = argparse.ArgumentParser(description="Отбор признаков с учётом модификаций")
p.add_argument("--list", action="store_true", help="показать доступные методы и датасеты")
p.add_argument("--method", nargs="+", help="методы отбора или 'all'")
p.add_argument("--datasets", nargs="+", help="встроенные датасеты (по умолчанию все)")
p.add_argument("--csv", help="путь к своему CSV")
p.add_argument("--target", help="целевой столбец CSV")
p.add_argument("--time-col", help="столбец времени: разбиение блоками без перемешивания")
p.add_argument("--sample", type=int, help="взять N случайных строк")
p.add_argument("--name", help="имя датасета в результатах (по умолчанию имя файла)")
a = p.parse_args()

if a.list or not a.method:
    print("Методы:  ", ", ".join(sorted(REGISTRY)))
    print("Датасеты:", ", ".join(DATASETS))
    sys.exit(0)

methods = sorted(REGISTRY) if a.method == ["all"] else a.method
unknown = [m for m in methods if m not in REGISTRY]
if unknown:
    sys.exit(f"Неизвестные методы: {unknown}. Доступны: {sorted(REGISTRY)}")

if a.csv:
    if not a.target:
        sys.exit("Для --csv нужен --target (целевой столбец)")
    name = a.name or Path(a.csv).stem
    datasets = {name: lambda: csv_dataset(a.csv, a.target, a.time_col, a.sample)}
else:
    names = a.datasets or list(DATASETS)
    missing = [d for d in names if d not in DATASETS]
    if missing:
        sys.exit(f"Неизвестные датасеты: {missing}. Доступны: {list(DATASETS)}")
    datasets = {d: DATASETS[d] for d in names}

for m in methods:
    for d, loader in datasets.items():
        t = time.time()
        r2, nf = run_method(d, loader, m, REGISTRY[m])
        print(f"{m:18s} {d:14s} призн.={nf:5.1f} " +
              " ".join(f"{k}={v:.4f}" for k, v in r2.items()) + f"  [{time.time()-t:.0f}с]", flush=True)
