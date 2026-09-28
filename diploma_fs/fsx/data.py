"""Шаг 1 конвейера: загрузка данных.

Каждый загрузчик возвращает Dataset:
  X      — исходные факторы (DataFrame; числовые и категориальные столбцы)
  y      — целевая переменная (ndarray)
  truth  — {фактор: истинная трансформация}, известна только для синтетики
  time   — True, если строки упорядочены по времени (тогда разбиение блоками, шаг 2)

Правило шага: ничего не "учить" на данных (никаких средних, нормировок) —
всё, что подстраивается под данные, делается на шаге 3 внутри train-фолда.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from .paths import DATA_DIRS

TRUTH = {"x1": "log", "x2": "sq", "x3": "inv", "x4": "sqrt"}


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    truth: dict | None = None
    time: bool = False
    meta: dict = field(default_factory=dict)


def _target(X, rng, noise=0.3):
    return (3 * np.log(X["x1"]) + 0.2 * X["x2"] ** 2 + 2 / X["x3"]
            + 1.5 * np.sqrt(X["x4"]) + rng.normal(0, noise, len(X))).to_numpy()


def synth_indep(n=1500, seed=0):
    """10 независимых факторов U(0.5, 5); 4 информативных, 6 шумовых.
    Имена x1..x10 выбраны намеренно: startswith('x1') ловит и x10."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.uniform(0.5, 5, n) for i in range(1, 11)})
    return Dataset(X, _target(X, rng), dict(TRUTH))


def synth_corr(n=1500, seed=1, rho=0.8):
    """То же, но каждый информативный фактор коррелирует (rho) с шумовым двойником:
    x1~x5, x2~x6, x3~x7, x4~x8."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 10))
    for a, b in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        Z[:, b] = rho * Z[:, a] + np.sqrt(1 - rho ** 2) * Z[:, b]
    X = pd.DataFrame(0.5 + 4.5 * norm.cdf(Z), columns=[f"x{i}" for i in range(1, 11)])
    return Dataset(X, _target(X, rng), dict(TRUTH))


def diabetes():
    from sklearn.datasets import load_diabetes
    d = load_diabetes(as_frame=True)
    return Dataset(d.data.copy(), d.target.to_numpy())


def find_data(rel):
    """Ищет файл в diploma_fs/data и в корневой data/ репозитория."""
    for base in DATA_DIRS:
        if (base / rel).exists():
            return base / rel
    return None


def csv_dataset(path, target, time_col=None, sample=None, seed=0, sep=None):
    """Любой CSV: python scripts/run.py --csv файл.csv --target столбец [--time-col дата] [--sample N]

    - строки без значения цели удаляются;
    - time_col: строки сортируются по времени, разбиение будет блоками (без перемешивания);
    - sample: случайная выборка N строк (для скорости на слабом компьютере);
    - числовые столбцы получат модификации, категориальные пойдут в модель как есть.
    """
    df = pd.read_csv(path, sep=sep, engine="python" if sep is None else "c")
    if target not in df.columns:
        raise ValueError(f"Нет столбца '{target}'. Есть: {list(df.columns)}")
    df = df.dropna(subset=[target]).dropna(axis=1, how="all")
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=seed)
    is_time = time_col is not None
    if is_time:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col).drop(columns=[time_col])
    y = pd.to_numeric(df.pop(target), errors="coerce")
    keep = y.notna().to_numpy()
    X = df.loc[keep].reset_index(drop=True)
    return Dataset(X, y[keep].to_numpy(dtype=float), None, is_time,
                   {"source": str(path), "target": target, "time_col": time_col,
                    "sample": sample})


DATASETS = {
    "synth_indep": synth_indep,
    "synth_corr": synth_corr,
    "diabetes": diabetes,
}
if find_data("housing.csv"):
    DATASETS["housing"] = lambda: csv_dataset(find_data("housing.csv"), "median_house_value")
