"""Шаг 1 конвейера: загрузка данных.

Каждый загрузчик возвращает Dataset:
  X      — исходные факторы (DataFrame; числовые и категориальные столбцы)
  y      — целевая переменная (ndarray)
  truth  — {фактор: истинная трансформация}, известна только для синтетики
  time   — True, если строки упорядочены по времени (тогда разбиение по времени, шаг 2)
  meta   — описание источника; meta["cat_cols"] — числовые по записи, но категориальные столбцы

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


class DataError(ValueError):
    """Данные не соответствуют описанию — падаем громко, а не теряем строки молча."""


def _strict_numeric(series, name, what):
    v = pd.to_numeric(series, errors="coerce")
    bad = v.isna() & series.notna()
    if bad.any():
        ex = list(pd.unique(series[bad].astype(str)))[:5]
        raise DataError(f"{what} '{name}': {int(bad.sum())} из {len(series)} значений не числа, "
                        f"примеры: {ex}. Проверьте, та ли это колонка.")
    return v


def csv_dataset(path, target, time_col=None, sample=None, seed=0, sep=None,
                drop=(), cat_cols=(), name=None, time_format=None):
    """Любой CSV: python scripts/run.py --csv файл.csv --target столбец [--time-col дата] [--sample N]

    Правила (нарушение -> DataError с объяснением, ничего не выбрасывается молча):
    - цель должна быть числовой во всех строках, где она заполнена; строки с ПУСТОЙ целью
      удаляются, их число пишется в meta["dropped_empty_target"];
    - time_col: все даты должны распознаваться; строки сортируются по времени;
      time_format (например "%d/%m/%Y %H:%M") задаётся явно: без него pandas молча читает
      13/01 и 01/02 по-разному, и порядок ряда ломается;
      sample для временных данных = непрерывный хвост ряда (последние N строк);
    - sample для обычных данных = случайные N строк (seed);
    - drop: колонки-утечки и идентификаторы; cat_cols: категориальные, даже если числа.
    """
    df = pd.read_csv(path, sep=sep, engine="python" if sep is None else "c")
    df.columns = [str(c).strip() for c in df.columns]
    target, time_col = target.strip(), time_col.strip() if time_col else None
    need = [target] + ([time_col] if time_col else []) + list(drop) + list(cat_cols)
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise DataError(f"{path}: нет колонок {miss}. Есть: {list(df.columns)}")
    df = df.drop(columns=list(drop))
    n0 = len(df)
    df = df.dropna(subset=[target])
    n_empty = n0 - len(df)
    y = _strict_numeric(df[target], target, "Цель")
    if time_col:
        t = pd.to_datetime(df[time_col], errors="coerce", format=time_format)
        if t.isna().any():
            ex = list(df.loc[t.isna(), time_col].astype(str).head(5))
            raise DataError(f"Время '{time_col}': {int(t.isna().sum())} значений не распознаны, "
                            f"примеры: {ex}")
        order = np.argsort(t.to_numpy(), kind="stable")
        df, y = df.iloc[order], y.iloc[order]
        df = df.drop(columns=[time_col])
        if sample and len(df) > sample:
            df, y = df.iloc[-sample:], y.iloc[-sample:]
    elif sample and len(df) > sample:
        idx = df.sample(n=sample, random_state=seed).index
        df, y = df.loc[idx], y.loc[idx]
    X = df.drop(columns=[target]).dropna(axis=1, how="all").reset_index(drop=True)
    y_arr = y.to_numpy(dtype=float)
    meta = {"source": str(path), "name": name, "target": target, "time_col": time_col,
            "sample": sample, "time_format": time_format, "drop": list(drop), "cat_cols": [c for c in cat_cols if c in X.columns],
            "rows_file": n0, "dropped_empty_target": int(n_empty),
            "n_unique_target": int(pd.Series(y_arr).nunique())}
    return Dataset(X, y_arr, None, time_col is not None, meta)


def registry_dataset(name):
    """Датасет из fsx/registry.py по имени (python scripts/run.py --dataset имя)."""
    from .registry import DATASETS as REG
    if name not in REG:
        raise DataError(f"Нет '{name}' в fsx/registry.py. Есть: {sorted(REG)}")
    r = REG[name]
    path = find_data(r["file"])
    if path is None:
        raise DataError(f"Файл data/{r['file']} не найден. Скачайте: python scripts/download_data.py {name}")
    ds = csv_dataset(path, r["target"], r["time_col"], r["sample"], drop=r["drop"],
                     cat_cols=r["cat_cols"], name=name, time_format=r.get("time_format"))
    ds.meta.update(role=r["role"], status=r["status"], uci_id=r["uci_id"])
    return ds


DATASETS = {
    "synth_indep": synth_indep,
    "synth_corr": synth_corr,
    "diabetes": diabetes,
}
if find_data("housing.csv"):
    DATASETS["housing"] = lambda: csv_dataset(find_data("housing.csv"), "median_house_value")
