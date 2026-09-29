"""Шаг 1 конвейера: загрузка данных.

Каждый загрузчик возвращает Dataset:
  X      — исходные факторы (DataFrame; числовые и категориальные столбцы)
  y      — целевая переменная (ndarray)
  truth  — {фактор: истинная трансформация}, известна только для синтетики
  time   — True, если строки упорядочены по времени (тогда разбиение по времени, шаг 2)
  meta   — описание источника; meta["cat_cols"] — числовые по записи, но категориальные столбцы
  t      — моменты времени строк (pd.Series datetime, отсортированы) или None;
           нужны для временных признаков (fsx/lags.py) в ветке research/forecast

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
    t: pd.Series | None = None


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


def _ar1(rng, n, phi, sd=1.0):
    e = rng.normal(0, sd, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def synth_factory(n=6000, seed=0, n_sensors=30, n_informative=6, delay=6):
    """Модельный завод (временной ряд, шаг 15 мин, ~2 месяца) — для проверки ветки research/forecast.

    Потребление (кВт·ч) = график смен (час, будни/выходные) + вклад 6 информативных датчиков
    С ЗАПАЗДЫВАНИЕМ delay шагов (загрузка линии сказывается на потреблении через ~1.5 ч) +
    инерция AR(0.8) + шум. Поэтому прошлые показания датчиков действительно помогают прогнозу.
    Остальные 24 датчика — шум; у первых 6 шумовых есть «двойник»-информативный (корр. ~0.7).
    Истина: meta["informative"] — какие датчики влияют на цель.
    """
    rng = np.random.default_rng(seed)
    t = pd.Series(pd.date_range("2024-01-01", periods=n, freq="15min"))
    hour = (t.dt.hour + t.dt.minute / 60).to_numpy()
    work = (t.dt.dayofweek < 5).to_numpy()
    shift = np.where((hour >= 7) & (hour < 23), 1.0, 0.35) * np.where(work, 1.0, 0.45)
    S = {}
    for j in range(n_sensors):
        S[f"s{j + 1:02d}"] = _ar1(rng, n, phi=0.97 if j < n_informative else 0.9)
    for j in range(min(n_informative, n_sensors - n_informative)):   # коррелированные двойники
        twin = f"s{n_informative + j + 1:02d}"
        S[twin] = 0.7 * S[f"s{j + 1:02d}"] + 0.3 * S[twin]
    X = pd.DataFrame(S)
    coef = np.array([6, -5, 4, 3, -3, 2], dtype=float)[:n_informative]
    signal = sum(c * X[f"s{j + 1:02d}"].to_numpy() for j, c in enumerate(coef))
    signal = np.r_[np.zeros(delay), signal[:-delay]]    # эффект с запаздыванием
    inertia = _ar1(rng, n, phi=0.8, sd=2.0)
    y = 20 + 60 * shift + signal * shift + inertia + rng.normal(0, 2, n)
    meta = {"name": "synth_factory", "informative": [f"s{j + 1:02d}" for j in range(n_informative)],
            "horizon": "1h", "known_ahead": [], "cat_cols": []}
    return Dataset(X, np.maximum(y, 0.0), None, True, meta, t)


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
                drop=(), cat_cols=(), name=None, time_format=None, hour_col=None):
    """Любой CSV: python scripts/run.py --csv файл.csv --target столбец [--time-col дата] [--sample N]

    Правила (нарушение -> DataError с объяснением, ничего не выбрасывается молча):
    - цель должна быть числовой во всех строках, где она заполнена; строки с ПУСТОЙ целью
      удаляются, их число пишется в meta["dropped_empty_target"];
    - time_col: все даты должны распознаваться; строки сортируются по времени;
      time_format (например "%d/%m/%Y %H:%M") задаётся явно: без него pandas молча читает
      13/01 и 01/02 по-разному, и порядок ряда ломается;
      sample для временных данных = непрерывный хвост ряда (последние N строк);
    - sample для обычных данных = случайные N строк (seed);
    - drop: колонки-утечки и идентификаторы; cat_cols: категориальные, даже если числа;
    - hour_col: если дата без времени, а час в отдельной колонке (Seoul Bike) — время = дата + час.
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
        if hour_col:
            if hour_col not in df.columns:
                raise DataError(f"{path}: нет колонки часа '{hour_col}'")
            t = t + pd.to_timedelta(_strict_numeric(df[hour_col], hour_col, "Час"), unit="h")
        order = np.argsort(t.to_numpy(), kind="stable")
        df, y, t = df.iloc[order], y.iloc[order], t.iloc[order]
        df = df.drop(columns=[time_col])
        if sample and len(df) > sample:
            df, y, t = df.iloc[-sample:], y.iloc[-sample:], t.iloc[-sample:]
    elif sample and len(df) > sample:
        idx = df.sample(n=sample, random_state=seed).index
        df, y = df.loc[idx], y.loc[idx]
    X = df.drop(columns=[target]).dropna(axis=1, how="all").reset_index(drop=True)
    y_arr = y.to_numpy(dtype=float)
    meta = {"source": str(path), "name": name, "target": target, "time_col": time_col,
            "sample": sample, "time_format": time_format, "drop": list(drop), "cat_cols": [c for c in cat_cols if c in X.columns],
            "rows_file": n0, "dropped_empty_target": int(n_empty),
            "n_unique_target": int(pd.Series(y_arr).nunique())}
    t_out = t.reset_index(drop=True) if time_col else None
    return Dataset(X, y_arr, None, time_col is not None, meta, t_out)


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
                     cat_cols=r["cat_cols"], name=name, time_format=r.get("time_format"),
                     hour_col=r.get("hour_col"))
    ds.meta.update(role=r["role"], status=r["status"], uci_id=r["uci_id"],
                   horizon=r.get("horizon"), known_ahead=r.get("known_ahead", []))
    return ds


DATASETS = {
    "synth_indep": synth_indep,
    "synth_corr": synth_corr,
    "diabetes": diabetes,
}
TIME_DATASETS = {"synth_factory": synth_factory}   # встроенные временные ряды (research/forecast)
if find_data("housing.csv"):
    DATASETS["housing"] = lambda: csv_dataset(find_data("housing.csv"), "median_house_value")
