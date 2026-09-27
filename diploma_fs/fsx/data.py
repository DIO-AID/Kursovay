"""Датасеты для экспериментов по отбору признаков с трансформациями.

Каждый загрузчик возвращает (X: DataFrame исходных признаков, y: ndarray, truth: dict|None).
truth = {базовый_признак: истинная_трансформация} — известна только для синтетики.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

TRUTH = {"x1": "log", "x2": "sq", "x3": "inv", "x4": "sqrt"}


def _target(X, rng, noise=0.3):
    return (3 * np.log(X["x1"]) + 0.2 * X["x2"] ** 2 + 2 / X["x3"]
            + 1.5 * np.sqrt(X["x4"]) + rng.normal(0, noise, len(X))).to_numpy()


def synth_indep(n=1500, seed=0):
    """10 независимых факторов U(0.5, 5); 4 информативных, 6 шумовых.
    Имена x1..x10 выбраны намеренно: startswith('x1') ловит и x10."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.uniform(0.5, 5, n) for i in range(1, 11)})
    return X, _target(X, rng), dict(TRUTH)


def synth_corr(n=1500, seed=1, rho=0.8):
    """То же, но каждый информативный фактор коррелирует (rho) с шумовым двойником:
    x1~x5, x2~x6, x3~x7, x4~x8. Проверка устойчивости к мультиколлинеарности факторов."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 10))
    for a, b in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        Z[:, b] = rho * Z[:, a] + np.sqrt(1 - rho ** 2) * Z[:, b]
    U = norm.cdf(Z)
    X = pd.DataFrame(0.5 + 4.5 * U, columns=[f"x{i}" for i in range(1, 11)])
    return X, _target(X, rng), dict(TRUTH)


def diabetes():
    from sklearn.datasets import load_diabetes
    d = load_diabetes(as_frame=True)
    return d.data.copy(), d.target.to_numpy(), None


def _find(rel):
    """Ищет файл в ./data и ../data (код лежит в подпапке репозитория Kursovay)."""
    for base in (Path("data"), Path("../data")):
        if (base / rel).exists():
            return base / rel
    return None


def housing_csv(path=None):
    """California Housing из репозитория Kursovay (запуск на своей машине)."""
    df = pd.read_csv(path or _find("housing.csv")).dropna()
    y = df.pop("median_house_value").to_numpy()
    X = df.select_dtypes("number")
    return X, y, None


def superconductivity_csv(path=None):
    """UCI Superconductivity (81 признак, группы по физическим свойствам)."""
    df = pd.read_csv(path or _find("superconductivity/train.csv"))
    y = df.pop("critical_temp").to_numpy()
    return df, y, None


DATASETS = {
    "synth_indep": synth_indep,
    "synth_corr": synth_corr,
    "diabetes": diabetes,
}
# Подключаются автоматически, если файл лежит на диске
if _find("housing.csv"):
    DATASETS["housing"] = housing_csv
if _find("superconductivity/train.csv"):
    DATASETS["superconductivity"] = superconductivity_csv
