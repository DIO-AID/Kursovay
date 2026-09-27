"""Генерация модификаций признаков (raw, log, sqrt, sq, inv) с явными группами.

Сдвиг к положительной области оценивается ТОЛЬКО на train (без утечки).
Группа = базовый фактор -> список его модификаций. Это заменяет startswith() из статьи.
"""
import numpy as np
import pandas as pd

TRANSFORMS = {
    "raw": lambda v: v,
    "log": np.log,
    "sqrt": np.sqrt,
    "sq": np.square,
    "inv": lambda v: 1.0 / v,
}
SEP = "__"


def base_of(col):
    return col.split(SEP)[0]


def tr_of(col):
    return col.split(SEP)[1]


class GroupFE:
    def fit(self, X: pd.DataFrame):
        # сдвиг: минимум train -> 1 (для уже положительных данных сдвиг не нужен)
        self.shift_ = {c: (0.0 if X[c].min() > 0 else 1.0 - X[c].min()) for c in X.columns}
        self.base_ = list(X.columns)
        self.groups_ = {c: [f"{c}{SEP}{t}" for t in TRANSFORMS] for c in X.columns}
        return self

    def positive(self, X):
        return pd.DataFrame({c: np.maximum(X[c].to_numpy() + self.shift_[c], 1e-3)
                             for c in self.base_}, index=X.index)

    def transform(self, X):
        P = self.positive(X)
        cols = {}
        for c in self.base_:
            v = P[c].to_numpy()
            for t, f in TRANSFORMS.items():
                cols[f"{c}{SEP}{t}"] = X[c].to_numpy() if t == "raw" else f(v)
        return pd.DataFrame(cols, index=X.index)
