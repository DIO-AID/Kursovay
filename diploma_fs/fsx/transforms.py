"""Шаг 3 конвейера: модификации признаков (raw, log, sqrt, sq, inv) с явными группами.

Всё, что подстраивается под данные, оценивается ТОЛЬКО на train-фолде (без утечки):
  - пропуски в числовых столбцах заполняются медианой train;
  - сдвиг к положительной области (минимум train -> 1) для log/sqrt/inv;
  - список категорий для категориальных столбцов.
Группа = базовый фактор -> список его модификаций (замена startswith() из статьи).
Категориальные столбцы НЕ модифицируются: кодируются 0/1 и идут в модель как есть,
мимо отбора (в отборе участвуют только числовые группы).
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
MAX_CATEGORIES = 30   # редкие категории сверх этого числа объединяются в "other"


def base_of(col):
    return col.split(SEP)[0]


def tr_of(col):
    return col.split(SEP)[1]


class GroupFE:
    def fit(self, X: pd.DataFrame):
        self.num_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
                     and not pd.api.types.is_bool_dtype(X[c])]
        self.cat_ = [c for c in X.columns if c not in self.num_]
        self.median_ = {c: float(X[c].median()) if X[c].notna().any() else 0.0 for c in self.num_}
        filled = {c: X[c].fillna(self.median_[c]) for c in self.num_}
        self.shift_ = {c: (0.0 if filled[c].min() > 0 else 1.0 - filled[c].min()) for c in self.num_}
        self.groups_ = {c: [f"{c}{SEP}{t}" for t in TRANSFORMS] for c in self.num_}
        self.levels_ = {}
        for c in self.cat_:
            vc = X[c].astype(str).fillna("NA").value_counts()
            self.levels_[c] = list(vc.index[:MAX_CATEGORIES])
        self.cat_features_ = [f"{c}{SEP}cat_{v}" for c in self.cat_ for v in self.levels_[c]]
        return self

    def transform(self, X):
        cols = {}
        for c in self.num_:
            raw = X[c].fillna(self.median_[c]).to_numpy(dtype=float)
            v = np.maximum(raw + self.shift_[c], 1e-3)
            for t, f in TRANSFORMS.items():
                cols[f"{c}{SEP}{t}"] = raw if t == "raw" else f(v)
        for c in self.cat_:
            s = X[c].astype(str).fillna("NA")
            for v in self.levels_[c]:
                cols[f"{c}{SEP}cat_{v}"] = (s == v).to_numpy(dtype=float)
        return pd.DataFrame(cols, index=X.index)

    @property
    def num_features_(self):
        return [f for g in self.groups_.values() for f in g]
