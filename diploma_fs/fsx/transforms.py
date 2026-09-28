"""Шаг 3 конвейера: модификации признаков (raw, log, sqrt, sq, inv) с явными группами.

Всё, что подстраивается под данные, оценивается ТОЛЬКО на train-фолде (без утечки):
  - пропуски в числовых столбцах заполняются медианой train;
  - диапазон train [min, max]: перед log/sqrt/sq/inv значение обрезается по нему,
    иначе точка теста левее минимума train даёт inv = 1000 и log = -6.9 (выброс);
    raw не обрезается (линейная экстраполяция безопасна);
  - сдвиг к положительной области (минимум train -> 1) для log/sqrt/inv;
  - список категорий для категориальных столбцов.
Группа = базовый фактор -> список его модификаций (замена startswith() из статьи).
Категориальные столбцы НЕ модифицируются: кодируются 0/1 и идут в модель мимо отбора.
Категориальными считаются нечисловые столбцы и столбцы из cat_cols (например, час, сезон,
закодированные числами) — их список задаётся в реестре датасетов.
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
MAX_CATEGORIES = 30   # категории сверх этого числа объединяются в "other"
NA_LEVEL = "NA"
OTHER_LEVEL = "other"


def base_of(col):
    return col.split(SEP)[0]


def tr_of(col):
    return col.split(SEP)[1]


def _as_cat(s):
    """Категориальный столбец -> строки; пропуск -> 'NA' (astype(str) дал бы 'nan')."""
    return s.astype("object").where(s.notna(), NA_LEVEL).astype(str)


class GroupFE:
    def __init__(self, cat_cols=()):
        self.cat_cols = tuple(cat_cols)

    def fit(self, X: pd.DataFrame):
        forced = set(self.cat_cols)
        missing = forced - set(X.columns)
        if missing:
            raise ValueError(f"cat_cols нет в данных: {sorted(missing)}")
        self.num_ = [c for c in X.columns if c not in forced
                     and pd.api.types.is_numeric_dtype(X[c])
                     and not pd.api.types.is_bool_dtype(X[c])]
        self.cat_ = [c for c in X.columns if c not in self.num_]
        self.median_ = {c: float(X[c].median()) if X[c].notna().any() else 0.0 for c in self.num_}
        filled = {c: X[c].fillna(self.median_[c]).astype(float) for c in self.num_}
        self.lo_ = {c: float(filled[c].min()) for c in self.num_}
        self.hi_ = {c: float(filled[c].max()) for c in self.num_}
        self.shift_ = {c: (0.0 if self.lo_[c] > 0 else 1.0 - self.lo_[c]) for c in self.num_}
        self.groups_ = {c: [f"{c}{SEP}{t}" for t in TRANSFORMS] for c in self.num_}
        self.levels_ = {}
        for c in self.cat_:
            vc = _as_cat(X[c]).value_counts()
            lv = list(vc.index[:MAX_CATEGORIES])
            if len(vc) > MAX_CATEGORIES:
                lv.append(OTHER_LEVEL)
            self.levels_[c] = lv
        self.cat_features_ = [f"{c}{SEP}cat_{v}" for c in self.cat_ for v in self.levels_[c]]
        return self

    def transform(self, X):
        cols = {}
        for c in self.num_:
            raw = X[c].fillna(self.median_[c]).to_numpy(dtype=float)
            v = np.clip(raw, self.lo_[c], self.hi_[c]) + self.shift_[c]
            v = np.maximum(v, 1e-3)
            for t, f in TRANSFORMS.items():
                cols[f"{c}{SEP}{t}"] = raw if t == "raw" else f(v)
        for c in self.cat_:
            s = _as_cat(X[c])
            known = set(self.levels_[c]) - {OTHER_LEVEL}
            if OTHER_LEVEL in self.levels_[c]:
                s = s.where(s.isin(known), OTHER_LEVEL)
            for v in self.levels_[c]:
                cols[f"{c}{SEP}cat_{v}"] = (s == v).to_numpy(dtype=float)
        return pd.DataFrame(cols, index=X.index)

    @property
    def num_features_(self):
        return [f for g in self.groups_.values() for f in g]
