"""Отбор признаков по важности CatBoost (ветка research/forecast, эксперименты E3–E4).

Три способа ранжировать признаки — и ОДНО общее правило, сколько оставить:
  cb_importance  — встроенная важность CatBoost (PredictionValuesChange);
  cb_permutation — перестановочная важность: насколько растёт MAE на валидации, если
                   перемешать признак;
  cb_shap        — средний |SHAP| (CatBoost считает SHAP сам, пакет shap не нужен).

Правило «сколько оставить» (одинаково для всех трёх, чтобы сравнивались именно ранжирования):
  берём верхние k% признаков, k = 10, 20, 30, 40, 50, 70, 100; на валидации считаем MAE
  быстрой модели; оставляем НАИМЕНЬШИЙ набор, чей MAE не хуже лучшего более чем на tol (1%).

Валидация — последние 20% train ПО ВРЕМЕНИ (строки F уже упорядочены по времени), а не
случайные строки: у рядов соседние строки похожи, случайная валидация льстит модели.
Если catboost не установлен — ранжирование делает HistGradientBoosting (перестановки), а в
info пишется backend="hgb_fallback"; встроенную важность и SHAP тогда заменяют перестановки.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

from ..models import has_catboost

GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def _split(F, y, frac=0.2):
    cut = int(len(F) * (1 - frac))
    return F.iloc[:cut], F.iloc[cut:], y[:cut], y[cut:]


def _fast(seed):
    if has_catboost():
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, loss_function="RMSE",
                                 random_seed=seed, verbose=0, allow_writing_files=False), "catboost"
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=seed), "hgb_fallback"


def _choose(order, Ftr, Fva, ytr, yva, seed, tol):
    """Наименьший топ-k по ранжированию order с MAE <= (1+tol)·лучший."""
    n, path, seen = len(order), [], set()
    for frac in GRID:
        k = max(1, int(round(n * frac)))
        if k in seen:
            continue
        seen.add(k)
        cols = order[:k]
        m, _ = _fast(seed)
        m.fit(Ftr[cols], ytr)
        path.append((k, float(mean_absolute_error(yva, m.predict(Fva[cols])))))
    best = min(v for _, v in path)
    k = min(k for k, v in path if v <= best * (1 + tol))
    return list(order[:k]), {"path": path, "k": k}


def _rank(F, y, seed, how):
    Ftr, Fva, ytr, yva = _split(F, y)
    m, backend = _fast(seed)
    m.fit(Ftr, ytr)
    cols = list(F.columns)
    if backend == "catboost" and how == "importance":
        imp = m.get_feature_importance(type="PredictionValuesChange")
    elif backend == "catboost" and how == "shap":
        from catboost import Pool
        sv = m.get_feature_importance(Pool(Fva, yva), type="ShapValues")[:, :-1]
        imp = np.abs(sv).mean(axis=0)
    else:                                            # перестановки (или замена без catboost)
        r = permutation_importance(m, Fva, yva, scoring="neg_mean_absolute_error",
                                   n_repeats=3, random_state=seed)
        imp = r.importances_mean
    order = [cols[i] for i in np.argsort(-np.asarray(imp), kind="stable")]
    return order, dict(zip(cols, map(float, imp))), backend, (Ftr, Fva, ytr, yva)


def _make(how):
    def selector(F, y, groups, seed, tol=0.01):
        y = np.asarray(y, float)
        order, imp, backend, parts = _rank(F, y, seed, how)
        feats, info = _choose(order, *parts, seed, tol)
        top = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:15])
        return feats, {**info, "backend": backend, "top_importance": top}
    selector.__name__ = f"cb_{how}"
    return selector


METHODS = {"cb_importance": _make("importance"), "cb_permutation": _make("permutation"),
           "cb_shap": _make("shap")}
