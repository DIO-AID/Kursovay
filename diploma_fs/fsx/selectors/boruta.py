"""Boruta (Kursa & Rudnicki, 2010), собственная реализация.

К признакам добавляются "теневые" копии с перемешанными строками. На каждой итерации
признак получает "попадание", если его важность выше максимальной важности теней.
Признак подтверждается, если число попаданий значимо больше n/2 (биномиальный тест).
В отличие от порога t=0.1 из статьи — это статистический критерий.
Ожидание: Boruta не устраняет дубли — все модификации важного фактора подтверждаются.
У себя: пакет BorutaShap (важность по SHAP вместо impurity).
"""
import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestRegressor


def boruta(F, y, groups, seed, n_iter=12, alpha=0.05):
    rng = np.random.default_rng(seed)
    X = F.to_numpy()
    hits = np.zeros(X.shape[1], dtype=int)
    for it in range(n_iter):
        shadow = np.apply_along_axis(rng.permutation, 0, X)
        rf = RandomForestRegressor(n_estimators=60, max_depth=8, max_features=0.33,
                                   max_samples=0.6, random_state=seed + it, n_jobs=1)
        rf.fit(np.c_[X, shadow], y)
        imp = rf.feature_importances_
        hits += imp[:X.shape[1]] > imp[X.shape[1]:].max()
    p = [binomtest(int(h), n_iter, 0.5, alternative="greater").pvalue for h in hits]
    feats = [c for c, pv in zip(F.columns, p) if pv < alpha]
    return feats, {"hits": dict(zip(F.columns, hits.tolist()))}


METHODS = {"boruta": boruta}
