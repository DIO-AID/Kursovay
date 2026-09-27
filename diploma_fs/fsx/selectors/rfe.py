"""Рекурсивное исключение признаков (аналог CatBoost select_features).

На каждом шаге модель обучается на текущем наборе, 20% наименее важных признаков
удаляются, R2 на внутренней валидации запоминается. Итог — наименьший набор,
чей R2 не хуже лучшего более чем на tol (правило "проще при равном качестве").
Группы НЕ учитываются — это базовая линия "как делают по умолчанию".
У себя: CatBoostRegressor.select_features(algorithm=RecursiveByShapValues, steps=...).
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def rfe(F, y, groups, seed, drop_frac=0.2, tol=0.002):
    Ftr, Fva, ytr, yva = train_test_split(F, y, test_size=0.2, random_state=seed)
    cur, path = list(F.columns), []
    while cur:
        m = GradientBoostingRegressor(n_estimators=100, max_depth=3, subsample=0.8,
                                      random_state=seed).fit(Ftr[cur], ytr)
        path.append((list(cur), r2_score(yva, m.predict(Fva[cur]))))
        k = max(1, int(len(cur) * drop_frac))
        keep = np.argsort(-m.feature_importances_)[:len(cur) - k]
        cur = [cur[i] for i in sorted(keep)]
    best = max(r for _, r in path)
    feats = min((s for s, r in path if r >= best - tol), key=len)
    return feats, {"path": [(len(s), round(r, 4)) for s, r in path]}


METHODS = {"rfe": rfe}
