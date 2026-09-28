"""Групповая пермутационная важность.

1) Бустинг обучается на ВСЕХ модификациях.
2) Для каждой группы (фактора) все её модификации перемешиваются ОДНОЙ перестановкой
   строк; падение R2 на внутренней валидации = важность фактора целиком.
   Так важность не "размазывается" между копиями: сумма по группе честная.
3) Остаются группы с падением R2 > eps.
4) Представитель группы — модификация с наибольшей |корреляцией Пирсона| с y
   (для деревьев выбор безразличен, для линейной модели/MLP — критичен).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def group_perm_importance(model, Fva, yva, groups, rng, n_repeats=5):
    base = r2_score(yva, model.predict(Fva))
    imp = {}
    for g, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            P = Fva.copy()
            idx = rng.permutation(len(P))
            P[cols] = Fva[cols].to_numpy()[idx]
            drops.append(base - r2_score(yva, model.predict(P)))
        imp[g] = float(np.mean(drops))
    return imp


def _abs_corr(a, b):
    """|корреляция Пирсона|; для постоянного столбца 0 (а не NaN, из-за которого падал argmax)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def group_importance(F, y, groups, seed, eps=0.005):
    Ftr, Fva, ytr, yva = train_test_split(F, y, test_size=0.2, random_state=seed)
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                      random_state=seed).fit(Ftr, ytr)
    imp = group_perm_importance(m, Fva, yva, groups, np.random.default_rng(seed))
    feats = []
    for g, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        if v > eps:
            cols = groups[g]
            corr = np.array([_abs_corr(F[c].to_numpy(), y) for c in cols])
            feats.append(cols[int(np.argmax(corr))])
    return feats, {"group_importance": imp}


METHODS = {"group_importance": group_importance}
