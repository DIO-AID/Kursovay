"""Восстановление трансформации по форме зависимости (аналог SHAP dependence / EBM).

Идея: не генерировать трансформации до обучения, а обучить модель на ИСХОДНЫХ
признаках и извлечь из неё форму зависимости y от каждого фактора.
1) Аддитивный бустинг (interaction_cst='no_interactions') — это аналог EBM:
   f(x) = f1(x1) + ... + fm(xm). Для такой модели partial dependence совпадает
   с формой fj (с точностью до константы) и с SHAP dependence plot.
2) К кривой fj подбираются формы a + c*g(x), g in {raw, log, sqrt, sq, inv};
   выбирается g с максимальным R2 подгонки.
3) Важность фактора = Var(fj(xj)) / Var(y); остаются факторы с долей > eps.
У себя: заменить на shap.TreeExplainer(CatBoost) + подгонку к shap_values[:, j].
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence

from ..transforms import SEP, TRANSFORMS


def _fit_r2(g, f):
    A = np.c_[np.ones_like(g), g]
    coef, *_ = np.linalg.lstsq(A, f, rcond=None)
    res = f - A @ coef
    tot = ((f - f.mean()) ** 2).sum()
    return 1 - (res ** 2).sum() / tot if tot > 0 else 0.0


def shape_fit(F, y, groups, seed, eps=0.005):
    bases = list(groups)
    raw = [f"{b}{SEP}raw" for b in bases]
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                      interaction_cst="no_interactions",
                                      random_state=seed).fit(F[raw], y)
    vy = np.var(y)
    feats, info = [], {}
    for j, b in enumerate(bases):
        pd_ = partial_dependence(m, F[raw], [j], grid_resolution=40, kind="average")
        grid, f = pd_["grid_values"][0], pd_["average"][0]
        shift = float(np.mean(F[f"{b}{SEP}sqrt"] ** 2 - F[f"{b}{SEP}raw"]))
        pos = np.maximum(grid + shift, 1e-3)
        fits = {t: _fit_r2(pos if t == "raw" else TRANSFORMS[t](pos), f) for t in TRANSFORMS}
        best = max(fits, key=fits.get)
        if fits[best] - fits["raw"] < 1e-3:          # при равенстве — без трансформации
            best = "raw"
        share = float(np.var(np.interp(F[f"{b}{SEP}raw"], grid, f)) / vy)
        info[b] = {"transform": best, "fit_r2": {k: round(v, 4) for k, v in fits.items()},
                   "share": share, "grid": np.round(grid, 4).tolist(),
                   "pd": np.round(f, 4).tolist()}
        if share > eps:
            feats.append(f"{b}{SEP}{best}")
    return feats, info


METHODS = {"shape_fit": shape_fit}
