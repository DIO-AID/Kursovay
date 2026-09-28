"""Базовые линии ВЫБОРА ФОРМЫ — то, с чем «Форму зависимости» спросят сравнить на защите.

boxcox      — для каждого фактора оценивается λ Бокса–Кокса (Box & Cox, 1964) на train
              и округляется к ближайшей из наших форм: λ≈1 raw, 0 log, 0.5 sqrt, 2 sq, −1 inv.
              Критерий — нормальность САМОГО x, связь с y не учитывается (так делают на практике).
              Отбора факторов нет: берутся все.
yeojohnson  — то же с преобразованием Йео–Джонсона (Yeo & Johnson, 2000).
ace         — упрощённый ACE (Breiman & Friedman, 1985) с θ(y) = y: аддитивная модель
              y ≈ Σ φ_j(x_j) подгоняется backfitting-ом со сглаживанием средними по квантильным
              бинам; к каждой φ_j подбирается ближайшая форма a + c·g(x), как в shape_fit;
              остаются факторы с долей дисперсии Var(φ_j)/Var(y) > eps.
              Отличие от shape_fit только в оценщике формы (сглаживатель вместо бустинга),
              поэтому это прямой контроль: даёт ли что-то именно бустинг.
"""
import numpy as np
from scipy import stats

from ..transforms import SEP, TRANSFORMS

LAMBDA_TO_FORM = {1.0: "raw", 0.0: "log", 0.5: "sqrt", 2.0: "sq", -1.0: "inv"}


def _nearest_form(lam):
    if not np.isfinite(lam):
        return "raw"
    key = min(LAMBDA_TO_FORM, key=lambda k: abs(k - lam))
    return LAMBDA_TO_FORM[key]


def _positive(F, b):
    # sqrt-столбец GroupFE = sqrt(x + shift) > 0, поэтому его квадрат — уже сдвинутый x
    return np.asarray(F[f"{b}{SEP}sqrt"], dtype=float) ** 2


def _power_select(F, y, groups, seed, kind):
    feats, info = [], {}
    for b in groups:
        v = _positive(F, b)
        if np.ptp(v) == 0:
            info[b] = {"lambda": None, "transform": "raw"}
            feats.append(f"{b}{SEP}raw")
            continue
        try:
            lam = stats.boxcox_normmax(v, method="mle") if kind == "boxcox" else \
                stats.yeojohnson_normmax(v)
        except Exception:
            lam = np.nan
        t = _nearest_form(float(lam))
        info[b] = {"lambda": None if not np.isfinite(lam) else round(float(lam), 3), "transform": t}
        feats.append(f"{b}{SEP}{t}")
    return feats, info


def boxcox(F, y, groups, seed):
    return _power_select(F, y, groups, seed, "boxcox")


def yeojohnson(F, y, groups, seed):
    return _power_select(F, y, groups, seed, "yeojohnson")


def _fit_r2(g, f):
    A = np.c_[np.ones_like(g), g]
    coef, *_ = np.linalg.lstsq(A, f, rcond=None)
    tot = ((f - f.mean()) ** 2).sum()
    return 1 - ((f - A @ coef) ** 2).sum() / tot if tot > 0 else 0.0


def _bin_smoother(x, r, bins):
    """Сглаживание средним по квантильным бинам x; возвращает значения в точках x и (центры, средние)."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.zeros_like(r), (np.array([x.mean()]), np.array([0.0]))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    cnt = np.bincount(idx, minlength=len(edges) - 1)
    s = np.bincount(idx, weights=r, minlength=len(edges) - 1)
    means = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
    centers = np.bincount(idx, weights=x, minlength=len(edges) - 1) / np.maximum(cnt, 1)
    ok = cnt > 0
    return means[idx], (centers[ok], means[ok])


def ace(F, y, groups, seed, eps=0.005, bins=20, n_iter=15):
    bases = list(groups)
    X = {b: _positive(F, b) for b in bases}
    yc = np.asarray(y, dtype=float) - np.mean(y)
    phi = {b: np.zeros(len(yc)) for b in bases}
    for _ in range(n_iter):                     # backfitting
        for b in bases:
            partial = yc - sum(phi[o] for o in bases if o != b)
            f, _ = _bin_smoother(X[b], partial, bins)
            phi[b] = f - f.mean()
    vy = np.var(yc)
    feats, info = [], {}
    for b in bases:
        _, (cx, cf) = _bin_smoother(X[b], phi[b], bins)
        if len(cx) < 3:
            continue
        fits = {t: _fit_r2(cx if t == "raw" else TRANSFORMS[t](cx), cf) for t in TRANSFORMS}
        best = max(fits, key=fits.get)
        if fits[best] - fits["raw"] < 1e-3:
            best = "raw"
        share = float(np.var(phi[b]) / vy) if vy > 0 else 0.0
        info[b] = {"transform": best, "share": share,
                   "fit_r2": {k: round(v, 4) for k, v in fits.items()}}
        if share > eps:
            feats.append(f"{b}{SEP}{best}")
    return feats, info


METHODS = {"boxcox": boxcox, "yeojohnson": yeojohnson, "ace": ace}
