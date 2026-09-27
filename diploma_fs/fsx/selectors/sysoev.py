"""Алгоритм 1 из статьи Сысоева (SUMMA 2023) и его исправленная версия.

sysoev_paper — буквально по псевдокоду:
  * важность = встроенная (impurity) важность бустинга, нормированная на сумму;
  * порог t=0.1 применяется к ПЕРЕНОРМИРОВАННЫМ весам на каждой итерации;
  * группа удаляется через startswith(имя фактора) -> 'x1' удаляет и 'x10';
  * при i=0 лучший признак удаляется, но НЕ добавляется в результат (строки 7-8).

sysoev_fixed — четыре исправления:
  1) явный словарь групп вместо startswith;
  2) лучший признак добавляется на каждой итерации, включая первую;
  3) важность оставшихся признаков оценивается В ПРИСУТСТВИИ уже выбранных
     (условная важность): в статье модель переобучается без выбранных факторов,
     и их вклад в y превращается в "шум", маскирующий слабые факторы;
  4) остановка, когда оставшиеся признаки не добавляют R2 на внутренней
     валидации (прирост < eps), а не по перенормированному порогу t.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from ..transforms import base_of


def _gbr(seed):
    return GradientBoostingRegressor(n_estimators=100, max_depth=3, subsample=0.8,
                                     random_state=seed)


def _weights(F, y, cols, seed):
    m = _gbr(seed).fit(F[cols], y)
    w = m.feature_importances_ / m.feature_importances_.sum()
    order = np.argsort(-w)
    return [cols[i] for i in order], w[order]


def sysoev_paper(F, y, groups, seed, t=0.1):
    cur = list(F.columns)
    names, w = _weights(F, y, cur, seed)
    n_iter = int((w > t).sum())
    features, log = [], []
    for i in range(n_iter):
        if i > 0:
            if not cur:
                break
            names, w = _weights(F, y, cur, seed)
            names = [n for n, v in zip(names, w) if v > t]
            if not names:
                break
        top = names[0]
        prefix = base_of(top)                      # 'x1' -> startswith ловит и 'x10'
        dropped = [c for c in cur if c.startswith(prefix)]
        cur = [c for c in cur if c not in dropped]
        if i > 0:
            features.append(top)
        log.append({"top": top, "dropped_groups": sorted({base_of(c) for c in dropped})})
    return features, {"iterations": log, "lost_first": log[0]["top"] if log else None}


def sysoev_fixed(F, y, groups, seed, eps=0.005):
    Ftr, Fva, ytr, yva = train_test_split(F, y, test_size=0.2, random_state=seed)
    rem, sel = list(F.columns), []
    r2_sel = 0.0
    while rem:
        m = _gbr(seed).fit(Ftr[sel + rem], ytr)
        r2_full = r2_score(yva, m.predict(Fva[sel + rem]))
        if r2_full - r2_sel < eps:                    # остаток ничего не добавляет
            break
        w = m.feature_importances_[len(sel):]          # важность только среди оставшихся
        top = rem[int(np.argmax(w))]
        sel.append(top)
        grp = set(groups[base_of(top)])                # явная группа, без startswith
        rem = [c for c in rem if c not in grp]
        r2_sel = r2_score(yva, _gbr(seed).fit(Ftr[sel], ytr).predict(Fva[sel]))
    return sel, {}


METHODS = {"sysoev_paper": sysoev_paper, "sysoev_fixed": sysoev_fixed}
