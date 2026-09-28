"""Проверка H0: монотонное преобразование признака не меняет разбиений дерева.

Запуск (из папки diploma_fs):  python scripts/theory_check.py
Результат: results/theory_check.json (используется в scripts/report.py, график fig0).

Для 40 комбинаций модификаций факторов x1..x4 (истинная + 39 случайных) обучаем
бустинг и Ridge на ОДНОМ представителе каждого фактора. Если H0 верна, R² бустинга
почти не зависит от комбинации, а R² Ridge — зависит сильно.

Это известный факт (деревья инвариантны к строго монотонным преобразованиям,
см. Breiman et al., 1984; Hastie et al., ESL, гл. 9–10) — здесь он проверяется
на наших данных, а не открывается заново. Небольшой разброс у бустинга остаётся
из-за гистограммного бинирования HistGradientBoosting (границы бинов по квантилям
почти, но не строго инвариантны) и числовых эффектов.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import RidgeCV                    # noqa: E402
from sklearn.metrics import r2_score                        # noqa: E402
from sklearn.model_selection import train_test_split        # noqa: E402
from sklearn.pipeline import make_pipeline                  # noqa: E402
from sklearn.preprocessing import StandardScaler            # noqa: E402

from fsx.data import synth_indep                            # noqa: E402
from fsx.paths import RESULTS                               # noqa: E402
from fsx.transforms import SEP, TRANSFORMS, GroupFE         # noqa: E402

N_COMBOS = 40


def main():
    ds = synth_indep()
    Xtr, Xte, ytr, yte = train_test_split(ds.X, ds.y, test_size=0.25, random_state=0)
    fe = GroupFE().fit(Xtr)
    Ftr, Fte = fe.transform(Xtr), fe.transform(Xte)

    factors = sorted(ds.truth)                                   # x1..x4
    truth_combo = tuple(ds.truth[b] for b in factors)
    # выборка комбинаций как в цикле 1 (для воспроизводимости чисел): случайные 39 из всех 625
    combos = list(itertools.product(TRANSFORMS, repeat=len(factors)))
    rng = np.random.default_rng(0)
    sample = [truth_combo] + [combos[i] for i in rng.choice(len(combos), N_COMBOS - 1, replace=False)]

    rows = []
    for c in sample:
        cols = [f"{b}{SEP}{t}" for b, t in zip(factors, c)]
        hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        rid = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        rows.append({"combo": dict(zip(factors, c)), "is_truth": c == truth_combo,
                     "hgb": float(r2_score(yte, hgb.fit(Ftr[cols], ytr).predict(Fte[cols]))),
                     "ridge": float(r2_score(yte, rid.fit(Ftr[cols], ytr).predict(Fte[cols])))})

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "theory_check.json"
    out.write_text(json.dumps({"dataset": "synth_indep", "n_combos": len(rows), "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    h = [r["hgb"] for r in rows]
    l = [r["ridge"] for r in rows]
    print(f"HGB:   min={min(h):.4f} max={max(h):.4f} разброс={max(h) - min(h):.5f}")
    print(f"Ridge: min={min(l):.4f} max={max(l):.4f} разброс={max(l) - min(l):.4f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
