"""Проверка H0: монотонная трансформация не меняет разбиения дерева.

Для каждой комбинации модификаций (x1, x2, x3, x4) обучаем бустинг и Ridge
на ОДНОМ представителе каждого фактора. Если H0 верна, R2 бустинга не зависит
от комбинации, а R2 Ridge — зависит.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fsx.data import synth_indep
from fsx.transforms import GroupFE

X, y, truth = synth_indep()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
fe = GroupFE().fit(Xtr)
Ftr, Fte = fe.transform(Xtr), fe.transform(Xte)
rng = np.random.default_rng(0)
combos = list(itertools.product(["raw", "log", "sqrt", "sq", "inv"], repeat=4))
sample = [tuple(truth[f"x{i}"] for i in range(1, 5))] + \
         [combos[i] for i in rng.choice(len(combos), 39, replace=False)]
rows = []
for c in sample:
    cols = [f"x{i+1}__{t}" for i, t in enumerate(c)]
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    rid = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    rows.append({"combo": list(c), "is_truth": c == sample[0],
                 "hgb": r2_score(yte, hgb.fit(Ftr[cols], ytr).predict(Fte[cols])),
                 "ridge": r2_score(yte, rid.fit(Ftr[cols], ytr).predict(Fte[cols]))})
Path("results").mkdir(exist_ok=True)
json.dump(rows, open("results/theory_check.json", "w"), indent=1)
h = [r["hgb"] for r in rows]; l = [r["ridge"] for r in rows]
print(f"HGB:   min={min(h):.4f} max={max(h):.4f} разброс={max(h)-min(h):.5f}")
print(f"Ridge: min={min(l):.4f} max={max(l):.4f} разброс={max(l)-min(l):.4f}")
