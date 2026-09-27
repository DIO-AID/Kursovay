"""Единый протокол оценки (одинаков для всех веток).

- 5-fold CV x 3 сида (42, 123, 456) = 15 парных точек на метод.
- FE и ОТБОР выполняются внутри train-части каждого фолда (нет утечки отбора).
- Отобранный набор оценивается тремя классами моделей:
  ridge (линейная), mlp (нейросеть), hgb (градиентный бустинг).
- Гиперпараметры моделей зафиксированы и одинаковы во всех ветках.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .transforms import GroupFE

SEEDS = (42, 123, 456)
N_SPLITS = 5


def make_models(seed):
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "mlp": TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=400,
                learning_rate_init=3e-3, random_state=seed)),
            transformer=StandardScaler()),
        "hgb": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed),
    }


def run_method(dataset_name, loader, method_name, selector, out_dir="results"):
    X, y, truth = loader()
    records = []
    for seed in SEEDS:
        kf = KFold(N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(kf.split(X)):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            fe = GroupFE().fit(Xtr)
            Ftr, Fte = fe.transform(Xtr), fe.transform(Xte)
            t0 = time.perf_counter()
            feats, info = selector(Ftr, y[tr], fe.groups_, seed)
            sel_time = time.perf_counter() - t0
            if not feats:  # защита: пустой набор -> хотя бы один признак
                feats = [Ftr.columns[0]]
            scores = {}
            for mname, model in make_models(seed).items():
                model.fit(Ftr[feats], y[tr])
                scores[mname] = float(r2_score(y[te], model.predict(Fte[feats])))
            records.append({"seed": seed, "fold": fold, "features": list(feats),
                            "n_features": len(feats), "select_time": sel_time,
                            "r2": scores, "info": info})
    out = Path(out_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": dataset_name, "method": method_name, "truth": truth,
               "n_rows": int(len(X)), "n_base": int(X.shape[1]), "records": records}
    (out / f"{method_name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    r2 = {m: np.mean([r["r2"][m] for r in records]) for m in records[0]["r2"]}
    return r2, np.mean([r["n_features"] for r in records])
