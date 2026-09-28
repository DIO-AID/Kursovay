"""Шаги 2, 4, 5, 6 конвейера (одинаковы для всех веток, кроме подменяемого отбора).

Шаг 2. Разбиение:
  - обычные данные: 5-fold CV x 3 сида (42, 123, 456) = 15 парных точек;
  - временные данные (Dataset.time=True): ряд режется на N_BLOCKS блоков подряд; тест —
    блоки 2..N, обучение — ТОЛЬКО строки раньше теста минус зазор GAP_FRAC (скользящее
    "прошлое -> будущее", будущее в обучение не попадает) -> N_BLOCKS-1 пар.
Шаг 4. Отбор — только по числовым группам train-фолда; категориальные 0/1 добавляются как есть.
        select_sample: медленным методам можно дать случайную подвыборку train (фиксируется
        в records[].select_rows); модели шага 5 всё равно учатся на всём train.
        Пустой отбор не подменяется молча: records[].empty_selection = True, модель получает
        одну константу-заглушку (R2 ~ 0), и это видно в отчёте.
Шаг 5. Три модели с ЗАМОРОЖЕННЫМИ настройками: ridge, mlp, hgb.
Шаг 6. Оценка на test: R2, MAE, RMSE, время обучения -> results/raw/<датасет>/<метод>.json
"""
import json
import platform
import subprocess
import time
import warnings
from datetime import datetime

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .paths import RAW, ROOT
from .transforms import GroupFE

SEEDS = (42, 123, 456)
N_SPLITS = 5
N_BLOCKS = 10
GAP_FRAC = 0.01


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


def make_splits(n, is_time):
    """Возвращает список (seed, fold, train_idx, test_idx)."""
    out = []
    if is_time:
        idx = np.arange(n)
        gap = max(1, int(n * GAP_FRAC))
        for b, te in enumerate(np.array_split(idx, N_BLOCKS)):
            if b == 0:
                continue                      # для первого блока нет прошлого
            tr = idx[idx < te[0] - gap]
            out.append((0, b, tr, te))
    else:
        for seed in SEEDS:
            for f, (tr, te) in enumerate(KFold(N_SPLITS, shuffle=True, random_state=seed).split(np.arange(n))):
                out.append((seed, f, tr, te))
    return out


def _slim(obj, limit=20):
    """Убирает длинные списки (кривые и т.п.) — полные данные храним только для первого фолда."""
    if isinstance(obj, dict):
        return {k: _slim(v, limit) for k, v in obj.items()
                if not (isinstance(v, list) and len(v) > limit)}
    return obj


def passport(ds):
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "fsx", "scripts"],
                                    cwd=ROOT, capture_output=True, text=True,
                                    timeout=5).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return {"created": datetime.now().isoformat(timespec="seconds"), "git_commit": commit, "git_dirty": dirty,
            "python": platform.python_version(),
            "packages": {m: ver(m) for m in ("sklearn", "numpy", "pandas", "scipy")},
            "split": f"forward_time{N_BLOCKS - 1}" if ds.time else f"kfold{N_SPLITS}x{len(SEEDS)}",
            "dataset_meta": ds.meta}


def run_method(dataset_name, loader, method_name, selector, progress=None, select_sample=None):
    ds = loader()
    X, y = ds.X, ds.y
    splits = make_splits(len(X), ds.time)
    records = []
    for i, (seed, fold, tr, te) in enumerate(splits):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        fe = GroupFE(ds.meta.get("cat_cols", ())).fit(Xtr)
        Ftr, Fte = fe.transform(Xtr), fe.transform(Xte)
        sel_idx = np.arange(len(tr))
        if select_sample and len(tr) > select_sample:
            sel_idx = np.sort(np.random.default_rng(seed or 42).choice(len(tr), select_sample, replace=False))
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats, info = selector(Ftr[fe.num_features_].iloc[sel_idx], y[tr][sel_idx],
                                   fe.groups_, seed or 42)
        sel_time = time.perf_counter() - t0
        feats = list(feats) + fe.cat_features_
        empty = not feats
        if empty:                                  # честная заглушка вместо чужого признака
            Ftr = Ftr.assign(__const=0.0)
            Fte = Fte.assign(__const=0.0)
            feats = ["__const"]
        metrics = {}
        for mname, model in make_models(seed or 42).items():
            t1 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Ftr[feats], y[tr])
            fit_time = time.perf_counter() - t1
            p = model.predict(Fte[feats])
            metrics[mname] = {"r2": float(r2_score(y[te], p)),
                              "mae": float(mean_absolute_error(y[te], p)),
                              "rmse": float(np.sqrt(mean_squared_error(y[te], p))),
                              "fit_time": fit_time}
        records.append({"seed": seed, "fold": fold, "features": [] if empty else feats,
                        "n_features": 0 if empty else len(feats), "empty_selection": empty,
                        "select_rows": int(len(sel_idx)), "n_train": int(len(tr)),
                        "n_test": int(len(te)), "select_time": sel_time, "metrics": metrics,
                        "info": info if i == 0 else _slim(info)})
        if progress:
            progress(i + 1, len(splits))
    out = RAW / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": dataset_name, "method": method_name, "truth": ds.truth,
               "n_rows": int(len(X)), "n_base": int(X.shape[1]), "passport": passport(ds),
               "select_sample": select_sample, "records": records}
    (out / f"{method_name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    r2 = {m: float(np.mean([r["metrics"][m]["r2"] for r in records])) for m in records[0]["metrics"]}
    return r2, float(np.mean([r["n_features"] for r in records]))
