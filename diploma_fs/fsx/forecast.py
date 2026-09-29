"""Конвейер прогноза энергопотребления (ветка research/forecast, ADR-001).

Те же 7 шагов, что и в основном стенде, но шаги 3–5 другие:
  1. данные: временной ряд (Dataset.t задано) — steel, tetouan, seoul_bike, synth_factory;
  2. разбиение: ряд режется на N_BLOCKS блоков; тест — блоки 2..N; обучение — только строки
     РАНЬШЕ теста минус зазор max(1% ряда, h);
  3. признаки: LagFE (fsx/lags.py), один раз на весь ряд — это только сдвиги, утечки нет;
     первые warmup строк (где ещё нет недельного лага) выбрасываются для ВСЕХ наборов, чтобы
     все сравнивались на одних и тех же строках; перед прогоном — тест на утечку (check_no_leak);
  4. отбор: методы из fsx/selectors на train-фолде (пропуски -> медиана train, т.к. sklearn-
     методы не понимают NaN); select_sample — подвыборка ПОСЛЕДНИХ строк train;
  5. модели: naive_last / naive_day / ridge / catboost (fsx/models.py);
  6. метрики: R², MAE, RMSE, MAPE, WAPE, время, число признаков;
  7. статистика: scripts/report_forecast.py.

«Методы» в результатах (одно имя = один JSON в results/forecast/<mode>/raw/<датасет>/):
  naive            — наивные прогнозы (E1: нижняя планка);
  fs_no_lags       — без прошлых значений: календарь и известные заранее колонки
                     (в режиме virtual ещё и текущие показания датчиков);
  fs_all           — все временные признаки, без отбора (опорный набор для E2–E4);
  abl_no_<семья>   — все, кроме одного семейства (E2: calendar/short/daily/rolling/exog);
  sel_<метод>      — отбор методом из fsx/selectors по набору fs_all (E3–E4).
"""
import json
import time
import warnings

import numpy as np

from .evaluate import passport
from .lags import FAMILIES, LagFE, check_no_leak
from .models import fit_predict, make_models, metrics, naive_predictions, tune_catboost
from .paths import RESULTS
from .selectors import REGISTRY

N_BLOCKS = 10
GAP_FRAC = 0.01
DEFAULT_SELECTORS = ("cb_importance", "cb_permutation", "cb_shap", "rfe", "boruta",
                     "sysoev_paper", "sysoev_fixed", "group_importance")
OUT = RESULTS / "forecast"


def forward_splits(n, h, n_blocks=N_BLOCKS):
    idx = np.arange(n)
    gap = max(h, int(n * GAP_FRAC))
    out = []
    for b, te in enumerate(np.array_split(idx, n_blocks)):
        if b == 0:
            continue
        tr = idx[idx < te[0] - gap]
        if len(tr) >= 50:
            out.append((b, tr, te))
    return out, gap


def feature_sets(fe, cols, mode):
    fam = fe.families(cols)
    no_lags = list(fam["calendar"])
    if mode == "virtual":
        no_lags += [c for c in fam["exog"] if c.endswith("__cur") or "__curcat_" in c]
    sets = {"fs_no_lags": no_lags, "fs_all": list(cols)}
    for f in FAMILIES:
        if fam[f]:
            sets[f"abl_no_{f}"] = [c for c in cols if c not in set(fam[f])]
    return sets


def _jaccard_mean(sets):
    s = [set(x) for x in sets]
    if len(s) < 2:
        return float("nan")
    v = [len(a & b) / len(a | b) if a | b else 1.0 for i, a in enumerate(s) for b in s[i + 1:]]
    return float(np.mean(v))


def run_forecast(name, ds, mode="forecast", horizon=None, selectors=DEFAULT_SELECTORS,
                 with_sets=True, select_sample=None, tune=0, n_blocks=N_BLOCKS, progress=None,
                 out_dir=None):
    if ds.t is None:
        raise ValueError(f"{name}: нет времени (Dataset.t) — это не временной ряд")
    horizon = horizon or ds.meta.get("horizon") or "1h"
    fe = LagFE(horizon, mode, ds.meta.get("known_ahead", []), ds.meta.get("cat_cols", []))
    F_all = fe.fit_transform(ds.X, ds.y, ds.t)
    n_checked = check_no_leak(fe, ds.X, ds.y, ds.t, n_points=10)
    keep = np.arange(fe.warmup_, len(F_all))
    if len(keep) < 500:
        raise ValueError(f"{name}: после разогрева ({fe.warmup_} шагов) осталось {len(keep)} строк")
    F = F_all.iloc[keep].reset_index(drop=True)
    y = np.asarray(ds.y, float)[keep]
    t = ds.t.iloc[keep].reset_index(drop=True)
    h = fe.h_
    lag_last = f"target.short__lag{h}"
    lag_day = f"target.daily__lag{fe.daily_lags_[0]}"
    sets = feature_sets(fe, list(F.columns), mode) if with_sets else {"fs_all": list(F.columns)}
    splits, gap = forward_splits(len(F), h, n_blocks)
    unknown = [s for s in selectors if s not in REGISTRY]
    if unknown:
        raise ValueError(f"Нет методов отбора {unknown}. Есть: {sorted(REGISTRY)}")
    groups = fe.groups(F.columns)

    records = {"naive": []}
    records.update({k: [] for k in sets})
    records.update({f"sel_{s}": [] for s in selectors})
    cb_params, tune_info = None, {"tuned": False, "reason": "не запрошено"}
    total = len(splits) * (1 + len(sets) + len(selectors))
    step = 0

    def tick(what):
        nonlocal step
        step += 1
        if progress:
            progress(step, total, what)

    for fold, tr, te in splits:
        ytr, yte = y[tr], y[te]
        base = {"fold": int(fold), "n_train": int(len(tr)), "n_test": int(len(te)),
                "test_from": str(t[te[0]]), "test_to": str(t[te[-1]])}
        if tune and cb_params is None:
            cb_params, tune_info = tune_catboost(F.iloc[tr], ytr, 42, h, n_trials=tune)
        # наивные
        nv = naive_predictions(F.iloc[te], lag_last, lag_day)
        records["naive"].append({**base, "features": [], "n_features": 0,
                                 "metrics": {k: metrics(yte, p) for k, p in nv.items()}})
        tick("naive")
        # наборы признаков (E1, E2)
        for sname, cols in sets.items():
            records[sname].append({**base, "features": cols if sname == "fs_all" else [],
                                   "n_features": len(cols),
                                   "metrics": _fit_models(F, cols, tr, te, ytr, yte, cb_params)})
            tick(sname)
        # отбор (E3, E4) — по набору fs_all
        med = F.iloc[tr].median()
        Ftr_sel = F.iloc[tr].fillna(med).fillna(0.0)
        ysel = ytr
        if select_sample and len(tr) > select_sample:
            Ftr_sel, ysel = Ftr_sel.iloc[-select_sample:], ytr[-select_sample:]
        for s in selectors:
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feats, info = REGISTRY[s](Ftr_sel, ysel, groups, 42)
            sel_time = time.perf_counter() - t0
            feats = list(feats)
            empty = not feats
            if empty:          # честная заглушка: прогноз средним train (метод ничего не выбрал)
                const = np.full(len(te), float(np.mean(ytr)))
                m = {k: {**metrics(yte, const), "backend": "empty_stub"} for k in ("ridge", "catboost")}
            else:
                m = _fit_models(F, feats, tr, te, ytr, yte, cb_params)
            records[f"sel_{s}"].append({**base, "features": feats, "n_features": len(feats),
                                        "empty_selection": empty, "select_time": sel_time,
                                        "select_rows": int(len(Ftr_sel)), "metrics": m,
                                        "info": _short(info)})
            tick(f"sel_{s}")

    out = (out_dir or OUT / mode / "raw") / name
    out.mkdir(parents=True, exist_ok=True)
    pas = passport(ds)
    pas["split"] = f"forward_time{len(splits)}_gap{gap}"
    common = {"dataset": name, "mode": mode, "horizon": str(horizon), "n_rows": int(len(F)), "n_all_features": F.shape[1],
              "lagfe": fe.info(), "leak_check_points": n_checked, "gap": gap,
              "select_sample": select_sample, "catboost_params": cb_params, "tuning": tune_info,
              "passport": pas}
    summary = {}
    for mname, recs in records.items():
        stab = _jaccard_mean([r["features"] for r in recs]) if mname.startswith("sel_") else None
        payload = {**common, "method": mname, "stability": stab, "records": recs}
        (out / f"{mname}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                                                      default=str), encoding="utf-8")
        summary[mname] = _mean_mae(recs)
    return summary, fe.info()


def _fit_models(F, cols, tr, te, ytr, yte, cb_params):
    res = {}
    for mname, (model, backend) in make_models(42, cb_params).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, ft = fit_predict(model, F.iloc[tr][cols], ytr, F.iloc[te][cols])
        res[mname] = {**metrics(yte, p, ft), "backend": backend}
    return res


def _short(info, limit=30):
    if isinstance(info, dict):
        return {k: _short(v, limit) for k, v in info.items()
                if not (isinstance(v, (list, dict)) and len(v) > limit)}
    return info


def _mean_mae(recs):
    out = {}
    for r in recs:
        for m, v in r["metrics"].items():
            out.setdefault(m, []).append(v["mae"])
    return {m: float(np.mean(v)) for m, v in out.items()}
