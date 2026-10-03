"""Стенд для гипотез о настройке гиперпараметров (Optuna) — ветки research/hpo-*.

Вопрос: когда настройка Optuna реально улучшает модель на БУДУЩИХ данных, а когда это самообман?

БАЗОВАЯ НАСТРОЙКА (как в tuning/run_optuna_model.py основного проекта) — от неё отличается каждая гипотеза:
  * проверочная выборка V — случайные 20% обучающих строк (random_state=42);
  * ранняя остановка обучения (50 раундов) по той же V, по которой Optuna считает оценку (es_mode="shared");
  * TPE-сэмплер, seed=42, multivariate, стартовых случайных попыток max(5·число параметров, 20);
  * 50 попыток; пространства поиска — как в tuning/search_spaces.py.

Что меняют гипотезы (одна вещь за раз, см. experiments/hpo/*.json):
  n_trials / budgets — число попыток (10, 25, 50, 100): считается ОДНО исследование на max попыток,
                       бюджет k = лучшая из первых k попыток (при том же seed это ровно то, что дал бы запуск на k);
  es_mode            — shared (как в базе) | separate (остановка по отдельной выборке E, оценка по V) |
                       none (без остановки: число деревьев подбирает Optuna);
  val_scheme         — random (как в базе) | time (V = последние 20% обучения по времени, с зазором h).
Служебное: max_rows — взять только последние N строк признаков (быстрая проверка кода, не для результатов).

Внешняя проверка (одинакова для всех): ряд режется на 10 блоков, тест — последние K блоков, обучение — только
строки раньше теста (fsx/forecast.forward_splits). Тест НИКОГДА не участвует в выборе настроек: ошибка на тесте
для каждой попытки записывается только для анализа (связь «оценка на проверке ↔ ошибка на будущем»).

Модели: naive_last, ridge (без настройки), hgb (sklearn, всегда доступен), lightgbm, xgboost, catboost.
Для каждой: «стандартная» (параметры библиотеки по умолчанию, обучение на всём train) и настроенная Optuna.
Настроенная считается двумя способами:
  test_mae       — модель лучшей попытки как есть (обучена на части fit; так делает базовый проект);
  refit_test_mae — те же настройки и найденное число деревьев, но обучение на всём train, как у стандартной
                   (контроль: без него настроенная проигрывает уже потому, что видела меньше данных).
Если пакета нет — модель пропускается с сообщением. Если нет optuna — случайный поиск с пометкой
sampler="random_fallback" (такие результаты годятся только для проверки кода).
"""
import json
import time
import warnings

import numpy as np

from .evaluate import passport
from .forecast import forward_splits
from .lags import LagFE
from .models import make_ridge
from .paths import RESULTS

OUT = RESULTS / "hpo"
ES_ROUNDS = 50
BASE = dict(n_trials=50, budgets=[50], val_scheme="random", es_modes=["shared"], folds=5, seed=42,
            mode="forecast", datasets=["tetouan", "steel", "appliances"],
            models=["hgb", "lightgbm", "xgboost", "catboost"])


# ------------------------------------------------------------------ пространства поиска (= tuning/search_spaces.py)
def xgb_space(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 1000, log=True),
            "max_depth": t.suggest_int("max_depth", 4, 10),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": t.suggest_int("min_child_weight", 1, 10),
            "gamma": t.suggest_float("gamma", 0, 3),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-4, 10, log=True)}


def lgbm_space(t):
    return {"n_estimators": t.suggest_int("n_estimators", 100, 1000, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 10),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": t.suggest_int("num_leaves", 15, 63, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 50),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-4, 10, log=True)}


def catboost_space(t):
    return {"iterations": t.suggest_int("iterations", 200, 800),
            "depth": t.suggest_int("depth", 4, 8),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 5),
            "bagging_temperature": t.suggest_float("bagging_temperature", 0, 1),
            "random_strength": t.suggest_float("random_strength", 0, 5)}


def hgb_space(t):
    return {"max_iter": t.suggest_int("max_iter", 100, 1000, log=True),
            "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 15, 127, log=True),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 50),
            "l2_regularization": t.suggest_float("l2_regularization", 1e-4, 10, log=True),
            "max_features": t.suggest_float("max_features", 0.6, 1.0)}


# ------------------------------------------------------------------ модели: единый интерфейс fit -> (predict, n_iter)
def _has(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


class _HGB:
    """sklearn-бустинг с ранней остановкой по ВНЕШНЕЙ выборке (сам sklearn так не умеет):
    обучение идёт порциями по 25 деревьев (warm_start), остановка — если 50 деревьев нет улучшения."""
    space, n_key, default = staticmethod(hgb_space), "max_iter", {}
    CHUNK = 25

    @staticmethod
    def fit(params, X, y, es, seed, threads):
        from sklearn.ensemble import HistGradientBoostingRegressor
        p = dict(params)
        if es is None:
            m = HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **p).fit(X, y)
            return m.predict, int(m.n_iter_)
        max_iter = int(p.pop("max_iter", 100))
        m = HistGradientBoostingRegressor(early_stopping=False, random_state=seed, warm_start=True,
                                          max_iter=min(_HGB.CHUNK, max_iter), **p)
        best_n, best_e = 0, float("inf")
        while True:
            m.fit(X, y)
            e = float(np.sqrt(np.mean((es[1] - m.predict(es[0])) ** 2)))
            if e < best_e:
                best_n, best_e = m.n_iter_, e
            if m.n_iter_ >= max_iter or m.n_iter_ - best_n >= ES_ROUNDS:
                break
            m.set_params(max_iter=min(m.n_iter_ + _HGB.CHUNK, max_iter))

        def predict(Z, k=best_n):
            if k >= m.n_iter_:
                return m.predict(Z)
            for i, pr in enumerate(m.staged_predict(Z), 1):
                if i == k:
                    return pr
        return predict, int(best_n)


class _LGBM:
    space, n_key, default = staticmethod(lgbm_space), "n_estimators", {}

    @staticmethod
    def fit(params, X, y, es, seed, threads):
        import lightgbm as lgb
        p = dict(params)
        if p.get("subsample", 1.0) < 1.0:
            p.setdefault("subsample_freq", 1)            # без этого subsample в LightGBM не действует
        m = lgb.LGBMRegressor(**p, random_state=seed, n_jobs=threads, verbose=-1)
        if es is None:
            m.fit(X, y)
            return m.predict, int(m.n_estimators_)
        m.fit(X, y, eval_set=[es], callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
        return m.predict, int(m.best_iteration_ or m.n_estimators_)


class _XGB:
    space, n_key, default = staticmethod(xgb_space), "n_estimators", {}

    @staticmethod
    def fit(params, X, y, es, seed, threads):
        from xgboost import XGBRegressor
        if es is None:
            m = XGBRegressor(**params, random_state=seed, n_jobs=threads).fit(X, y)
            return m.predict, int(m.get_booster().num_boosted_rounds())
        m = XGBRegressor(**params, random_state=seed, n_jobs=threads, early_stopping_rounds=ES_ROUNDS)
        m.fit(X, y, eval_set=[es], verbose=False)
        return m.predict, int(m.best_iteration) + 1


class _CatBoost:
    space, n_key, default = staticmethod(catboost_space), "iterations", {}

    @staticmethod
    def fit(params, X, y, es, seed, threads):
        from catboost import CatBoostRegressor
        p = dict(params)
        if "bagging_temperature" in p:
            p.setdefault("bootstrap_type", "Bayesian")   # bagging_temperature действует только с Bayesian
        m = CatBoostRegressor(**p, loss_function="RMSE", random_seed=seed, verbose=0,
                              thread_count=threads, allow_writing_files=False)
        if es is None:
            m.fit(X, y)
            return m.predict, int(m.tree_count_)
        m.fit(X, y, eval_set=es, early_stopping_rounds=ES_ROUNDS, use_best_model=True)
        return m.predict, int(m.tree_count_)


MODELS = {"hgb": (_HGB, "sklearn"), "lightgbm": (_LGBM, "lightgbm"), "xgboost": (_XGB, "xgboost"),
          "catboost": (_CatBoost, "catboost")}


def available_models(names):
    ok, missing = [], []
    for n in names:
        (ok if _has(MODELS[n][1]) else missing).append(n)
    return ok, missing


# ------------------------------------------------------------------ сэмплер
class _RandomTrial:
    """Замена optuna.Trial для случайного поиска (когда optuna не установлена)."""

    def __init__(self, rng):
        self.rng, self.params = rng, {}

    def suggest_int(self, name, low, high, log=False, step=1):
        v = int(round(np.exp(self.rng.uniform(np.log(low), np.log(high))))) if log else int(self.rng.integers(low, high + 1))
        self.params[name] = int(min(max(v, low), high))
        return self.params[name]

    def suggest_float(self, name, low, high, log=False):
        v = float(np.exp(self.rng.uniform(np.log(low), np.log(high)))) if log else float(self.rng.uniform(low, high))
        self.params[name] = v
        return v


def _n_params(space):
    t = _RandomTrial(np.random.default_rng(0))
    space(t)
    return len(t.params)


def run_study(space, objective, n_trials, seed, progress=None):
    """Запускает n_trials попыток; objective(params) -> dict с ключом 'val_mae'. Возвращает (попытки, имя сэмплера)."""
    trials = []
    if _has("optuna"):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_startup = max(_n_params(space) * 5, 20)                       # как в базовой настройке проекта
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(
            seed=seed, multivariate=True, n_startup_trials=n_startup))

        def obj(trial):
            params = space(trial)
            r = objective(params)
            trials.append({"number": trial.number, "params": params, **r})
            if progress:
                progress(trial.number + 1, n_trials)
            return r["val_mae"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(obj, n_trials=n_trials)
        return trials, "optuna_tpe"
    rng = np.random.default_rng(seed)
    for i in range(n_trials):
        t = _RandomTrial(rng)
        params = space(t)
        trials.append({"number": i, "params": params, **objective(params)})
        if progress:
            progress(i + 1, n_trials)
    return trials, "random_fallback"


# ------------------------------------------------------------------ разбиения внутри обучения
def inner_split(n_train, val_scheme, es_mode, h, seed=42):
    """Индексы (fit, es, val) внутри обучающей части фолда.
    val   — по ней Optuna считает оценку; fit — на этом учится модель; es — ранняя остановка (или None).
    shared: es = val (как в базе).  separate: es — отдельные 20% от fit.  none: es = None."""
    idx = np.arange(n_train)
    if val_scheme == "random":
        perm = np.random.default_rng(seed).permutation(n_train)
        val, rest = np.sort(perm[:int(0.2 * n_train)]), np.sort(perm[int(0.2 * n_train):])
    elif val_scheme == "time":
        cut = int(0.8 * n_train)
        val, rest = idx[cut:], idx[:cut - h]                 # зазор h: признаки val не пересекаются с целями fit
    else:
        raise ValueError("val_scheme: random | time")
    if es_mode == "shared":
        return rest, val, val
    if es_mode == "none":
        return rest, None, val
    if es_mode == "separate":
        if val_scheme == "random":
            p2 = np.random.default_rng(seed + 1).permutation(len(rest))
            k = int(0.2 * len(rest))
            return np.sort(rest[p2[k:]]), np.sort(rest[p2[:k]]), val
        k = int(0.8 * len(rest))
        return rest[:k - h], rest[k:], val                    # es — кусок по времени перед val
    raise ValueError("es_mode: shared | separate | none")


def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def summarize_budget(trials, k):
    """Что выбрала бы настройка на k попытках: лучшая по проверке из первых k."""
    first = [t for t in trials if t["number"] < k]
    best = min(first, key=lambda t: t["val_mae"])
    return {"budget": k, "best_trial": best["number"], "val_mae": best["val_mae"], "test_mae": best["test_mae"],
            "n_iter": best["n_iter"], "oracle_test_mae": min(t["test_mae"] for t in first),
            "params": best["params"]}


# ------------------------------------------------------------------ один датасет
def load_features(name, mode="forecast"):
    from .data import TIME_DATASETS, registry_dataset
    ds = TIME_DATASETS[name]() if name in TIME_DATASETS else registry_dataset(name)
    fe = LagFE(ds.meta.get("horizon") or "1h", mode, ds.meta.get("known_ahead", []), ds.meta.get("cat_cols", []))
    F = fe.fit_transform(ds.X, ds.y, ds.t).iloc[fe.warmup_:].reset_index(drop=True)
    y = np.asarray(ds.y, float)[fe.warmup_:]
    return ds, fe, F, y


def run_experiment(cfg, progress=None, threads=-1, out_dir=None):
    """cfg — словарь как в experiments/hpo/*.json (недостающее берётся из BASE). Пишет JSON по каждому
    (датасет, модель, es_mode, фолд) и возвращает список кратких итогов."""
    cfg = {**BASE, **cfg}
    name = cfg.get("name", "base")
    models, missing = available_models(cfg["models"])
    out_root = (out_dir or OUT) / name
    brief = []
    for dname in cfg["datasets"]:
        ds, fe, F, y = load_features(dname, cfg["mode"])
        if cfg.get("max_rows"):                             # только для быстрых проверок кода
            F, y = F.iloc[-cfg["max_rows"]:].reset_index(drop=True), y[-cfg["max_rows"]:]
        X = F.to_numpy(dtype=float)                         # массивы: имена столбцов с пробелами ломают LightGBM
        splits, gap = forward_splits(len(F), fe.h_, 10)
        splits = splits[-cfg["folds"]:]
        lag = F[f"target.short__lag{fe.h_}"].to_numpy()
        for fold, tr, te in splits:
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            ref = {"naive_last": _mae(yte, np.where(np.isnan(lag[te]), np.nanmean(ytr), lag[te])),
                   "ridge": _mae(yte, make_ridge().fit(Xtr, ytr).predict(Xte))}
            for mname in models:
                M = MODELS[mname][0]
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred, n_def = M.fit(dict(M.default), Xtr, ytr, None, cfg["seed"], threads)   # стандартная
                default = {"test_mae": _mae(yte, pred(Xte)), "n_iter": n_def, "fit_time": time.perf_counter() - t0}
                for es_mode in cfg["es_modes"]:
                    i_fit, i_es, i_val = inner_split(len(tr), cfg["val_scheme"], es_mode, fe.h_, cfg["seed"])

                    def objective(params, i_fit=i_fit, i_es=i_es, i_val=i_val, M=M):
                        t1 = time.perf_counter()
                        es = None if i_es is None else (Xtr[i_es], ytr[i_es])
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pr, n_iter = M.fit(dict(params), Xtr[i_fit], ytr[i_fit], es, cfg["seed"], threads)
                        return {"val_mae": _mae(ytr[i_val], pr(Xtr[i_val])),
                                "test_mae": _mae(yte, pr(Xte)),          # только для анализа, в выборе не участвует
                                "n_iter": n_iter, "fit_time": time.perf_counter() - t1}

                    def prog(i, n, what=f"{dname} {mname} {es_mode} фолд {fold}"):
                        if progress:
                            progress(what, i, n)
                    t2 = time.perf_counter()
                    trials, sampler = run_study(M.space, objective, cfg["n_trials"], cfg["seed"], prog)
                    budgets = [summarize_budget(trials, k) for k in cfg["budgets"] if k <= len(trials)]
                    refit = {}                              # лучшая попытка -> ошибка после обучения на всём train
                    for b in budgets:
                        if b["best_trial"] not in refit:
                            p = dict(b["params"])
                            if i_es is not None:            # число деревьев — то, что нашла ранняя остановка
                                p[M.n_key] = max(1, int(b["n_iter"]))
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                pr, _ = M.fit(p, Xtr, ytr, None, cfg["seed"], threads)
                            refit[b["best_trial"]] = _mae(yte, pr(Xte))
                        b["refit_test_mae"] = refit[b["best_trial"]]
                    payload = {"experiment": name, "dataset": dname, "model": mname, "es_mode": es_mode,
                               "val_scheme": cfg["val_scheme"], "fold": int(fold), "sampler": sampler,
                               "n_train": int(len(tr)), "n_test": int(len(te)), "gap": int(gap),
                               "n_fit": int(len(i_fit)), "n_val": int(len(i_val)),
                               "n_es": 0 if i_es is None else int(len(i_es)), "n_features": int(X.shape[1]),
                               "reference": ref, "default": default, "budgets": budgets, "trials": trials,
                               "study_time": time.perf_counter() - t2, "config": cfg, "lagfe": fe.info(),
                               "passport": passport(ds)}
                    d = out_root / dname
                    d.mkdir(parents=True, exist_ok=True)
                    (d / f"{mname}__{es_mode}__fold{fold}.json").write_text(
                        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
                    last = budgets[-1]
                    brief.append({"dataset": dname, "model": mname, "es_mode": es_mode, "fold": int(fold),
                                  "default": default["test_mae"], "tuned": last["test_mae"],
                                  "refit": last["refit_test_mae"],
                                  "val": last["val_mae"], "sampler": sampler})
    return brief, missing
