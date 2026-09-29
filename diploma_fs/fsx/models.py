"""Шаг 5 (ветка research/forecast): модели прогноза.

  naive_last — «как час назад»: последнее известное значение цели (лаг h);
  naive_day  — «как вчера в это же время» (сезонный наивный прогноз);
  ridge      — линейная модель (пропуски -> медиана train, стандартизация, RidgeCV);
  catboost   — CatBoostRegressor, 500 деревьев, CPU. Если пакет catboost не установлен,
               вместо него честно берётся HistGradientBoosting, и это пишется в результаты
               (поле backend), чтобы не выдавать одно за другое.

Настройка CatBoost (tune_catboost, Optuna, <= 30 попыток) — ТОЛЬКО на train: последние 20%
train служат валидацией (с зазором h), ранняя остановка по ней. В конвейере настройка
делается один раз на train ПЕРВОГО фолда (самые ранние данные) и замораживается:
так будущие блоки теста не влияют на параметры, а счёт укладывается в ноутбук.
"""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CB_DEFAULT = dict(iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0)


def has_catboost():
    try:
        import catboost  # noqa: F401
        return True
    except Exception:
        return False


def has_optuna():
    try:
        import optuna  # noqa: F401
        return True
    except Exception:
        return False


def make_ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-3, 3, 13)))


def make_catboost(seed, params=None, threads=-1):
    """(модель, backend). params — из tune_catboost или CB_DEFAULT."""
    p = dict(CB_DEFAULT, **(params or {}))
    if has_catboost():
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**p, loss_function="RMSE", random_seed=seed, verbose=0,
                                 thread_count=threads, allow_writing_files=False), "catboost"
    return HistGradientBoostingRegressor(max_iter=p["iterations"], learning_rate=p["learning_rate"],
                                         max_depth=p["depth"], l2_regularization=p["l2_leaf_reg"],
                                         random_state=seed), "hgb_fallback"


def make_models(seed, cb_params=None):
    """Модели, которые учатся на признаках (наивные считаются отдельно, им признаки не нужны)."""
    cb, backend = make_catboost(seed, cb_params)
    return {"ridge": (make_ridge(), "sklearn"), "catboost": (cb, backend)}


def naive_predictions(F, lag_last, lag_day):
    """Наивные прогнозы из уже построенных лагов (NaN -> заменяем последним лагом)."""
    last = F[lag_last].to_numpy()
    day = F[lag_day].to_numpy()
    day = np.where(np.isnan(day), last, day)
    return {"naive_last": last, "naive_day": day}


def metrics(y, p, fit_time=0.0):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    nz = np.abs(y) > 1e-6
    return {"r2": float(r2_score(y, p)),
            "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            # MAPE считается только по ненулевым y (у стали бывают нулевые интервалы),
            # WAPE = сумма |ошибок| / сумма |y| — устойчивее к нулям
            "mape": float(np.mean(np.abs((y[nz] - p[nz]) / y[nz]))) if nz.any() else float("nan"),
            "wape": float(np.sum(np.abs(y - p)) / max(np.sum(np.abs(y)), 1e-12)),
            "n_eval": int(ok.sum()), "fit_time": float(fit_time)}


def fit_predict(model, Ftr, ytr, Fte):
    t0 = time.perf_counter()
    model.fit(Ftr, ytr)
    return model.predict(Fte), time.perf_counter() - t0


def tune_catboost(Ftr, ytr, seed, h, n_trials=30, timeout=None, log=None):
    """Optuna по train: валидация — последние 20% train (после зазора h шагов).
    Возвращает (params, info). Без optuna/catboost — CB_DEFAULT и причина в info."""
    if not (has_optuna() and has_catboost()):
        return dict(CB_DEFAULT), {"tuned": False,
                                  "reason": "нет optuna" if not has_optuna() else "нет catboost"}
    import optuna
    from catboost import CatBoostRegressor
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n = len(Ftr)
    cut = int(n * 0.8)
    Xa, ya = Ftr.iloc[:cut - h], ytr[:cut - h]
    Xv, yv = Ftr.iloc[cut:], ytr[cut:]

    def objective(trial):
        p = {"depth": trial.suggest_int("depth", 4, 8),
             "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
             "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)}
        m = CatBoostRegressor(iterations=CB_DEFAULT["iterations"], **p, loss_function="RMSE",
                              random_seed=seed, verbose=0, allow_writing_files=False,
                              od_type="Iter", od_wait=50)
        m.fit(Xa, ya, eval_set=(Xv, yv), use_best_model=True)
        trial.set_user_attr("best_iter", int(m.get_best_iteration() or CB_DEFAULT["iterations"]))
        return float(mean_absolute_error(yv, m.predict(Xv)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   callbacks=[(lambda s, tr: log(tr.number + 1, n_trials))] if log else None)
    best = dict(study.best_params)
    # число деревьев: лучшее по ранней остановке с запасом (на полном train данных больше)
    best["iterations"] = int(min(2000, max(100, study.best_trial.user_attrs["best_iter"] * 1.2)))
    return best, {"tuned": True, "n_trials": len(study.trials), "val_mae": study.best_value}
