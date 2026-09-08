import os
import optuna
import streamlit as st

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import train_test_split


def _safe_fit(model, X_tr, y_tr, fit_params=None):

    if not fit_params:
        return model.fit(X_tr, y_tr)

    params = dict(fit_params)
    model_name = model.__class__.__name__.lower()

    # =====================================================
    # XGBOOST FIX (CRITICAL)
    # =====================================================
    if "xgb" in model_name or "xgboost" in model_name:

        params.pop("verbose", None)

        es = params.pop("early_stopping_rounds", None)

        if es is not None:
            try:
                from xgboost import callback

                params["callbacks"] = [
                    callback.EarlyStopping(rounds=es, save_best=True)
                ]
            except Exception:
                pass

        # 🔥 HARD FILTER — XGBoost sklearn API SAFE PARAMS ONLY
        allowed = model.fit.__code__.co_varnames
        params = {k: v for k, v in params.items() if k in allowed}

        return model.fit(X_tr, y_tr, **params)

    # =====================================================
    # LIGHTGBM
    # =====================================================
    if "lgbm" in model_name:

        params.pop("verbose", None)

        es = params.pop("early_stopping_rounds", None)

        if es is not None:
            try:
                import lightgbm as lgb
                callbacks = params.get("callbacks", [])
                callbacks.append(lgb.early_stopping(es, verbose=False))
                params["callbacks"] = callbacks
            except Exception:
                pass

        return model.fit(X_tr, y_tr, **params)

    # =====================================================
    # CATBOOST
    # =====================================================
    if "catboost" in model_name:

        safe = {}
        allowed = model.fit.__code__.co_varnames

        for k, v in params.items():
            if k in allowed:
                safe[k] = v

        return model.fit(X_tr, y_tr, **safe)

    # =====================================================
    # FALLBACK
    # =====================================================
    return model.fit(X_tr, y_tr, **params)


# =========================================================
# OPTUNA
# =========================================================
def _count_params(search_space_func):
    """Count number of parameters in search space."""
    import optuna
    
    class DummyTrial:
        def __init__(self):
            self.params = {}
        
        def suggest_int(self, name, low, high, log=False, step=1):
            self.params[name] = 0
            return 0
        
        def suggest_float(self, name, low, high, log=False):
            self.params[name] = 0.0
            return 0.0
        
        def suggest_categorical(self, name, choices):
            self.params[name] = choices[0]
            return choices[0]
    
    trial = DummyTrial()
    search_space_func(trial)
    return len(trial.params)


def run_optuna_model(
    model_class,
    search_space_func,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test,
    metric_func,
    study_name,
    direction="maximize",
    n_trials=50,
    timeout=None,
    callback=None,
    X_val=None,
    y_val=None,
):
    if X_val is None or y_val is None:
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
    else:
        X_tr = X_train
        y_tr = y_train

    n_params = _count_params(search_space_func)
    n_startup = max(n_params * 5, 20)

    # =========================
    # UI
    # =========================
    if callback is None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        best_text = st.empty()

        def default_callback(study, trial):
            progress = (trial.number + 1) / n_trials

            progress_bar.progress(min(progress, 1.0))

            score = trial.user_attrs.get("score")
            best = study.best_value if study.best_trial else None

            if score is not None and best is not None:
                status_text.text(f"Trial {trial.number+1} | {score:.5f}")
                best_text.text(f"Best: {best:.5f}")

        callback_to_use = default_callback
    else:
        callback_to_use = callback

    # =========================
    # OBJECTIVE
    # =========================
    def objective(trial):

        params = search_space_func(trial)
        model = model_class(**params)

        try:
            model_name = model.__class__.__name__

            fit_params = {}

            if "CatBoost" in model_name:
                fit_params = {
                    "eval_set": [(X_val, y_val)],
                    "early_stopping_rounds": 50,
                    "verbose": False,
                }

            elif "XGB" in model_name:
                fit_params = {
                    "eval_set": [(X_val, y_val)],
                    "early_stopping_rounds": 50,
                    "verbose": 0,
                }

            elif "LGBM" in model_name:
                fit_params = {
                    "eval_set": [(X_val, y_val)]
                }

            model = _safe_fit(model, X_tr, y_tr, fit_params)

            preds = model.predict(X_val)
            score = metric_func(y_val, preds)

            trial.set_user_attr("score", score)
            trial.report(score, step=trial.number)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return score

        except Exception as e:
            print("Trial error:", e)
            return -1e10 if direction == "maximize" else 1e10

    # =========================
    # STUDY
    # =========================
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=TPESampler(seed=42, multivariate=True, n_startup_trials=n_startup),
        pruner=MedianPruner(n_startup_trials=n_startup, n_warmup_steps=5),
        storage=f"sqlite:///{os.path.join(os.path.dirname(__file__), 'optuna.db')}",
        load_if_exists=True,
    )

    st.info(f"Optuna: {study_name}")

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[callback_to_use]
    )

    # =========================
    # FINAL MODEL
    # =========================
    best_model = model_class(**study.best_params)

    best_model = _safe_fit(
        best_model,
        X_tr,
        y_tr,
        {
            "eval_set": [(X_val, y_val)],
            "early_stopping_rounds": 50,
            "verbose": 0,
        }
    )

    # =========================
    # SAVE BEST MODEL
    # =========================
    import joblib
    model_path = os.path.join(os.path.dirname(__file__), f"best_model_{study_name}.joblib")
    joblib.dump(best_model, model_path)

    return study, best_model