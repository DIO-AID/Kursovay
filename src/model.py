from .models_library import MODEL_LIBRARY
from .gpu_utils import get_gpu_params


def build_model(model_name="xgb", **kwargs):
    model_name = model_name.lower()

    if model_name not in MODEL_LIBRARY:
        raise ValueError(f"Неизвестная модель: {model_name}")

    model_class = MODEL_LIBRARY[model_name]

    base_model = model_class()
    base_params = base_model.get_params()

    gpu_params = get_gpu_params(model_name=model_name)

    safe_gpu_params = {
        k: v for k, v in gpu_params.items()
        if k in base_params
    }

    default_params = {}

    if model_name == "xgb":
        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "random_state": 42
        }

    elif model_name == "lgbm":
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42
        }

    elif model_name == "catboost":
        default_params = {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "verbose": False,
            "random_seed": 42
        }

    elif model_name == "rf":
        default_params = {
            "n_estimators": 200,
            "max_depth": None,
            "random_state": 42
        }

    elif model_name == "linear":
        default_params = {}

    model = model_class(
        **default_params,
        **safe_gpu_params,
        **kwargs
    )

    model._model_name = model_name
    model._is_fitted = False

    return model


def incremental_fit(model, X, y):
    model_name = getattr(model, "_model_name", "")

    if not getattr(model, "_is_fitted", False):
        model.fit(X, y)
        model._is_fitted = True

        if model_name == "xgb":
            model._booster = model.get_booster()

        elif model_name == "lgbm":
            model._booster = model.booster_

        elif model_name == "catboost":
            model._booster = model

    else:
        try:
            if model_name == "xgb":
                model.fit(X, y, xgb_model=model._booster)
                model._booster = model.get_booster()

            elif model_name == "lgbm":
                model.fit(X, y, init_model=model._booster)
                model._booster = model.booster_

            elif model_name == "catboost":
                model.fit(X, y, init_model=model._booster)
                model._booster = model

            else:
                model.fit(X, y)

        except Exception as e:
            print(f"[WARNING] incremental failed → fallback: {e}")
            model.fit(X, y)

    return model


def get_model_and_space(model_name):
    """Returns model class and search space function for Optuna."""
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from sklearn.ensemble import RandomForestRegressor

    from tuning.search_spaces import (
        xgb_space,
        lgbm_space,
        catboost_space,
        rf_space,
    )

    model_map = {
        "xgb": (XGBRegressor, xgb_space),
        "xgboost": (XGBRegressor, xgb_space),
        "lgbm": (LGBMRegressor, lgbm_space),
        "lightgbm": (LGBMRegressor, lgbm_space),
        "catboost": (CatBoostRegressor, catboost_space),
        "rf": (RandomForestRegressor, rf_space),
        "random_forest": (RandomForestRegressor, rf_space),
    }

    if model_name.lower() not in model_map:
        raise ValueError(f"Unknown model: {model_name}")

    return model_map[model_name.lower()]