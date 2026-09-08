# MODELS LIBRARY (SAFE + OPTUNA READY + SHAP READY)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


# XGBOOST
def get_xgb_model(**params):
    default_params = {
        "tree_method": "hist",
        "predictor": "cpu_predictor",
    }
    default_params.update(params)
    return XGBRegressor(objective="reg:squarederror", **default_params)


# LIGHTGBM
def get_lgbm_model(**params):
    default_params = {
        "device": "cpu",
        "verbose": -1
    }
    # early_stopping_rounds убираем из дефолтов
    default_params.update(params)
    return LGBMRegressor(random_state=42, **default_params)


# CATBOOST
def get_catboost_model(**params):
    default_params = {
        "task_type": "CPU",  # Использовать CPU для совместимости (GPU требует CUDA)
        "thread_count": -1,  # Все ядра CPU
    }
    # od_type и od_wait только если нужен overfitting detector
    # они могут замедлять обучение
    default_params.update(params)
    return CatBoostRegressor(verbose=False, random_state=42, **default_params)


# RANDOM FOREST
def get_rf_model(**params):
    return RandomForestRegressor(random_state=42, **params)


# EXTRA TREES
def get_extra_trees_model(**params):
    return ExtraTreesRegressor(random_state=42, **params)


# GRADIENT BOOSTING
def get_gb_model(**params):
    return GradientBoostingRegressor(random_state=42, **params)


# HIST GRADIENT BOOSTING
def get_hist_gb_model(**params):
    return HistGradientBoostingRegressor(random_state=42, **params)


# LINEAR MODELS
def get_linear_model(**params):
    return LinearRegression(**params)


def get_ridge_model(**params):
    return Ridge(random_state=42, **params)


def get_lasso_model(**params):
    return Lasso(max_iter=3000, random_state=42, **params)


def get_elastic_model(**params):
    return ElasticNet(max_iter=3000, random_state=42, **params)


# MODEL REGISTRY
MODEL_LIBRARY = {
    "xgb": get_xgb_model,
    "lgbm": get_lgbm_model,
    "catboost": get_catboost_model,
    "rf": get_rf_model,
    "extra_trees": get_extra_trees_model,
    "gb": get_gb_model,
    "hist_gb": get_hist_gb_model,
    "linear": get_linear_model,
    "ridge": get_ridge_model,
    "lasso": get_lasso_model,
    "elastic": get_elastic_model,
}