from tuning.search_spaces import xgb_space, lgbm_space, rf_space, catboost_space

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

BASE_CONFIG = {
    "target": "median_house_value",
    "metric": "r2",
    "direction": "maximize",
    "n_trials": 50,
}


MODEL_CONFIGS = {
    "xgb": (XGBRegressor, xgb_space),
    "lgbm": (LGBMRegressor, lgbm_space),
    "rf": (RandomForestRegressor, rf_space),
    "catboost": (CatBoostRegressor, catboost_space),
}