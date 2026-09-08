import pandas as pd
import numpy as np
import warnings

from src.feature_engineering import apply_advanced_feature_engineering
from utils.cleaning import safe_clean
from utils.feature_names import clean_feature_names

warnings.filterwarnings("ignore")


# =========================
# MAIN PROCESSOR
# =========================
def process_features(X, use_feature=False, use_shap=False, y_train=None):
    """
    Стабильная обработка признаков (без утечек и рассинхрона)
    """

    X = X.copy()

    # =========================
    # FEATURE ENGINEERING
    # =========================
    if use_feature:
        try:
            X = apply_advanced_feature_engineering(X, verbose=False)
        except Exception as e:
            print(f"⚠️ Feature engineering error: {e}")

    # =========================
    # CLEANING (SAFE)
    # =========================
    try:
        X = safe_clean(X)
        X = clean_feature_names(X)
    except Exception as e:
        print(f"⚠️ Cleaning error: {e}")

        # 🔥 безопасный fallback без потери индексов
        X = X.copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

    # =========================
    # FINAL SANITY CHECK
    # =========================
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # убираем дубли колонок
    X = X.loc[:, ~X.columns.duplicated()]

    return X


# =========================
# SHAP SELECTION (FIXED)
# =========================
def select_features_with_shap(X_train, y_train, max_features=30):

    try:
        from xgboost import XGBRegressor
        from src.shap_analysis import explain_model

        if len(X_train) == 0 or len(y_train) == 0:
            return X_train.columns.tolist()[:max_features]

        # =========================
        # ТОЛЬКО ЧИСЛОВЫЕ
        # =========================
        X_num = X_train.select_dtypes(include=[np.number]).copy()

        if X_num.shape[1] == 0:
            return X_train.columns.tolist()[:max_features]

        X_num = clean_feature_names(X_num)

        # =========================
        # ALIGN X и y (КРИТИЧНО)
        # =========================
        if isinstance(y_train, pd.Series):
            valid_idx = y_train.dropna().index
            X_num = X_num.loc[valid_idx]
            y_clean = y_train.loc[valid_idx]
        else:
            mask = ~np.isnan(y_train)
            X_num = X_num[mask]
            y_clean = y_train[mask]

        if len(X_num) < 10:
            return X_train.columns.tolist()[:max_features]

        actual_max = min(max_features, X_num.shape[1])

        # =========================
        # MODEL
        # =========================
        model = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            verbosity=0
        )

        model.fit(X_num, y_clean)

        shap_importance = explain_model(
            model,
            X_num,
            max_features=actual_max
        )

        top_features = shap_importance["feature"].tolist()

        print(f"✅ SHAP selected {len(top_features)} features")

        return top_features

    except Exception as e:
        print(f"⚠️ SHAP error: {e}")
        return X_train.columns.tolist()[:max_features]