"""
Production-ready feature engineering
Гарантии:
- только scalar значения
- только numeric dtype
- защита от list/ndarray
- совместимость с SHAP / sklearn Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.kernel_approximation import RBFSampler
import warnings

warnings.filterwarnings("ignore")

# ГЛОБАЛЬНАЯ КОНСТАНТА
EPS = 1e-9


# =========================================================
# CORE SAFETY LAYER
# =========================================================


def _fix_scalar(x):
    if isinstance(x, (list, tuple, set)):
        return np.nan

    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.item())
        return np.nan

    return x


def force_scalar(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()

    for col in df_result.columns:
        if df_result[col].dtype == "object":
            df_result[col] = df_result[col].map(_fix_scalar)

    return df_result


def assert_no_object(df: pd.DataFrame, stage="unknown"):
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        raise ValueError(f"[{stage}] Остались object колонки: {list(obj_cols)}")


def finalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df_result = df.copy()

    for col in df_result.columns:
        df_result[col] = pd.to_numeric(df_result[col], errors="coerce")

    df_result = df_result.replace([np.inf, -np.inf], np.nan)
    df_result = df_result.fillna(0)

    return df_result


# =========================================================
# NUMERIC FEATURES (FIXED)
# =========================================================


def create_numeric_transforms(df, numeric_cols):
    df_result = df.copy()
    new_features = {}

    for col in numeric_cols:
        if col not in df_result.columns:
            continue

        series = pd.to_numeric(df_result[col], errors="coerce")

        std_val = series.std()
        if std_val == 0 or pd.isna(std_val):
            std_val = 1.0

        # КЛЮЧЕВОЙ ФИКС — ограничение значений
        safe_series = np.clip(series / std_val, -20, 20)

        try:
            # LOG
            if (series > 0).all():
                new_features[f"{col}_log1p"] = np.log1p(series)
                new_features[f"{col}_log10"] = np.log10(series + EPS)
            else:
                min_val = series.min()
                if min_val < 0:
                    shifted = series - min_val + 1
                    new_features[f"{col}_shifted_log1p"] = np.log1p(shifted)

            # POWERS
            new_features[f"{col}_square"] = series**2
            new_features[f"{col}_cube"] = series**3
            new_features[f"{col}_sqrt"] = np.sqrt(np.abs(series))
            new_features[f"{col}_cubert"] = np.cbrt(np.abs(series))

            # FIX: безопасные экспоненты
            new_features[f"{col}_exp"] = np.exp(safe_series)
            new_features[f"{col}_exp_neg"] = np.exp(-safe_series)

            # NONLINEAR
            new_features[f"{col}_sin"] = np.sin(series)
            new_features[f"{col}_cos"] = np.cos(series)
            new_features[f"{col}_sigmoid"] = 1 / (1 + np.exp(-safe_series))
            new_features[f"{col}_tanh"] = np.tanh(safe_series)

            # STATS
            mean_val = series.mean()
            new_features[f"{col}_standardized"] = (series - mean_val) / std_val

            # FIX: rank строго [0,1]
            rank = series.rank(pct=True)
            new_features[f"{col}_rank"] = rank.clip(0, 1)

            new_features[f"{col}_abs"] = np.abs(series)

            # POWER TRANSFORM
            try:
                if (series > 0).all():
                    pt = PowerTransformer(method="box-cox")
                else:
                    pt = PowerTransformer(method="yeo-johnson")

                new_features[f"{col}_power"] = pt.fit_transform(
                    series.to_frame()
                ).ravel()
            except Exception:
                pass

            # RBF
            try:
                rbf = RBFSampler(gamma=1.0 / std_val, n_components=1, random_state=42)
                new_features[f"{col}_rbf"] = rbf.fit_transform(
                    series.to_frame()
                ).ravel()
            except Exception:
                pass

            # NORMALIZATION
            min_val, max_val = series.min(), series.max()
            if max_val > min_val:
                new_features[f"{col}_normalized"] = (series - min_val) / (
                    max_val - min_val
                )

            # THRESHOLD
            median_val = series.median()
            new_features[f"{col}_above_median"] = (series > median_val).astype(float)

        except Exception:
            continue

    if new_features:
        new_df = pd.DataFrame(new_features, index=df_result.index)
        df_result = pd.concat([df_result, new_df], axis=1)

    return df_result


# =========================================================
# STAT FEATURES (FIXED)
# =========================================================


def generate_stat_features(df, cols):
    df_result = df.copy()

    X = df_result[cols].select_dtypes(include=["number"]).astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)

    row_mean = X.mean(axis=1)
    row_std = X.std(axis=1)
    row_min = X.min(axis=1)
    row_max = X.max(axis=1)

    row_range = row_max - row_min

    row_skew = (row_mean - row_min) / (row_std + EPS)

    df_result["stat_mean"] = row_mean.astype(float)
    df_result["stat_std"] = row_std.fillna(0).astype(float)
    df_result["stat_min"] = row_min.astype(float)
    df_result["stat_max"] = row_max.astype(float)
    df_result["stat_range"] = row_range.astype(float)
    df_result["stat_skew_proxy"] = (
        row_skew.replace([np.inf, -np.inf], 0).fillna(0).astype(float)
    )

    return df_result


# =========================================================
# RANK FEATURES (NEW)
# =========================================================


def generate_rank_features(df, cols):
    df_result = df.copy()

    for col in cols:
        if col not in df_result.columns:
            continue

        if not np.issubdtype(df_result[col].dtype, np.number):
            continue

        series = pd.to_numeric(df_result[col], errors="coerce")

        rank = series.rank(pct=True)

        df_result[f"{col}_rank_pct"] = rank.clip(0, 1).astype(float)

    return df_result


# =========================================================
# MAIN PIPELINE
# =========================================================


def apply_advanced_feature_engineering(X, verbose=True):
    if verbose:
        print("\n🔧 Feature engineering (SAFE MODE)")

    X_result = X.copy()

    X_result = force_scalar(X_result)

    numeric_cols = X_result.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_result.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric_cols:
        X_result[numeric_cols] = X_result[numeric_cols].fillna(X_result[numeric_cols].median())

    for col in cat_cols:
        X_result[col] = X_result[col].fillna("missing")

    # pipeline
    X_result = create_numeric_transforms(X_result, numeric_cols)
    X_result = generate_stat_features(X_result, numeric_cols)
    X_result = generate_rank_features(X_result, numeric_cols)

    X_result = finalize_numeric(X_result)

    assert_no_object(X_result, "FINAL")

    if verbose:
        print(f"   ✅ Итог: {X_result.shape[1]} признаков")

    return X_result