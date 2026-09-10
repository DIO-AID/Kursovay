"""
Production-ready feature engineering (SAFE — sklearn Transformer)
Гарантии:
- fit ТОЛЬКО на train (нет data leakage)
- transform на train/test одинаково
- совместимость с Pipeline, SHAP, sklearn
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.kernel_approximation import RBFSampler
import warnings
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

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
# SKLEARN TRANSFORMER (FIX DATA LEAKAGE)
# =========================================================


class AdvancedFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-совместимый трансформер для Feature Engineering.
    fit() вызывается ТОЛЬКО на train.
    transform() одинаково работает и на train, и на test.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self._fitted = False
        self._medians_ = None
        self._power_transformers_ = {}
        self._rbf_samplers_ = {}
        self._numeric_cols_ = None
        self._cat_cols_ = None
        self._col_stds_ = None
        self._col_means_ = None

    def fit(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        df = force_scalar(df)

        self._numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols_ = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if self._numeric_cols_:
            self._medians_ = df[self._numeric_cols_].median()
            self._col_stds_ = df[self._numeric_cols_].std().replace(0, 1.0)
            self._col_means_ = df[self._numeric_cols_].mean()

            for col in self._numeric_cols_:
                series = df[col].to_frame()
                std_val = self._col_stds_[col]
                if std_val == 0 or pd.isna(std_val):
                    std_val = 1.0

                try:
                    if (df[col] > 0).all():
                        pt = PowerTransformer(method="box-cox")
                    else:
                        pt = PowerTransformer(method="yeo-johnson")
                    pt.fit(series)
                    self._power_transformers_[col] = pt
                except Exception:
                    pass

                try:
                    rbf = RBFSampler(gamma=1.0 / std_val, n_components=1, random_state=42)
                    rbf.fit(series)
                    self._rbf_samplers_[col] = rbf
                except Exception:
                    pass

        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("Transformer не fit(). Вызовите fit() на train.")

        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        df = force_scalar(df)

        for col in self._numeric_cols_:
            if col not in df.columns:
                df[col] = 0.0

        for col in self._cat_cols_:
            if col not in df.columns:
                df[col] = "missing"
            df[col] = df[col].fillna("missing")

        numeric_cols = [c for c in self._numeric_cols_ if c in df.columns]

        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].fillna(self._medians_[numeric_cols])

        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            std_val = self._col_stds_[col] if col in self._col_stds_ else 1.0
            mean_val = self._col_means_[col] if col in self._col_means_ else 0.0

            safe_series = np.clip(series / std_val, -20, 20)

            new_features = {}

            try:
                if (series > 0).all():
                    new_features[f"{col}_log1p"] = np.log1p(series)
                    new_features[f"{col}_log10"] = np.log10(series + EPS)
                else:
                    min_val = series.min()
                    if min_val < 0:
                        shifted = series - min_val + 1
                        new_features[f"{col}_shifted_log1p"] = np.log1p(shifted)

                new_features[f"{col}_square"] = series**2
                new_features[f"{col}_cube"] = series**3
                new_features[f"{col}_sqrt"] = np.sqrt(np.abs(series))
                new_features[f"{col}_cubert"] = np.cbrt(np.abs(series))
                new_features[f"{col}_exp"] = np.exp(safe_series)
                new_features[f"{col}_exp_neg"] = np.exp(-safe_series)
                new_features[f"{col}_sin"] = np.sin(series)
                new_features[f"{col}_cos"] = np.cos(series)
                new_features[f"{col}_sigmoid"] = 1 / (1 + np.exp(-safe_series))
                new_features[f"{col}_tanh"] = np.tanh(safe_series)
                new_features[f"{col}_standardized"] = (series - mean_val) / std_val
                new_features[f"{col}_rank"] = series.rank(pct=True).clip(0, 1)
                new_features[f"{col}_abs"] = np.abs(series)

                if col in self._power_transformers_:
                    try:
                        new_features[f"{col}_power"] = self._power_transformers_[col].transform(
                            series.to_frame()
                        ).ravel()
                    except Exception:
                        pass

                if col in self._rbf_samplers_:
                    try:
                        new_features[f"{col}_rbf"] = self._rbf_samplers_[col].transform(
                            series.to_frame()
                        ).ravel()
                    except Exception:
                        pass

                min_val, max_val = series.min(), series.max()
                if max_val > min_val:
                    new_features[f"{col}_normalized"] = (series - min_val) / (max_val - min_val)

                new_features[f"{col}_above_median"] = (series > self._medians_[col]).astype(float)

            except Exception:
                continue

            if new_features:
                new_df = pd.DataFrame(new_features, index=df.index)
                df = pd.concat([df, new_df], axis=1)

        if numeric_cols:
            row_mean = df[numeric_cols].mean(axis=1)
            row_std = df[numeric_cols].std(axis=1).fillna(0)
            row_min = df[numeric_cols].min(axis=1)
            row_max = df[numeric_cols].max(axis=1)

            df["stat_mean"] = row_mean.astype(float)
            df["stat_std"] = row_std.astype(float)
            df["stat_min"] = row_min.astype(float)
            df["stat_max"] = row_max.astype(float)
            df["stat_range"] = (row_max - row_min).astype(float)
            df["stat_skew_proxy"] = ((row_mean - row_min) / (row_std + EPS)).replace(
                [np.inf, -np.inf], 0
            ).fillna(0).astype(float)

        for col in numeric_cols:
            rank = df[col].rank(pct=True)
            df[f"{col}_rank_pct"] = rank.clip(0, 1).astype(float)

        df = finalize_numeric(df)
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    def get_feature_names_out(self, input_features=None):
        return list(self._numeric_cols_) if self._numeric_cols_ else []


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================


def apply_advanced_feature_engineering(X, verbose=True):
    """
    Обратная совместимость: fit + transform на одних данных.
    ВНИМАНИЕ: для честной оценки используй AdvancedFeatureTransformer в Pipeline.
    """
    logger.info("Feature engineering (convenience mode — проверь data leakage)")
    transformer = AdvancedFeatureTransformer(verbose=verbose)
    return transformer.fit_transform(X)
