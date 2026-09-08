import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class Preprocessor:
    def __init__(self):
        self.preprocessor = None
        self.columns = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.feature_names = None

    # =========================
    # FIT
    # =========================
    def fit(self, X: pd.DataFrame):

        X = X.copy()

        self.columns = list(X.columns)

        self.numeric_cols = X.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()

        self.categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # защита от пустых списков
        transformers = []

        if self.numeric_cols:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("num", numeric_pipeline, self.numeric_cols))

        if self.categorical_cols:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ])
            transformers.append(("cat", categorical_pipeline, self.categorical_cols))

        self.preprocessor = ColumnTransformer(transformers)

        self.preprocessor.fit(X)

        self._build_feature_names()

        return self

    # =========================
    # FIT TRANSFORM
    # =========================
    def fit_transform(self, X):
        return self.fit(X).transform(X)

    # =========================
    # TRANSFORM (SAFE)
    # =========================
    def transform(self, X: pd.DataFrame):

        X = X.copy()

        # 🔥 КРИТИЧНО: добавляем отсутствующие колонки
        for col in self.columns:
            if col not in X.columns:
                X[col] = np.nan

        # 🔥 убираем лишние колонки (иначе падения)
        X = X[self.columns]

        return self.preprocessor.transform(X)

    # =========================
    # FEATURE NAMES (FIXED)
    # =========================
    def _build_feature_names(self):

        feature_names = []

        if self.numeric_cols:
            feature_names.extend(self.numeric_cols)

        if self.categorical_cols:
            try:
                ohe = self.preprocessor.named_transformers_["cat"]["encoder"]
                cat_names = ohe.get_feature_names_out(self.categorical_cols)
                feature_names.extend(cat_names)
            except Exception:
                pass

        self.feature_names = feature_names

    def get_feature_names(self):
        return self.feature_names


def build_preprocessor(X: pd.DataFrame):
    return Preprocessor().fit(X)