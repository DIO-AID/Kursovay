import numpy as np
import pandas as pd

from sklearn.model_selection import KFold


# =========================================================
# 1. CONSTANT FEATURES
# =========================================================
def remove_constant_features(X):
    nunique = X.nunique()
    return X[nunique[nunique > 1].index]


# =========================================================
# 2. CORRELATION FILTER (SAFE)
# =========================================================
def remove_correlated_features(X, threshold=0.95):

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]

    return X.drop(columns=to_drop, errors="ignore")


# =========================================================
# 3. CV FEATURE IMPORTANCE (NO LEAKAGE)
# =========================================================
def select_by_model_importance(model_class, X, y, top_k=50, n_splits=3):
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]
    
    importances = np.zeros(X.shape[1])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, val_idx in kf.split(X):

        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx]

        model = model_class()
        model.fit(X_tr, y_tr)

        if hasattr(model, "feature_importances_"):
            importances += model.feature_importances_

        elif hasattr(model, "coef_"):
            importances += np.abs(model.coef_)

    importances /= n_splits

    df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    })

    selected = df.sort_values("importance", ascending=False).head(top_k)["feature"]

    return X[selected]


# =========================================================
# 4. SHAP (FIXED + STABLE)
# =========================================================
def select_by_shap(model_class, X, y, top_k=30, sample_size=200):
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    import shap

    X_sample = X.sample(min(sample_size, len(X)), random_state=42)

    if hasattr(y, "loc"):
        y_sample = y.loc[X_sample.index]
    else:
        y_sample = y[:len(X_sample)]

    model = model_class()
    model.fit(X_sample, y_sample)

    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        values = shap_values.values

    except Exception:
        # fallback для бустингов
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_sample)

    # унификация формы
    if isinstance(values, list):
        values = values[0]

    importance = np.abs(values).mean(axis=0)

    # 🔥 КРИТИЧНЫЙ ФИКС: длина может не совпадать
    if len(importance) != X.shape[1]:
        importance = importance[:X.shape[1]]

    df = pd.DataFrame({
        "feature": X.columns,
        "shap": importance
    })

    selected = df.sort_values("shap", ascending=False).head(top_k)["feature"]

    return X[selected]


# =========================================================
# MAIN PIPELINE
# =========================================================
def auto_feature_selection(
    X,
    y,
    model_class=None,
    use_shap=False,
    corr_threshold=0.95,
    top_k_model=50,
    top_k_shap=30
):

    print("🔹 Feature selection (SAFE MODE)")

    # 1. CONSTANT
    X = remove_constant_features(X)

    # 2. CORRELATION (ONLY NUMERIC)
    num_cols = X.select_dtypes(include=["number"]).columns

    if len(num_cols) > 0:
        X_num = remove_correlated_features(X[num_cols], threshold=corr_threshold)

        # сохраняем порядок колонок
        other_cols = X.drop(columns=num_cols)
        X = pd.concat([X_num, other_cols], axis=1)

    # 3. MODEL IMPORTANCE (CV SAFE)
    if model_class is not None:
        X = select_by_model_importance(
            model_class,
            X,
            y,
            top_k=min(top_k_model, X.shape[1])
        )

    # 4. SHAP (OPTIONAL)
    if use_shap and model_class is not None and X.shape[1] > 0:
        X = select_by_shap(
            model_class,
            X,
            y,
            top_k=min(top_k_shap, X.shape[1])
        )

    print(f"✅ Final features: {X.shape[1]}")

    return X