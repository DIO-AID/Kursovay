import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, LinearRegression, ElasticNet, BayesianRidge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from tuning.run_optuna_model import _safe_fit


# =========================
# STACKING MODEL
# =========================
class StackingModel:
    def __init__(self, base_models, meta_model, preprocessor, use_weighted=False, meta_type="ridge"):
        self.base_models = base_models
        self.meta_model = meta_model
        self.preprocessor = preprocessor
        self.use_weighted = use_weighted
        self.meta_type = meta_type

    def predict(self, X):
        X_proc = self.preprocessor.transform(X)

        meta_features = []

        for _, models in self.base_models:
            preds = np.column_stack([m.predict(X_proc) for m in models])
            meta_features.append(preds.mean(axis=1))

        meta_features = np.column_stack(meta_features)

        if self.use_weighted and hasattr(self, 'weights_'):
            return (meta_features * self.weights_).sum(axis=1)
        
        return self.meta_model.predict(meta_features)


# =========================
# WEIGHTED ENSEMBLE HELPER
# =========================
def compute_weights_by_r2(oof_preds, y_train):
    """Compute weights proportional to R² of each model."""
    r2_scores = []
    for pred in oof_preds.T:
        r2 = r2_score(y_train, pred)
        r2_scores.append(max(r2, 0.01))  # floor at 0.01 to avoid zero
    
    weights = np.array(r2_scores)
    weights = weights / weights.sum()
    return weights


# =========================
# MAIN STACKING (CLEAN)
# =========================
def train_stacking_verbose(X_train_proc, X_test_proc, y_train, y_test, preprocessor, best_params_map=None, use_weighted=False, meta_model_type="ridge"):

    base_models = ["xgb", "lgbm", "catboost"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    meta_features_train = np.zeros((len(X_train_proc), len(base_models)))
    meta_features_test = np.zeros((len(X_test_proc), len(base_models)))

    trained_base_models = []

    # =========================
    # BASE MODELS
    # =========================
    for idx, model_name in enumerate(base_models):

        print(f"\n🚀 Base model: {model_name}")

        if model_name == "xgb":
            from xgboost import XGBRegressor
            model_class = XGBRegressor

        elif model_name == "lgbm":
            from lightgbm import LGBMRegressor
            model_class = LGBMRegressor

        elif model_name == "catboost":
            from catboost import CatBoostRegressor
            model_class = CatBoostRegressor

        if best_params_map and model_name in best_params_map:
            best_params = best_params_map[model_name]
            print(f"   📌 Using pre-tuned params: {best_params}")
        else:
            best_params = {}
            print(f"   ⚠️ No pre-tuned params, using defaults")

        # =========================
        # OOF TRAINING
        # =========================
        oof = np.zeros(len(X_train_proc))
        fold_models = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_proc)):

            X_tr = X_train_proc[tr_idx]
            X_val = X_train_proc[val_idx]

            y_tr = y_train.iloc[tr_idx] if isinstance(y_train, pd.Series) else y_train[tr_idx]
            y_val = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]

            model = model_class(**best_params)

            fit_params = {}

            if model_name == "xgb":
                fit_params = {"eval_set": [(X_val, y_val)], "verbose": 0}

            elif model_name == "lgbm":
                fit_params = {"eval_set": [(X_val, y_val)]}

            elif model_name == "catboost":
                fit_params = {"verbose": False}

            model = _safe_fit(model, X_tr, y_tr, fit_params)

            oof[val_idx] = model.predict(X_val)

            fold_models.append(model)

        meta_features_train[:, idx] = oof

        # =========================
        # FINAL MODEL ON FULL TRAIN
        # =========================
        final_model = model_class(**best_params)
        final_model = _safe_fit(final_model, X_train_proc, y_train, {})

        trained_base_models.append((model_name, fold_models))

        oof_r2 = r2_score(y_train, oof)
        print(f"✅ {model_name} OOF R2: {oof_r2:.4f}")

    # =========================
    # WEIGHTED or RIDGE/ELASTIC META
    # =========================
    weights = None
    meta_type = "ridge"
    
    if use_weighted:
        print(f"\n⚖️ Using Weighted Ensemble (no meta-leak)")
        weights = compute_weights_by_r2(meta_features_train, y_train)
        print(f"   Weights: XGB={weights[0]:.3f}, LGBM={weights[1]:.3f}, CAT={weights[2]:.3f}")
        meta_type = "weighted"
    elif meta_model_type == "elasticnet":
        print(f"\n🟣 Using ElasticNet Meta (L1+L2)")
        meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        meta_model.fit(meta_features_train, y_train)
        meta_type = "elasticnet"
    elif meta_model_type == "bayesian":
        print(f"\n🔵 Using BayesianRidge Meta (auto regularization)")
        meta_model = BayesianRidge()
        meta_model.fit(meta_features_train, y_train)
        meta_type = "bayesian"
    else:
        print(f"\n🔮 Using Ridge Meta (may overfit)")
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_features_train, y_train)

    # =========================
    # TEST PREDICTION
    # =========================
    for idx, (_, models) in enumerate(trained_base_models):
        preds = np.column_stack([m.predict(X_test_proc) for m in models])
        meta_features_test[:, idx] = preds.mean(axis=1)

    if use_weighted:
        final_pred = (meta_features_test * weights).sum(axis=1)
        score = None
        if y_test is not None:
            score = r2_score(y_test, final_pred)
            print(f"\n⚖️ WEIGHTED STACK R2: {score:.4f}")
    else:
        final_pred = meta_model.predict(meta_features_test)
        score = None
        if y_test is not None:
            score = r2_score(y_test, final_pred)
            print(f"\n{meta_type.upper()} STACK R2: {score:.4f}")

    # Create model wrapper
    if use_weighted:
        stack_model = StackingModel(trained_base_models, None, preprocessor, use_weighted=True, meta_type="weighted")
        stack_model.weights_ = weights
    else:
        stack_model = StackingModel(trained_base_models, meta_model, preprocessor, use_weighted=False, meta_type=meta_type)

    return stack_model, final_pred, score


train_stacking = train_stacking_verbose