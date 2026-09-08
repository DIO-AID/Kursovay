import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_stacking_verbose(
    X_train, X_test, y_train, y_test, preprocessor,
    base_models_to_train=None, return_meta_only=False
):
    """
    Стекинг с live прогрессом и выбором базовых моделей.
    Поддержка обучения на чанках.
    
    Параметры:
    - base_models_to_train: список базовых моделей, которые нужно обучать ['xgb','lgbm','catboost']
    - return_meta_only: если True, возвращает только meta_features для инкрементального обучения
    """
    
    if base_models_to_train is None:
        base_models_to_train = ["xgb", "lgbm", "catboost"]
    
    base_models_all = {
        "xgb": ("xgb", "xgb_space"),
        "lgbm": ("lgbm", "lgbm_space"),
        "catboost": ("catboost", "catboost_space"),
    }
    
    base_models = {k: v for k, v in base_models_all.items() if k in base_models_to_train}

    X_full_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    meta_features_train = np.zeros((len(X_train), len(base_models)))
    meta_features_test = np.zeros((len(X_test), len(base_models)))
    trained_base_models = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results_list = []

    for idx, (model_name, _) in enumerate(base_models.values()):
        print(f"\n🚀 Обучение базовой модели: {model_name}")

        # Динамический импорт модели
        if model_name == "xgb":
            from xgboost import XGBRegressor as ModelClass
        elif model_name == "lgbm":
            from lightgbm import LGBMRegressor as ModelClass
        elif model_name == "catboost":
            from catboost import CatBoostRegressor as ModelClass

        oof = np.zeros(len(X_train))
        test_preds = np.zeros((len(X_test), kf.n_splits))
        fold_models = []

        fold_progress = tqdm(total=kf.n_splits, desc=f"K-Fold {model_name}", leave=True)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_full_proc)):
            X_fold_train = X_full_proc[train_idx]
            X_fold_val = X_full_proc[val_idx]

            y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else y_train[train_idx]

            model = ModelClass()
            model.fit(X_fold_train, y_fold_train)

            oof[val_idx] = model.predict(X_fold_val)
            test_preds[:, fold] = model.predict(X_test_proc)
            fold_models.append(model)

            fold_progress.update(1)
        fold_progress.close()

        meta_features_train[:, idx] = oof
        meta_features_test[:, idx] = test_preds.mean(axis=1)
        trained_base_models.append((model_name, fold_models))

        # Метрики для базовой модели
        r2 = r2_score(y_train, oof)
        mae = mean_absolute_error(y_train, oof)
        rmse = mean_squared_error(y_train, oof, squared=False)

        results_list.append({"model": model_name, "type": "base", "R2": r2, "MAE": mae, "RMSE": rmse})
        print(f"✅ {model_name} OOF metrics: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    # Если нужно только meta_features для инкрементального обучения
    if return_meta_only:
        return meta_features_train, meta_features_test, trained_base_models

    # Мета-модель
    meta_model = Ridge()
    meta_model.fit(meta_features_train, y_train)
    final_pred = meta_model.predict(meta_features_test)

    if y_test is not None:
        r2 = r2_score(y_test, final_pred)
        mae = mean_absolute_error(y_test, final_pred)
        rmse = mean_squared_error(y_test, final_pred, squared=False)
        results_list.append({"model": "meta_ridge", "type": "meta", "R2": r2, "MAE": mae, "RMSE": rmse})
        print(f"\n🏆 STACKING META metrics: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    return trained_base_models, meta_model, results_list, final_pred