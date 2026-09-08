import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import build_preprocessor
from src.model import build_model, incremental_fit
from src.stacking import train_stacking_verbose as train_stacking
from tuning.run_optuna_model import run_optuna_model
from tuning.metrics import get_metric

from src.feature_selection import auto_feature_selection
from experiments.feature_pipeline import process_features
from src.results_db import get_results_db

import pandas as pd


# =========================================================
# ALIGN COLUMNS
# =========================================================
def align_columns(df_list):
    common_cols = df_list[0].columns
    for df in df_list[1:]:
        common_cols = common_cols.intersection(df.columns)

    return [df[common_cols].copy() for df in df_list]


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_pipeline(file_chunks, y_column, config):

    mode = config.get("mode", "base")
    model_name = config.get("model_name", "xgb")
    
    # Get data file name
    data_file = file_chunks[0] if hasattr(file_chunks[0], 'shape') else "unknown"
    data_file_name = getattr(file_chunks[0], 'name', 'unknown') if hasattr(file_chunks[0], 'name') else str(file_chunks[0])
    
    # Initialize results DB
    results_db = get_results_db()
    run_id, run_name = results_db.start_run(model_name, data_file_name, y_column, mode)

    file_chunks = align_columns(file_chunks)

    first_df = file_chunks[0]

    from sklearn.model_selection import train_test_split

    X_full = first_df.drop(columns=[y_column])
    y_full = first_df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================
    if config.get("feature_engineering", False):
        X_train = process_features(X_train, True, False, y_train)
        X_test = process_features(X_test, True, False, None)

        common_cols = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

    # =====================================================
    # FEATURE SELECTION
    # =====================================================
    selected_cols = None

    if config.get("feature_selection", False):

        from sklearn.ensemble import RandomForestRegressor

        selected_X = auto_feature_selection(
            X=X_train,
            y=y_train,
            model_class=RandomForestRegressor,
            use_shap=config.get("use_shap_selection", False),
            top_k_model=50,
            top_k_shap=30
        )

        selected_cols = list(selected_X.columns)

        X_train = X_train[selected_cols]
        X_test = X_test.reindex(columns=selected_cols, fill_value=0)

    # =====================================================
    # PREPROCESSOR
    # =====================================================
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # =====================================================
    # STACKING MODE
    # =====================================================
    if mode == "stack":

        from sklearn.model_selection import train_test_split as tts
        X_tune, X_stack_train, y_tune, y_stack_train = tts(
            X_train, y_train, test_size=0.2, random_state=42
        )

        X_tune_proc = preprocessor.transform(X_tune)
        X_stack_train_proc = preprocessor.transform(X_stack_train)
        X_test_proc = preprocessor.transform(X_test)

        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor
        from tuning.search_spaces import xgb_space, lgbm_space, catboost_space

        best_params_map = {}

        for model_name, (model_class, space) in [
            ("xgb", (XGBRegressor, xgb_space)),
            ("lgbm", (LGBMRegressor, lgbm_space)),
            ("catboost", (CatBoostRegressor, catboost_space)),
        ]:
            study, _ = run_optuna_model(
                model_class=model_class,
                search_space_func=space,
                preprocessor=None,
                X_train=X_tune_proc,
                X_test=None,
                y_train=y_tune,
                y_test=None,
                metric_func=get_metric("r2"),
                study_name=f"stack_pre_{model_name}",
                direction="maximize",
                n_trials=config.get("n_trials", 20),
                timeout=config.get("timeout"),
            )
            best_params_map[model_name] = study.best_params
            print(f"✅ {model_name} tuned: {study.best_params}")

        model, preds, score = train_stacking(
            X_train_proc=X_stack_train_proc,
            X_test_proc=X_test_proc,
            y_train=y_stack_train,
            y_test=y_test,
            preprocessor=preprocessor,
            best_params_map=best_params_map,
            use_weighted=config.get("use_weighted_stack", False),
            meta_model_type=config.get("meta_model_type", "ridge")
        )
        
        # Save to results DB
        if score is not None:
            results_db.save_score(run_id, score)
            results_db.save_best_params(run_id, best_params_map)
        
        # Save selected features
        final_features = selected_cols if selected_cols is not None else list(X_train.columns)
        results_db.save_selected_features(run_id, final_features)

        return {
            "mode": "stack",
            "model_name": "stacking",
            "run_id": run_id,
            "model": model,
            "score": score,
            "predictions": preds,
            "best_params_map": best_params_map,

            "X_test": X_test,
            "y_test": y_test,

            "X_test_original": X_test,
            "X_test_processed": X_test_proc
        }

    # =====================================================
    # TRANSFORM
    # =====================================================
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # =====================================================
    # OPTUNA MODE
    # =====================================================
    if mode == "optuna":

        model_name = config.get("model_name")

        from sklearn.model_selection import train_test_split as tts
        X_train_proc, X_val_proc, y_train_opt, y_val_opt = tts(
            X_train_proc, y_train, test_size=0.2, random_state=42
        )

        from src.model import get_model_and_space
        model_class, search_space = get_model_and_space(model_name)

        study, best_model = run_optuna_model(
            model_class=model_class,
            search_space_func=search_space,
            preprocessor=None,
            X_train=X_train_proc,
            X_test=None,
            y_train=y_train_opt,
            y_test=None,
            metric_func=get_metric("r2"),
            study_name="streamlit_optuna",
            direction="maximize",
            n_trials=config.get("n_trials", 20),
            timeout=config.get("timeout"),
            callback=config.get("optuna_callback"),
            X_val=X_val_proc,
            y_val=y_val_opt
        )

        preds = best_model.predict(X_test_proc)

        # Save to results DB
        if study.best_value is not None:
            results_db.save_score(run_id, study.best_value)
            results_db.save_best_params(run_id, study.best_params)
        
        # Save selected features
        final_features = selected_cols if selected_cols is not None else list(X_train.columns)
        results_db.save_selected_features(run_id, final_features)

        return {
            "mode": "optuna",
            "model_name": model_name,
            "run_id": run_id,
            "best_score": study.best_value,
            "score": study.best_value,
            "best_params": study.best_params,
            "model": best_model,
            "predictions": preds,

            "X_test": X_test,
            "y_test": y_test,

            "X_test_original": X_test,
            "X_test_processed": X_test_proc,
        
        "selected_features": final_features
    }

    # =====================================================
    # BASE MODE (no Optuna, no stacking)
    # =====================================================
    from src.model import get_model_and_space
    model_class, _ = get_model_and_space(model_name)
    model = model_class()

    model.fit(X_train_proc, y_train)
    preds = model.predict(X_test_proc)

    from sklearn.metrics import r2_score
    score = r2_score(y_test, preds)

    results_db.save_score(run_id, score)
    results_db.save_best_params(run_id, {})
    final_features = selected_cols if selected_cols is not None else list(X_train.columns)
    results_db.save_selected_features(run_id, final_features)

    return {
        "mode": "base",
        "model_name": model_name,
        "run_id": run_id,
        "score": score,
        "model": model,
        "predictions": preds,

        "X_test": X_test,
        "y_test": y_test,

        "X_test_original": X_test,
        "X_test_processed": X_test_proc,

        "selected_features": final_features
    }