from src.preprocessing import build_preprocessor
from src.model import build_model
from src.evaluation import evaluate
from src.stacking import train_stacking_verbose as train_stacking

from sklearn.model_selection import train_test_split
import numpy as np


def run_pipeline(file_chunks, y_column, config):
    """
    file_chunks: список DataFrame
    """

    # =========================
    # ОБЪЕДИНЯЕМ ЧАНКИ
    # =========================
    df = file_chunks[0] if len(file_chunks) == 1 else \
        np.concatenate([chunk.values for chunk in file_chunks])

    if not isinstance(df, np.ndarray):
        import pandas as pd
        df = pd.concat(file_chunks, axis=0)

    # =========================
    # SPLIT
    # =========================
    y = df[y_column]
    X = df.drop(columns=[y_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # PREPROCESSOR
    # =========================
    preprocessor = build_preprocessor(X_train)

    mode = config.get("mode", "base")

    # =========================
    # STACKING
    # =========================
    if mode == "stack":

        meta_tr, meta_test, trained_base_models = train_stacking(
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
            base_models_to_train=config.get("base_models", ["xgb", "lgbm", "catboost"]),
            return_meta_only=True
        )

        from sklearn.linear_model import Ridge

        meta_model = Ridge()
        meta_model.fit(meta_tr, y_train.values)

        preds = meta_model.predict(meta_test)

        from sklearn.metrics import r2_score
        score = r2_score(y_test, preds)

        return {
            "mode": "stack",
            "predictions": preds,
            "y_true": y_test.values,
            "score": score,
            "meta_model": meta_model
        }

    # =========================
    # OPTUNA
    # =========================
    elif mode == "optuna":

        import optuna

        def objective(trial):

            model_name = config.get("model_name", "xgb")

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3)
            }

            model = build_model(preprocessor, model_name, params)

            result = evaluate(model, X_train, X_test, y_train, y_test)

            return result["score"]

        study = optuna.create_study(direction="maximize")

        study.optimize(
            objective,
            n_trials=config.get("n_trials", 20),
            timeout=config.get("timeout"),
            callbacks=[config.get("optuna_callback")] if config.get("optuna_callback") else None
        )

        best_params = study.best_params

        model = build_model(preprocessor, config.get("model_name", "xgb"), best_params)

        result = evaluate(model, X_train, X_test, y_train, y_test)

        return {
            "mode": "optuna",
            "predictions": result["predictions"],
            "y_true": y_test.values,
            "score": result["score"],
            "best_params": best_params,
            "study": study
        }

    # =========================
    # BASE MODEL
    # =========================
    else:

        model_name = config.get("model_name", "xgb")

        model = build_model(preprocessor, model_name)

        result = evaluate(model, X_train, X_test, y_train, y_test)

        return {
            "mode": "base",
            "predictions": result["predictions"],
            "y_true": y_test.values,
            "score": result["score"],
            "model": model
        }