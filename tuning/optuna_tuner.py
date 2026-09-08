import optuna
from sklearn.pipeline import Pipeline
from tuning.search_spaces import xgb_space


def run_optuna_xgb(
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test,
    metric_func,
    direction="minimize",
    n_trials=50,
):
    """
    Запуск Optuna для XGBoost с безопасными фичами
    """

    def objective(trial):
        params = xgb_space(trial)
        from xgboost import XGBRegressor
        model = XGBRegressor(**params, random_state=42)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        try:
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            score = metric_func(y_test, preds)
        except Exception as e:
            print("Ошибка в trial:", e)
            return 1e10 if direction == "minimize" else -1e10
        return score

    study = optuna.create_study(
        direction=direction,
        storage="sqlite:///optuna.db",
        study_name="xgb_optimization",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)
    return study