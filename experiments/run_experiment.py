from sklearn.model_selection import train_test_split

from src.preprocessing import build_preprocessor
from src.evaluation import evaluate
from src.model import build_model
from src.stacking import train_stacking

from tuning.run_optuna_model import run_optuna_model
from tuning.metrics import get_metric

from experiments.feature_pipeline import process_features, select_features_with_shap


def run_experiment(df, config):

    X = df.drop(columns=[config["target"]])
    y = df[config["target"]]

    X = process_features(X, config.get("feature", False), config.get("shap", False), y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if config.get("shap", False):
        top_features = select_features_with_shap(X_train, y_train)
        X_train = X_train[top_features]
        X_test = X_test[top_features]

    preprocessor = build_preprocessor(X_train)

    if config.get("mode") == "stack":
        return train_stacking(X_train, X_test, y_train, y_test, preprocessor)

    if config.get("mode") == "optuna":

        model_class = config.get("model_class")
        search_space = config.get("search_space")

        if model_class is None or search_space is None:
            raise ValueError("model_class and search_space required for optuna mode")

        study = run_optuna_model(
            model_class=model_class,
            search_space_func=search_space,
            preprocessor=preprocessor,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            metric_func=get_metric(config.get("metric", "r2")),
            study_name=config.get("study_name", "experiment"),
            direction=config.get("direction", "maximize"),
            n_trials=config.get("n_trials", 50),
            timeout=600,
        )

        return study.best_value

    model = build_model(preprocessor, config.get("model_name", "xgb"))
    results = evaluate(model, X_train, X_test, y_train, y_test)

    return results.get("R2", 0)