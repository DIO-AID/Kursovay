import time
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


# =========================
# SIMPLE & STABLE EVALUATION
# =========================
def evaluate(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor=None,   # оставляем для совместимости, но НЕ используем
    return_model=True,
    val_size=0.2
):
    """
    Универсальная оценка модели:
    - работает с Pipeline
    - работает со StackingModel
    - не дублирует preprocessing
    - не ломает архитектуру
    """

    start_time = time.time()

    # =========================
    # FIT
    # =========================
    model.fit(X_train, y_train)

    train_time = time.time() - start_time

    # =========================
    # PREDICT
    # =========================
    pred_start = time.time()

    predictions = None
    train_predictions = None

    # безопасный predict
    if hasattr(model, "predict"):
        predictions = model.predict(X_test)
        train_predictions = model.predict(X_train)

    pred_time = time.time() - pred_start

    # =========================
    # METRICS
    # =========================
    result = {}

    if y_test is not None and predictions is not None:
        result.update({
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(np.mean((y_test - predictions) ** 2)),
            "R2": r2_score(y_test, predictions),
        })

    if train_predictions is not None:
        result.update({
            "train_MAE": mean_absolute_error(y_train, train_predictions),
            "train_RMSE": np.sqrt(np.mean((y_train - train_predictions) ** 2)),
            "train_R2": r2_score(y_train, train_predictions),
        })

    result.update({
        "predictions": predictions,
        "training_time": train_time,
        "prediction_time": pred_time,
        "model": model if return_model else None,
    })

    return result


# =========================
# CV (оставляем почти как есть)
# =========================
def evaluate_with_cv(model, X, y, cv=5):
    from sklearn.model_selection import cross_val_score, KFold

    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)

    mae_scores = -cross_val_score(
        model, X, y, cv=cv_strategy, scoring="neg_mean_absolute_error", n_jobs=1
    )
    rmse_scores = np.sqrt(
        -cross_val_score(
            model, X, y, cv=cv_strategy, scoring="neg_mean_squared_error", n_jobs=1
        )
    )
    r2_scores = cross_val_score(
        model, X, y, cv=cv_strategy, scoring="r2", n_jobs=1
    )

    return {
        "MAE_mean": mae_scores.mean(),
        "MAE_std": mae_scores.std(),
        "RMSE_mean": rmse_scores.mean(),
        "RMSE_std": rmse_scores.std(),
        "R2_mean": r2_scores.mean(),
        "R2_std": r2_scores.std(),
    }


# =========================
# PRINT (оставляем)
# =========================
def print_metrics(results, title="Результаты модели"):
    print(f"\n{'='*50}")
    print(f"📊 {title}")
    print(f"{'='*50}")

    if "MAE" in results:
        print(f"\n🎯 Test:")
        print(f"MAE:  {results['MAE']:.4f}")
        print(f"RMSE: {results['RMSE']:.4f}")
        print(f"R2:   {results['R2']:.4f}")

    if "train_MAE" in results:
        print(f"\n📈 Train:")
        print(f"MAE:  {results['train_MAE']:.4f}")
        print(f"RMSE: {results['train_RMSE']:.4f}")
        print(f"R2:   {results['train_R2']:.4f}")

    if "training_time" in results:
        print(f"\n⏱️ Time:")
        print(f"Train: {results['training_time']:.2f}s")
        print(f"Predict: {results['prediction_time']:.4f}s")