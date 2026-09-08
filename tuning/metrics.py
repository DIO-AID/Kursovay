from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_metric(name: str):
    """
    Возвращает функцию метрики по имени.
    """
    metrics = {
        "mae": mean_absolute_error,
        "rmse": lambda y_true, y_pred: mean_squared_error(
            y_true, y_pred, squared=False
        ),
        "r2": r2_score,
    }
    if name not in metrics:
        raise ValueError(f"Метрика {name} не поддерживается")
    return metrics[name]
