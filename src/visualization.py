import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(y_true, y_pred, model_name="model"):
    """
    Строит график предсказаний vs фактических значений
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("Фактические значения")
    plt.ylabel("Предсказанные значения")
    plt.title(f"{model_name} Predictions vs True")
    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Строит график важности признаков.
    Работает только для моделей с attribute feature_importances_.
    """
    # Получаем модель из pipeline если нужно
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        actual_model = model.named_steps["model"]
    else:
        actual_model = model

    if hasattr(actual_model, "feature_importances_"):
        importances = actual_model.feature_importances_
        df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(df["feature"], df["importance"])
        plt.gca().invert_yaxis()
        plt.title("Feature Importances")
        plt.show()
    else:
        print("Эта модель не имеет attribute feature_importances_")