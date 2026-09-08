import shap
import numpy as np
import pandas as pd


def explain_model(model, X, max_features=20, sample_size=2000):
    """
    Строит SHAP значения и возвращает важность признаков.

    Parameters
    ----------
    model : обученная модель
    X : DataFrame
    max_features : int
    sample_size : int
        размер подвыборки для SHAP (ускоряет расчёт)
    """

    # извлекаем модель из pipeline если нужно
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model = model.named_steps['model']

    # используем подвыборку для ускорения
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X

    # проверяем что все признаки числовые
    non_numeric = X_sample.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        raise ValueError(f"SHAP requires numeric features. Non-numeric columns: {list(non_numeric)}")

    print(f"SHAP sample size: {len(X_sample)}")

    # создаем explainer
    explainer = shap.Explainer(model, X_sample)

    shap_values = explainer(X_sample)

    # SHAP summary plot
    shap.summary_plot(shap_values, X_sample, max_display=max_features)

    # SHAP bar importance
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=max_features)

    # importance dataframe
    importance_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "shap_mean_abs": np.abs(shap_values.values).mean(axis=0),
        }
    )

    importance_df = importance_df.sort_values("shap_mean_abs", ascending=False)

    return importance_df