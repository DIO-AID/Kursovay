import streamlit as st
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def render_shap_analysis(model=None, X_test=None, results=None):
    st.subheader("🧬 SHAP анализ модели")

    if results:
        model = results.get("model", model)
        X_test = results.get("X_test_processed", X_test)
        y_test = results.get("y_test", None)

    if model is None or X_test is None or len(X_test) == 0:
        st.warning("Нет данных для SHAP анализа")
        return

    MAX_SAMPLES = 300

    try:
        if len(X_test) > MAX_SAMPLES:
            X_sample = X_test.sample(MAX_SAMPLES, random_state=42)
        else:
            X_sample = X_test
    except Exception:
        X_sample = X_test

    tabs = st.tabs(["📊 Summary", "📈 Importance", "🔬 Dependence", "📋 Table"])

    with tabs[0]:
        _render_shap_summary(model, X_sample)

    with tabs[1]:
        _render_shap_importance(model, X_sample)

    with tabs[2]:
        _render_shap_dependence(model, X_sample)

    with tabs[3]:
        _render_shap_table(model, X_sample)


def _render_shap_summary(model, X_sample):
    st.markdown("""
    **Как читать:**  
    - Ось X = SHAP value (влияние на предсказание)  
    - Цвет = значение признака (красный = высокое, синий = низкое)  
    - Точки справа от 0 = положительное влияние на prediction
    """)

    try:
        model_name = type(model).__name__.lower()

        if "xgb" in model_name or "lgbm" in model_name or "catboost" in model_name or "forest" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        fig = plt.figure(figsize=(10, max(5, min(X_sample.shape[1] * 0.3, 12))))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"SHAP error: {e}")


def _render_shap_importance(model, X_sample):
    st.markdown("""
    **Как читать:**  
    - Каждый столбец = средний |SHAP value| по всем наблюдениям  
    - Чем выше, тем больше признак влияет на предсказание
    """)

    try:
        model_name = type(model).__name__.lower()

        if "xgb" in model_name or "lgbm" in model_name or "catboost" in model_name or "forest" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": X_sample.columns[:len(mean_abs)],
            "importance": mean_abs
        }).sort_values("importance", ascending=True)

        if len(importance_df) > 20:
            importance_df = importance_df.tail(20)

        fig, ax = plt.subplots(figsize=(8, max(4, len(importance_df) * 0.4)))
        ax.barh(importance_df["feature"], importance_df["importance"], color="#3498db")
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title("SHAP Feature Importance")
        ax.grid(True, alpha=0.3, axis="x")

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Importance error: {e}")


def _render_shap_dependence(model, X_sample):
    st.markdown("""
    **Что это:**  
    - Зависимость SHAP value от значения признака  
    - Показывает как признак влияет на предсказание при разных значениях  
    - Цвет = значение второго по важности признака (взаимодействие)
    """)

    try:
        model_name = type(model).__name__.lower()

        if "xgb" in model_name or "lgbm" in model_name or "catboost" in model_name or "forest" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs = np.abs(shap_values).mean(axis=0)
        top_features = X_sample.columns[:len(mean_abs)]

        if len(top_features) < 1:
            st.info("Недостаточно признаков")
            return

        feature = st.selectbox("Признак", top_features[:10])

        idx = list(X_sample.columns).index(feature)

        shap_values_flat = shap_values[:, idx] if shap_values.ndim > 1 else shap_values

        feature_values = X_sample[feature].values

        interaction_feature = None
        if len(top_features) > 1:
            for f in top_features:
                if f != feature:
                    interaction_feature = f
                    break

        color_values = X_sample[interaction_feature].values if interaction_feature else None

        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(
            feature_values, shap_values_flat,
            c=color_values, cmap="coolwarm", alpha=0.6, s=20
        )

        ax.axhline(0, color="black", linestyle="--", alpha=0.3)
        ax.set_xlabel(feature)
        ax.set_ylabel(f"SHAP value for {feature}")
        ax.set_title(f"SHAP Dependence: {feature}")
        ax.grid(True, alpha=0.3)

        if color_values is not None:
            plt.colorbar(scatter, ax=ax, label=interaction_feature)

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.warning(f"Dependence error: {e}")


def _render_shap_table(model, X_sample):
    st.markdown("""
    **Как читать:**  
    - importance = среднее |SHAP| по всем наблюдениям  
    - positive = доля положительных SHAP values  
    - Чем выше importance, тем критичнее признак
    """)

    try:
        model_name = type(model).__name__.lower()

        if "xgb" in model_name or "lgbm" in model_name or "catboost" in model_name or "forest" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_val = shap_values.mean(axis=0)
        positive_ratio = (shap_values > 0).mean(axis=0)

        table_df = pd.DataFrame({
            "feature": X_sample.columns[:len(mean_abs)],
            "importance": mean_abs,
            "mean_shap": mean_val,
            "positive_share": positive_ratio
        }).sort_values("importance", ascending=False)

        st.dataframe(table_df.head(30))

        st.download_button(
            "📥 Download CSV",
            table_df.to_csv(index=False),
            "shap_importance.csv",
            "text/csv"
        )

    except Exception as e:
        st.warning(f"Table error: {e}")