import streamlit as st

from .overview import render_overview
from .model_analysis import render_model_analysis
from .shap_analysis import render_shap_analysis
from .optuna_analysis import render_optuna_analysis
from .data_analysis import render_data_analysis
from .errors_analysis import render_errors_analysis


# =========================================================
# DASHBOARD CONTROLLER (FIXED)
# =========================================================
def render_dashboard(results, file_chunks, config):

    st.title("📊 ML Model Analysis Dashboard")

    if results is None:
        st.warning("Нет данных для анализа")
        return

    model = results.get("model")
    preds = results.get("predictions")
    mode = results.get("mode")

    # =====================================================
    # SAFE DATA EXTRACTION
    # =====================================================
    if file_chunks is None or len(file_chunks) == 0:
        st.warning("Нет данных")
        return

    X = file_chunks[0]

    # y может быть внутри первого df (если есть target)
    y = None
    if config and "target" in config:
        target_col = config["target"]
        if target_col in X.columns:
            y = X[target_col]
            X = X.drop(columns=[target_col])

    # fallback (если y нет)
    if y is None:
        y = None

    st.markdown(f"### Режим: `{mode}`")

    # =====================================================
    # TABS
    # =====================================================
    tabs = st.tabs([
        "📌 Overview",
        "📦 Data",
        "🧠 Model",
        "🧬 SHAP",
        "⚙ Optuna",
        "❌ Errors"
    ])

    # =====================================================
    # OVERVIEW
    # =====================================================
    with tabs[0]:
        render_overview(results, X, y)

    # =====================================================
    # DATA ANALYSIS
    # =====================================================
    with tabs[1]:
        render_data_analysis(X, y)

    # =====================================================
    # MODEL ANALYSIS
    # =====================================================
    with tabs[2]:
        render_model_analysis(results)

    # =====================================================
    # SHAP ANALYSIS
    # =====================================================
    with tabs[3]:
        render_shap_analysis(
            model=model,
            X_test=X,
            results=results
        )

    # =====================================================
    # OPTUNA ANALYSIS
    # =====================================================
    with tabs[4]:
        render_optuna_analysis()

    # =====================================================
    # ERRORS ANALYSIS
    # =====================================================
    with tabs[5]:
        render_errors_analysis(results, X)