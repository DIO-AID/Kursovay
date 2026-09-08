import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# ERRORS ANALYSIS (DASHBOARD-COMPATIBLE)
# =========================================================
def render_errors_analysis(results, X=None):

    st.subheader("📉 Анализ ошибок модели")

    if results is None:
        st.warning("Нет results")
        return

    preds = results.get("predictions")
    y_test = results.get("y_test")

    if preds is None or y_test is None:
        st.warning("Нет данных для анализа ошибок")
        return

    preds = np.array(preds)
    y_test = np.array(y_test)

    if len(preds) != len(y_test):
        st.warning("Размеры preds и y_test не совпадают")
        return

    residuals = y_test - preds
    abs_errors = np.abs(residuals)

    # =====================================================
    # BASIC STATS
    # =====================================================
    st.subheader("📊 Статистика ошибок")

    col1, col2, col3 = st.columns(3)

    col1.metric("Mean Error", f"{np.mean(residuals):.4f}")
    col2.metric("Std Error", f"{np.std(residuals):.4f}")
    col3.metric("Max Error", f"{np.max(abs_errors):.4f}")

    # =====================================================
    # RESIDUAL DISTRIBUTION
    # =====================================================
    st.subheader("📊 Распределение ошибок")

    fig, ax = plt.subplots()
    ax.hist(residuals, bins=40)
    ax.axvline(0)
    ax.set_title("Residual Distribution")

    st.pyplot(fig)
    plt.close(fig)

    # =====================================================
    # PRED VS ERROR
    # =====================================================
    st.subheader("📉 Ошибки vs Предсказания")

    fig, ax = plt.subplots()
    ax.scatter(preds, residuals, alpha=0.5)
    ax.axhline(0)

    ax.set_xlabel("Predictions")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predictions")

    st.pyplot(fig)
    plt.close(fig)

    # =====================================================
    # WORST CASES
    # =====================================================
    st.subheader("🚨 Худшие предсказания")

    df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": preds,
        "error": abs_errors
    })

    df_sorted = df.sort_values("error", ascending=False)

    st.dataframe(df_sorted.head(20))

    # =====================================================
    # ERROR VS TRUE
    # =====================================================
    st.subheader("📊 Ошибка vs Истинное значение")

    fig, ax = plt.subplots()
    ax.scatter(y_test, abs_errors, alpha=0.5)

    ax.set_xlabel("True values")
    ax.set_ylabel("Absolute error")
    ax.set_title("Error vs True Value")

    st.pyplot(fig)
    plt.close(fig)