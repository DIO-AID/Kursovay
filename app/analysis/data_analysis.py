import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# DATA ANALYSIS DASHBOARD (SAFE VERSION)
# =========================================================
def render_data_analysis(X_train=None, y_train=None):

    st.subheader("📊 Анализ данных (EDA)")

    if X_train is None or len(X_train) == 0:
        st.warning("Нет данных для анализа")
        return

    # =====================================================
    # SAMPLE FOR PERFORMANCE
    # =====================================================
    MAX_ROWS = 5000

    if len(X_train) > MAX_ROWS:
        X = X_train.sample(MAX_ROWS, random_state=42)
        st.info(f"Используется сэмпл: {MAX_ROWS} строк")
    else:
        X = X_train

    # =====================================================
    # BASIC INFO
    # =====================================================
    st.markdown("### 📌 Общая структура")

    col1, col2, col3 = st.columns(3)

    col1.metric("Строки", X.shape[0])
    col2.metric("Признаки", X.shape[1])
    col3.metric(
        "Память (MB)",
        round(X.memory_usage(deep=True).sum() / 1024**2, 2)
    )

    st.write("Типы данных:")
    st.write(X.dtypes.value_counts())

    # =====================================================
    # MISSING VALUES
    # =====================================================
    st.markdown("### ❗ Пропуски")

    missing = X.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(30)

    if len(missing) > 0:
        fig, ax = plt.subplots()
        missing.plot(kind="bar", ax=ax)
        ax.set_title("Top missing values")
        st.pyplot(fig)
    else:
        st.success("Пропусков нет")

    # =====================================================
    # DUPLICATES
    # =====================================================
    st.markdown("### 🧬 Дубликаты")

    dup = X.duplicated().sum()
    st.metric("Дубликаты", dup)

    # =====================================================
    # NUMERIC ANALYSIS
    # =====================================================
    num_cols = X.select_dtypes(include=["number"]).columns

    if len(num_cols) > 0:

        st.markdown("### 📈 Числовые признаки")

        desc = X[num_cols].describe().T
        st.dataframe(desc)

        # correlation (SAFE LIMIT)
        if len(num_cols) <= 25:

            st.markdown("### 🔗 Корреляции")

            corr = X[num_cols].corr()

            fig, ax = plt.subplots()
            im = ax.imshow(corr.values, cmap="coolwarm", aspect="auto")
            plt.colorbar(im, ax=ax)

            ax.set_xticks(range(len(num_cols)))
            ax.set_yticks(range(len(num_cols)))
            ax.set_xticklabels(num_cols, rotation=90, fontsize=8)
            ax.set_yticklabels(num_cols, fontsize=8)

            st.pyplot(fig)

        else:
            st.info("Слишком много признаков для корреляционной матрицы")

    # =====================================================
    # TARGET ANALYSIS
    # =====================================================
    if y_train is not None:

        st.markdown("### 🎯 Target анализ")

        y = pd.Series(y_train)

        st.write(y.describe())

        fig, ax = plt.subplots()
        ax.hist(y, bins=40)
        ax.set_title("Distribution of target")
        st.pyplot(fig)

    # =====================================================
    # CONSTANT FEATURES
    # =====================================================
    st.markdown("### 🧱 Константные признаки")

    nunique = X.nunique()
    constant = nunique[nunique <= 1].index.tolist()

    st.write(f"Количество: {len(constant)}")

    if len(constant) > 0:
        st.write(constant[:30])