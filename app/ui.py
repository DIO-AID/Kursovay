import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_handler import DataHandler
from pipeline_adapter import run_pipeline
from utils.column_profiler import enrich_with_llm

# 🔥 ПОДКЛЮЧАЕМ АНАЛИТИКУ
from analysis.overview import render_overview
from analysis.model_analysis import render_model_analysis
from analysis.errors_analysis import render_errors_analysis
from analysis.optuna_analysis import render_optuna_analysis
from analysis.shap_analysis import render_shap_analysis
from analysis.results_management import render_results_management


def render_ui():
    st.set_page_config(page_title="ML Pipeline", layout="wide")
    st.title("🚀 ML Pipeline Dashboard")

    # =========================
    # CLEAR CACHE
    # =========================
    if st.button("🗑️ Очистить кэш"):
        for key in list(st.session_state.keys()):
            if key.startswith("results_"):
                del st.session_state[key]
        st.rerun()

    # =========================
    # MODE
    # =========================
    st.subheader("📦 Режим данных")

    data_mode = st.radio(
        "Выберите режим",
        ["Один файл", "Несколько файлов (чанки)", "Kaggle"]
    )

    limit_rows = st.checkbox("Ограничить выборку для демо", value=False)
    n_rows = st.number_input("Макс. строк для демо", 10000, 5000000, 100000)

    file_chunks = []
    y_column = None
    kaggle_mode = False
    submission_df = None

    # =========================
    # SINGLE FILE
    # =========================
    if data_mode == "Один файл":
        file = st.file_uploader("📂 Dataset", type=["csv", "xlsx", "parquet"])

        if file is None:
            return

        df = DataHandler.load_file(file, n_rows if limit_rows else None)
        st.dataframe(df.head())

        _render_data_profile(df)

        y_column = st.selectbox("🎯 Target", df.columns)
        file_chunks = [df]

    # =========================
    # MULTI FILE
    # =========================
    elif data_mode == "Несколько файлов (чанки)":
        files = st.file_uploader(
            "📂 Загрузите несколько CSV",
            type=["csv"],
            accept_multiple_files=True
        )

        if not files:
            return

        st.write(f"Загружено файлов: {len(files)}")

        df_preview = DataHandler.load_file(files[0], 1000)
        st.dataframe(df_preview.head())
        _render_data_profile(df_preview)

        y_column = st.selectbox("🎯 Target", df_preview.columns)

        for file in files:
            df = DataHandler.load_file(file, n_rows if limit_rows else None)
            file_chunks.append(df)

    # =========================
    # KAGGLE
    # =========================
    else:
        train_file = st.file_uploader("Train CSV", key="train")
        test_file = st.file_uploader("Test CSV", key="test")
        sub_file = st.file_uploader("Sample submission", key="sub")

        if not train_file or not test_file:
            return

        train_df = DataHandler.load_file(train_file, n_rows if limit_rows else None)
        test_df = DataHandler.load_file(test_file, n_rows if limit_rows else None)

        st.dataframe(train_df.head())
        _render_data_profile(train_df)

        y_column = st.selectbox("🎯 Target", train_df.columns)
        file_chunks = [train_df]

        if sub_file:
            submission_df = pd.read_csv(sub_file)

        kaggle_mode = True

    # =========================
    # SETTINGS
    # =========================
    st.subheader("⚙️ Настройки")

    model_name = st.selectbox(
        "Модель",
        ["xgb", "lgbm", "rf", "catboost", "linear"]
    )

    use_stacking = st.checkbox("Stacking")
    use_weighted_stack = False
    meta_model_type = "ridge"
    if use_stacking:
        col1, col2 = st.columns(2)
        use_weighted_stack = col1.checkbox("Weighted Ensemble")
        meta_model_type = col2.selectbox("Meta модель", ["ridge", "elasticnet", "bayesian"], index=0)
    use_optuna = st.checkbox("Optuna")
    feature_engineering = st.checkbox("Feature Engineering")

    base_models = st.multiselect(
        "Модели для стекинга",
        ["xgb", "lgbm", "catboost"],
        default=["xgb", "lgbm", "catboost"]
    )

    config = {}

    config["feature_selection"] = st.checkbox("Auto Feature Selection")
    config["use_shap_selection"] = st.checkbox("SHAP selection")
    config["use_weighted_stack"] = use_weighted_stack
    config["meta_model_type"] = meta_model_type

    n_trials, timeout = None, None

    if use_optuna:
        n_trials = st.slider("Trials", 5, 100, 20)
        timeout = st.slider("Timeout", 60, 3600, 600)

    # =========================
    # CACHE KEY
    # =========================
    cache_key = f"results_{model_name}_{use_stacking}_{use_optuna}_{feature_engineering}"

    # =========================
    # RUN
    # =========================
    if st.button("🚀 Запустить"):
        with st.spinner("Обучение..."):

            progress_bar = st.progress(0)
            status_text = st.empty()
            best_score_text = st.empty()

            def optuna_callback(study, trial):
                progress = len(study.trials)
                total = n_trials or 20

                progress_bar.progress(min(progress / total, 1.0))
                status_text.write(f"🔄 Trials: {progress}/{total}")

                if study.best_value is not None:
                    best_score_text.write(f"🏆 Best score: {study.best_value:.5f}")

            if use_stacking:
                mode = "stack"
            elif use_optuna:
                mode = "optuna"
            else:
                mode = "base"

            config.update({
                "mode": mode,
                "model_name": model_name,
                "feature_engineering": feature_engineering,
                "n_trials": n_trials,
                "timeout": timeout,
                "base_models": base_models,
                "optuna_callback": optuna_callback
            })

            results = run_pipeline(file_chunks, y_column, config)
            st.session_state[cache_key] = results

        st.success("✅ Обучение завершено")

    # =========================
    # RESULTS + FULL ANALYTICS
    # =========================
    if cache_key in st.session_state:

        results = st.session_state[cache_key]

        st.subheader("📊 Results")

        score = results.get("score")

        if score is not None:
            st.metric("R² Score", round(score, 5))

        # =========================
        # TABS С АНАЛИТИКОЙ
        # =========================
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview",
            "🤖 Model",
            "❌ Errors",
            "⚙️ Optuna",
            "🧠 SHAP",
            "📚 History"
        ])

        with tab1:
            render_overview(results, results.get("X_test"), results.get("y_test"))

        with tab2:
            render_model_analysis(results, results.get("X_test_original"))

        with tab3:
            render_errors_analysis(results)

        with tab4:
            render_optuna_analysis()

        with tab5:
            render_shap_analysis(results)
        
        with tab6:
            render_results_management()

    st.info("""
    💡 Теперь доступна полная аналитика:
    - Overview
    - Model analysis
    - Errors
    - Optuna
    - SHAP
    """)


def _render_data_profile(df):
    from utils.column_profiler import profile_dataframe, enrich_with_llm

    with st.expander("📋 Профиль данных (авто-расшифровка колонок)", expanded=False):
        profile = profile_dataframe(df)

        use_llm = st.checkbox("🧠 Расшифровать через ИИ (qwen2.5)", value=False, key="llm_profile")
        if use_llm:
            with st.spinner("Анализ колонок через ИИ..."):
                profile = enrich_with_llm(profile, df)
            if profile.get("_llm_error"):
                st.warning(f"Ошибка LLM: {profile['_llm_error']}")
            elif profile.get("_llm_enriched"):
                st.success("Колонки расшифрованы")

        profile_data = []
        for col_name, info in profile["columns"].items():
            desc = info.get("llm_description") or info.get("description", "")
            role = info.get("llm_role") or info.get("suggested_role", "")
            transform = info.get("llm_transform", "")
            null_str = f"{info['null_pct']}%" if info['null_pct'] > 0 else "0%"

            if info["type_category"] == "numeric":
                details = f"mean={info.get('mean','?')}, skew={info.get('skew','?')}"
            elif info["type_category"] == "categorical":
                tops = ", ".join(list(info.get("top_values", {}).keys())[:3])
                details = f"top: {tops}"
            else:
                details = ""

            profile_data.append({
                "Колонка": col_name,
                "Тип": info["type_category"],
                "Уникальных": info["unique_count"],
                "Пропуски": null_str,
                "Роль": role,
                "Описание": desc,
                "Детали": details,
            })

        if profile_data:
            st.dataframe(
                pd.DataFrame(profile_data),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Колонка": st.column_config.TextColumn("Колонка", width="small"),
                    "Тип": st.column_config.TextColumn("Тип", width="small"),
                    "Уникальных": st.column_config.NumberColumn("Уникальных", width="small"),
                    "Пропуски": st.column_config.TextColumn("Пропуски", width="small"),
                    "Роль": st.column_config.TextColumn("Роль", width="small"),
                    "Описание": st.column_config.TextColumn("Описание", width="large"),
                    "Детали": st.column_config.TextColumn("Детали", width="medium"),
                },
            )

            suggest_target = _suggest_target(profile)
            if suggest_target:
                st.info(f"💡 Возможная целевая колонка: **{suggest_target}**")


def _suggest_target(profile):
    for col_name, info in profile["columns"].items():
        llm_role = info.get("llm_role", "")
        if llm_role == "target":
            return col_name

    names = ["target", "price", "value", "label", "y", "class", "cost", "amount"]
    for col_name in profile["columns"]:
        for n in names:
            if n in col_name.lower():
                return col_name
    return None