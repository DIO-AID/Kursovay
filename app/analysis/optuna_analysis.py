import streamlit as st
import optuna
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# OPTUNA ANALYSIS (EXPANDED)
# =========================================================
def render_optuna_analysis(db_path=None):
    st.subheader("⚙️ Optuna анализ")

    uploaded_db = st.file_uploader("Загрузить Optuna DB (.db)", type=["db"], key="optuna_db_upload")

    if uploaded_db is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            tmp.write(uploaded_db.read())
            db_path = tmp.name

    if db_path is None:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [
            os.path.join(base, "tuning", "optuna.db"),
            os.path.join(base, "Final", "results_fine", "optuna_fine.db"),
            os.path.join(base, "Final", "results_fine", "optuna_fair_xgboost.db"),
            os.path.join(base, "Final", "results_fine", "optuna_fair_lightgbm.db"),
            os.path.join(base, "Final", "results_fine", "optuna_fair_catboost.db"),
            os.path.join(base, "optuna.db"),
            os.path.join(os.getcwd(), "tuning", "optuna.db"),
        ]
        for p in candidates:
            if os.path.exists(p):
                db_path = p
                break

    if not os.path.exists(db_path):
        st.warning(f"Файл БД не найден: {db_path}")
        return

    storage = f"sqlite:///{db_path}"

    try:
        study_summaries = optuna.study.get_all_study_summaries(storage=storage)
    except Exception as e:
        st.error(f"Ошибка загрузки DB: {e}")
        return

    if not study_summaries:
        st.warning("Нет Optuna studies")
        return

    study_names = [s.study_name for s in study_summaries]
    selected = st.selectbox("Study", study_names)

    try:
        study = optuna.load_study(study_name=selected, storage=storage)
    except Exception as e:
        st.error(f"Ошибка загрузки study: {e}")
        return

    st.write(f"📌 Study: **{selected}**")

    col1, col2 = st.columns(2)
    if study.best_trial is not None:
        col1.metric("Best score", f"{study.best_value:.4f}")
    else:
        col1.metric("Best score", "N/A")
    col2.metric("Trials", len(study.trials))

    completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]

    if len(completed_trials) < 3:
        st.warning("Минимум 3 завершённых trial для анализа")
        return

    tabs = st.tabs(["📊 Convergence", "📈 Params", "🔬 TPE Deep", "🔥 Interactions"])

    with tabs[0]:
        _render_convergence(study, completed_trials)

    with tabs[1]:
        _render_param_importance(study, completed_trials)

    with tabs[2]:
        _render_tpe_analysis(study, completed_trials)

    with tabs[3]:
        _render_interactions(study, completed_trials)


# =========================================================
# 1. CONVERGENCE
# =========================================================
def _render_convergence(study, trials):
    st.markdown("""
    **Как читать:**  
    - Синяя линия = значение каждого trial  
    - Пунктир = лучший результат на данный момент  
    - Плавное снижение = алгоритм сходится
    """)

    df = study.trials_dataframe()
    if "value" not in df.columns:
        st.info("Нет данных")
        return

    df = df.sort_values("number")
    best_so_far = df["value"].cummax() if study.direction.name == "MAXIMIZE" else df["value"].cummin()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["number"], df["value"], "o-", alpha=0.5, label="Trial value", markersize=3)
    ax.plot(df["number"], best_so_far, "--", color="red", label="Best so far", linewidth=2)

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Score")
    ax.set_title("Optuna Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# 2. PARAM IMPORTANCE
# =========================================================
def _render_param_importance(study, trials):
    st.markdown("""
    **Как читать:**  
    - Высота столбца = насколько этот параметр влияет на результат  
    - >0.3 = критичный параметр, >0.1 = важный, <0.05 = можно фиксировать
    """)

    try:
        import optuna.importance as importance
        param_importance = importance.get_param_importances(study)

        df_imp = pd.DataFrame(
            list(param_importance.items()),
            columns=["param", "importance"]
        ).sort_values("importance", ascending=True)

        if len(df_imp) == 0:
            st.info("Недостаточно данных для importance")
            return

        colors = ["#e74c3c" if v > 0.3 else "#f39c12" if v > 0.1 else "#27ae60" for v in df_imp["importance"]]

        fig, ax = plt.subplots(figsize=(8, max(3, len(df_imp) * 0.5)))
        ax.barh(df_imp["param"], df_imp["importance"], color=colors)
        ax.set_xlabel("Importance")
        ax.set_title("Hyperparameter Importance")
        ax.axvline(0.1, color="orange", linestyle="--", alpha=0.5, label="threshold 0.1")
        ax.axvline(0.3, color="red", linestyle="--", alpha=0.5, label="threshold 0.3")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(df_imp.sort_values("importance", ascending=False))

    except Exception as e:
        st.warning(f"Param importance недоступен: {e}")


# =========================================================
# 3. TPE DEEP ANALYSIS (l(x) vs g(x))
# =========================================================
def _render_tpe_analysis(study, trials):
    st.markdown("""
    **Что это:**  
    - TPE разделяет trials на "хорошие" (l(x)) и "плохие" (g(x)) по медиане score  
    - Графики показывают как распределены параметры в хороших vs плохих trial
    """)

    df = study.trials_dataframe()
    if "value" not in df.columns:
        st.info("Нет данных")
        return

    median = df["value"].median()
    is_good = df["value"] >= median if study.direction.name == "MAXIMIZE" else df["value"] <= median

    params = [p for p in df.columns if p.startswith("params_")]
    if not params:
        st.info("Нет параметров для анализа")
        return

    param_to_plot = st.selectbox("Параметр", params)

    good_vals = df.loc[is_good, param_to_plot].dropna()
    bad_vals = df.loc[~is_good, param_to_plot].dropna()

    if len(good_vals) < 2 or len(bad_vals) < 2:
        st.info("Недостаточно данных для распределения")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(good_vals, bins=20, alpha=0.7, color="green", label=f"Good (n={len(good_vals)})")
    axes[0].hist(bad_vals, bins=20, alpha=0.5, color="red", label=f"Bad (n={len(bad_vals)})")
    axes[0].set_xlabel(param_to_plot.replace("params_", ""))
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution: Good vs Bad")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    bins = np.linspace(min(good_vals.min(), bad_vals.min()), max(good_vals.max(), bad_vals.max()), 20)
    l_den = np.histogram(good_vals, bins=bins, density=True)[0]
    g_den = np.histogram(bad_vals, bins=bins, density=True)[0]

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ratio = np.where(g_den > 0, l_den / (g_den + 1e-10), 0)

    axes[1].bar(bin_centers, ratio, width=(bins[1] - bins[0]) * 0.8, alpha=0.7, color="blue")
    axes[1].axhline(1, color="red", linestyle="--", label="l(x)=g(x)")
    axes[1].set_xlabel(param_to_plot.replace("params_", ""))
    axes[1].set_ylabel("l(x) / g(x)")
    axes[1].set_title("Density Ratio (l(x)/g(x))")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Интерпретация:**  
    - Если l(x)/g(x) > 1 в области X → TPE будет чаще выбирать это значение  
    - Если l(x)/g(x) < 1 → область бесперспективная  
    - Пики в хорошем распределении = оптимальные области
    """)


# =========================================================
# 4. PARAMETER INTERACTIONS
# =========================================================
def _render_interactions(study, trials):
    st.markdown("""
    **Что это:**  
    - Heatmap показывает средний score для комбинаций двух параметров  
    - Тёмные области = лучшие комбинации  
    - Помогает найти зависимости между параметрами
    """)

    df = study.trials_dataframe()
    if "value" not in df.columns:
        st.info("Нет данных")
        return

    params = [p for p in df.columns if p.startswith("params_")]
    if len(params) < 2:
        st.info("Нужно минимум 2 параметра")
        return

    col1, col2 = st.columns(2)
    param1 = col1.selectbox("Параметр 1", params)
    param2 = col2.selectbox("Параметр 2", [p for p in params if p != param1])

    try:
        df_clean = df[[param1, param2, "value"]].dropna()

        if len(df_clean) < 10:
            st.info("Недостаточно данных для heatmap")
            return

        # Convert categorical params to numeric for pivot
        for p in [param1, param2]:
            if df_clean[p].dtype == object:
                df_clean[p] = pd.Categorical(df_clean[p]).codes

        pivot = df_clean.pivot_table(
            values="value",
            index=param1,
            columns=param2,
            aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", origin="lower")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], rotation=45, fontsize=8)
        ax.set_yticklabels([f"{i:.2f}" for i in pivot.index], fontsize=8)

        ax.set_xlabel(param2.replace("params_", ""))
        ax.set_ylabel(param1.replace("params_", ""))
        ax.set_title("Score Heatmap (param1 vs param2)")

        plt.colorbar(im, ax=ax, label="Score")

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **Как читать:**  
        - Зелёный = высокий score, Красный = низкий  
        - Найди "горячую" область и используй эти значения как начальные
        """)

    except Exception as e:
        st.warning(f"Heatmap error: {e}")