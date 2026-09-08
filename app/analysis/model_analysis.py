import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# MODEL ANALYSIS (IMPROVED)
# =========================================================
def render_model_analysis(results, X=None):

    st.subheader("🧠 Анализ модели")

    model = results.get("model")
    mode = results.get("mode", "unknown")

    if model is None:
        st.warning("Модель отсутствует")
        return

    st.write(f"Тип модели: **{type(model).__name__}**")
    st.write(f"Режим: **{mode}**")

    feature_names = None
    if X is not None:
        feature_names = list(X.columns)

    # =====================================================
    # STACKING
    # =====================================================
    if mode == "stack":

        st.subheader("📊 Stacking структура")

        if hasattr(model, "base_models"):

            base_models = model.base_models
            names = [name for name, _ in base_models]

            st.write("Базовые модели:")
            st.write(names)

            fig, ax = plt.subplots()

            weights = np.ones(len(names)) / len(names)
            ax.bar(names, weights)

            ax.set_title("Base model contribution (approx)")
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.warning("Base models недоступны")

        # META MODEL
        if hasattr(model, "meta_model"):

            st.subheader("⚙️ Meta model")

            meta_model = model.meta_model
            st.write(f"Тип: **{type(meta_model).__name__}**")

            if hasattr(meta_model, "coef_"):

                coefs = np.array(meta_model.coef_)

                fig, ax = plt.subplots()
                ax.bar(range(len(coefs)), coefs)
                ax.set_title("Meta model coefficients")

                st.pyplot(fig)
                plt.close(fig)

            else:
                st.info("Meta-model не линейная")

    # =====================================================
    # OPTUNA
    # =====================================================
    if mode == "optuna":

        st.subheader("⚙️ Optuna результаты")

        best_score = results.get("best_score")
        best_params = results.get("best_params")

        if best_score is not None:
            st.metric("Best score", f"{best_score:.4f}")

        if best_params:
            st.json(best_params)

    # =====================================================
    # BASE
    # =====================================================
    if mode == "base":
        st.subheader("📌 Base model")
        st.info("Обычная модель")

    # =====================================================
    # FEATURE IMPORTANCE / COEF
    # =====================================================
    st.subheader("📊 Feature Importance")

    try:
        importances = None

        # tree models
        if hasattr(model, "feature_importances_"):
            importances = np.array(model.feature_importances_)

        # linear models
        elif hasattr(model, "coef_"):
            importances = np.abs(np.array(model.coef_))

        if importances is None:
            st.info("Модель не поддерживает importance")
            return

        top_k = min(15, len(importances))
        indices = importances.argsort()[::-1][:top_k]

        if feature_names:
            labels = [feature_names[i] for i in indices]
        else:
            labels = indices

        fig, ax = plt.subplots()
        ax.barh(range(top_k), importances[indices])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

        ax.set_title("Top features")

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.warning(f"Ошибка importance: {e}")