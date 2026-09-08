import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def render_overview(results, X_test=None, y_test=None):
    st.subheader("📌 Общая информация о модели")

    model = results.get("model")
    preds = results.get("predictions")
    mode = results.get("mode", "unknown")
    score = results.get("score")
    best_params = results.get("best_params")

    if preds is None:
        st.warning("Нет предсказаний")
        return

    preds = np.array(preds)

    col1, col2, col3 = st.columns(3)
    col1.metric("Режим", mode)
    col2.metric("Модель", type(model).__name__ if model else "N/A")
    col3.metric("Размер теста", len(preds))

    if best_params:
        with st.expander("⚙️ Лучшие параметры (Optuna)"):
            st.json(best_params)

    if y_test is not None:
        y_test = np.array(y_test)

        if len(y_test) == len(preds):
            st.subheader("📊 Метрики качества")

            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            y_range = y_test.max() - y_test.min()
            nrmse = rmse / y_range if y_range > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{r2:.4f}", delta=_interpret_r2(r2))
            col2.metric("RMSE", f"{rmse:.4f}")
            col3.metric("MAE", f"{mae:.4f}")
            col4.metric("NRMSE", f"{nrmse:.4f}")

            st.markdown(_get_interpretation(r2, nrmse), unsafe_allow_html=True)

        else:
            st.warning("Размеры y_test и preds не совпадают")

    tabs = st.tabs(["📈 Scatter", "📊 Distribution", "📉 Residuals"])

    with tabs[0]:
        _render_scatter(y_test, preds)

    with tabs[1]:
        _render_distribution(preds, y_test)

    with tabs[2]:
        _render_residuals(y_test, preds)


def _interpret_r2(r2):
    if r2 >= 0.9:
        return "Отлично"
    elif r2 >= 0.7:
        return "Хорошо"
    elif r2 >= 0.5:
        return "Средне"
    else:
        return "Слабо"


def _get_interpretation(r2, nrmse):
    if r2 >= 0.9:
        return """
        <div style="padding:10px; background:#d4edda; border-radius:5px; margin-top:10px;">
        ✅ <b>Отличный результат!</b> Модель объясняет {r2:.1%} дисперсии. 
        NRMSE {nrmse:.2%} показывает низкую относительную ошибку.
        </div>
        """.format(r2, nrmse)
    elif r2 >= 0.7:
        return """
        <div style="padding:10px; background:#fff3cd; border-radius:5px; margin-top:10px;">
        ⚠️ <b>Хороший результат.</b> R²={r2:.1%} — модель улавливает основные паттерны.
        Для улучшения попробуйте добавить признаки или оптимизировать гиперпараметры.
        </div>
        """.format(r2)
    elif r2 >= 0.5:
        return """
        <div style="padding:10px; background:#f8d7da; border-radius:5px; margin-top:10px;">
        ⚠️ <b>Средний результат.</b> R²={r2:.1%} — модель работает лучше случайной, 
        но много дисперсии не объяснено. Рекомендуется: feature engineering, 
        другие модели, проверка данных на выбросы.
        </div>
        """.format(r2)
    else:
        return """
        <div style="padding:10px; background:#f8d7da; border-radius:5px; margin-top:10px;">
        🔴 <b>Результат слабый.</b> R²={r2:.1%} — модель практически не предсказывает.
        Проверьте: данные, целевую переменную, feature selection.
        </div>
        """.format(r2)


def _render_scatter(y_test, preds):
    if y_test is not None and len(y_test) == len(preds):
        st.markdown("""
        **Как читать:**  
        - Точки = каждое наблюдение (истинное vs предсказанное)  
        - Диагональ = идеальная линия (предсказание = истина)  
        - Отклонение от линии = ошибка
        """)

        fig, ax = plt.subplots(figsize=(8, 6))

        errors = np.abs(y_test - preds)
        vmax = np.percentile(errors, 95)

        scatter = ax.scatter(
            y_test, preds,
            c=errors,
            cmap="RdYlGn_r",
            alpha=0.6,
            s=20,
            vmin=0,
            vmax=vmax
        )

        min_val = min(np.min(y_test), np.min(preds))
        max_val = max(np.max(y_test), np.max(preds))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2, label="Ideal")

        ax.set_xlabel("Истинные значения")
        ax.set_ylabel("Предсказанные значения")
        ax.set_title("Predicted vs Actual")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label="|Error|")

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Нет y_test для отображения scatter")


def _render_distribution(preds, y_test):
    st.markdown("""
    **Как читать:**  
    - Сравнение распределений предсказаний и истинных значений  
    - Совпадение форм = модель хорошо улавливает распределение целевой переменной
    """)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(preds, bins=30, alpha=0.6, label="Predictions", color="#3498db")

    if y_test is not None and len(y_test) == len(preds):
        ax.hist(y_test, bins=30, alpha=0.4, label="Actual", color="#e74c3c")

    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


def _render_residuals(y_test, preds):
    if y_test is None or len(y_test) != len(preds):
        st.info("Нет y_test для анализа остатков")
        return

    st.markdown("""
    **Как читать:**  
    - Ось X = предсказанное значение  
    - Ось Y = остаток (истинное - предсказанное)  
    - Горизонтальная линия = идеально (остаток = 0)  
    - Воронкообразная форма = гетероскедастичность (разная точность в разных диапазонах)
    """)

    residuals = y_test - preds
    std_res = np.std(residuals)
    mean_res = np.mean(residuals)

    col1, col2 = st.columns(2)
    col1.metric("Mean residual", f"{mean_res:.4f}")
    col2.metric("Std residual", f"{std_res:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    errors = np.abs(residuals)
    vmax = np.percentile(errors, 95)

    axes[0].scatter(preds, residuals, c=errors, cmap="RdYlGn_r", alpha=0.6, s=20, vmin=0, vmax=vmax)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[0].axhline(2 * std_res, color="orange", linestyle=":", alpha=0.5)
    axes[0].axhline(-2 * std_res, color="orange", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual (y - pred)")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=30, alpha=0.7, color="#9b59b6", edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1].axvline(std_res, color="orange", linestyle=":", label=f"±1σ")
    axes[1].axvline(-std_res, color="orange", linestyle=":")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of Residuals")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    outlier_ratio = np.mean(np.abs(residuals) > 2 * std_res)
    st.metric("Outliers (>2σ)", f"{outlier_ratio:.1%}")