import streamlit as st
import os
import pandas as pd
import json
from src.results_db import get_results_db


def render_results_management():
    """Управление результатами обучения."""
    st.subheader("📚 История обучения")
    
    results_db = get_results_db()
    
    # Get all runs
    try:
        runs_df = results_db.get_all_runs()
    except Exception as e:
        st.warning(f"Нет данных: {e}")
        return
    
    if len(runs_df) == 0:
        st.info("Нет сохранённых результатов")
        return
    
    # Display summary
    st.write(f"**Всего прогонов:** {len(runs_df)}")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        model_filter = st.selectbox("Модель", ["Все"] + list(runs_df["model_name"].unique()))
    with col2:
        target_filter = st.selectbox("Целевая", ["Все"] + list(runs_df["target"].unique()))
    
    # Apply filters
    if model_filter != "Все":
        runs_df = runs_df[runs_df["model_name"] == model_filter]
    if target_filter != "Все":
        runs_df = runs_df[runs_df["target"] == target_filter]
    
    # Show runs
    for _, row in runs_df.iterrows():
        with st.expander(f"**{row['model_name']}** | {row['run_name']} | Score: {row['score']:.4f if row['score'] else 'N/A'}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Режим:** {row['mode']}")
                st.markdown(f"**Target:** {row['target']}")
                if row['score']:
                    st.metric("R² Score", f"{row['score']:.4f}")
            
            with col2:
                st.markdown(f"**Дата:** {row['created_at'][:19]}")
                if row['best_params']:
                    with st.expander("📋 Лучшие параметры"):
                        try:
                            params = json.loads(row['best_params'])
                            st.json(params)
                        except:
                            st.write(row['best_params'])
            
            if row['selected_features']:
                n_features = len(json.loads(row['selected_features']))
                st.markdown(f"**Признаков:** {n_features}")
                
                if st.checkbox(f"Показать признаки для {row['model_name']}", key=f"feat_{row['id']}"):
                    try:
                        features = json.loads(row['selected_features'])
                        st.write(", ".join(features[:20]))
                        if n_features > 20:
                            st.write(f"... и ещё {n_features - 20}")
                    except:
                        st.write(row['selected_features'])