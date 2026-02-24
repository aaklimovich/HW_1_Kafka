import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
import sys
import joblib
import numpy as np
import concurrent.futures
import plotly.express as px
from backend.producers_consumers import DataProducer, RiskClassifierConsumer
from backend.config_file import kafka_topic

parent_dir = Path(__file__).resolve().parent
sys.path.append(str(parent_dir))

if 'results' not in st.session_state:
    st.session_state.results = []
if 'model_artifacts' not in st.session_state:
    st.session_state.model_artifacts = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0
if 'processed_count' not in st.session_state:
    st.session_state.processed_count = 0

if st.session_state.model_artifacts is None:
    model_path = "/app/model/model_Logistic_Regression.joblib"
    model_artifacts = joblib.load(model_path)
    st.session_state.model_artifacts = model_artifacts
    st.session_state.scaler = model_artifacts['scaler']
    st.session_state.feature_names = model_artifacts['feature_names']

def load_test_data():
    return pd.read_csv('data/processed/X_test_data.csv')

def get_original_data():
    if not st.session_state.results:
        return pd.DataFrame()
    
    valid_results = [r for r in st.session_state.results if r and 'risk_level' in r]
    if not valid_results:
        return pd.DataFrame()
    
    scaled_data = [[r['raw_data'].get(f, 0) for f in st.session_state.feature_names] for r in valid_results]
    original_data = st.session_state.scaler.inverse_transform(scaled_data)
    df_original = pd.DataFrame(original_data, columns=st.session_state.feature_names)
    
    df_original['risk_level'] = [r['risk_level'] for r in valid_results]
    df_original['prediction'] = [r['prediction'] for r in valid_results]
    df_original['user_id'] = [r['user_id'] for r in valid_results]
    df_original['confidence'] = [r['confidence'] for r in valid_results]
    
    return df_original

def plot_risk_distribution(df):
    if df.empty or 'risk_level' not in df.columns:
        return None
    
    fig = px.bar(df['risk_level'].value_counts().reset_index(),
                 x='risk_level', y='count', color='risk_level',
                 labels={'risk_level': 'Уровень риска', 'count': 'Количество'})
    fig.update_layout(height=400, showlegend=True)
    return fig

def plot_gaming_hours_distribution(df):
    if df.empty or 'daily_gaming_hours' not in df.columns:
        return None
    
    fig = px.histogram(df, x='daily_gaming_hours', color='risk_level', nbins=30,
                      labels={'daily_gaming_hours': 'Ежедневные часы игры', 'count': 'Количество'})
    fig.update_layout(height=400)
    return fig

def plot_age_distribution(df):
    if df.empty or 'age' not in df.columns or 'risk_level' not in df.columns:
        return None
    
    df = df[df['risk_level'].isin(['Low', 'High'])]
    if df.empty:
        return None
    
    fig = px.histogram(df, x='age', color='risk_level', nbins=30,
                      labels={'age': 'Возраст', 'count': 'Количество'},
                      barmode='group',
                      category_orders={'risk_level': ['Low', 'High']})
    fig.update_layout(height=400)
    return fig

def update_statistics(df, stats_container):
    with stats_container.container():
        col1, col2 = st.columns(2)
        
        if df.empty:
            col1.metric("Высокий риск", "0 (0.0%)")
            col2.metric("Средние часы игры", "0.0 ч")
            return
        
        total = len(df)
        high_risk = len(df[df['prediction'] == 1])
        high_risk_pct = (high_risk / total * 100) if total > 0 else 0
        avg_hours = df['daily_gaming_hours'].mean()
        
        col1.metric("Высокий риск", f"{high_risk} ({high_risk_pct:.1f}%)")
        col2.metric("Средние часы игры", f"{avg_hours:.1f} ч")

def process_message_batch(consumer, messages):
    results = []
    error_count = 0
    
    for msg in messages:
        if msg and not msg.error():
            try:
                message = msg.value().decode('utf-8')
                result = consumer.process_message(message)
                if result:
                    results.append(result)
                else:
                    error_count += 1
            except:
                error_count += 1
                st.session_state.error_count += 1
    
    return results, error_count

def main():
    st.title("Gaming Risk Analyzer")
    st.write("Система анализа риска игровой зависимости")
    
    test_data = load_test_data()
    st.write(f"Загружено {len(test_data)} тестовых записей")
    
    if st.button("Запустить анализ"):
        st.session_state.results = []
        st.session_state.error_count = 0
        st.session_state.processed_count = 0
        
        producer = DataProducer()
        consumer = RiskClassifierConsumer()
        consumer.consumer.subscribe([kafka_topic])
        
        total_records = len(test_data)
        batch_size = 5000
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        error_text = st.empty()
        
        st.subheader("Результаты анализа")
        col1, col2 = st.columns(2)
        with col1:
            risk_chart_container = st.empty()
        with col2:
            hours_chart_container = st.empty()
        
        col3, col4 = st.columns(2)
        with col3:
            age_chart_container = st.empty()
        
        stats_container = st.empty()
        
        for i in range(0, total_records, batch_size):
            current_batch_size = min(batch_size, total_records - i)
            data_batch = [{'user_id': f"user_{i+j}", **test_data.iloc[i+j].to_dict()} 
                         for j in range(current_batch_size)]
            
            for data in data_batch:
                producer.send_data(kafka_topic, data)
            producer.flush()
            
            messages = [consumer.consumer.poll(timeout=5.0) for _ in range(current_batch_size)]
            
            batch_results, error_count = process_message_batch(consumer, messages)
            
            st.session_state.results.extend(batch_results)
            st.session_state.processed_count += len(batch_results)
            st.session_state.error_count += error_count
            
            if (i // batch_size) % 2 == 0 or i + batch_size >= total_records:
                df_original = get_original_data()
                
                with risk_chart_container:
                    if fig := plot_risk_distribution(df_original):
                        st.plotly_chart(fig, use_container_width=True)
                
                with hours_chart_container:
                    if fig := plot_gaming_hours_distribution(df_original):
                        st.plotly_chart(fig, use_container_width=True)
                
                with age_chart_container:
                    if fig := plot_age_distribution(df_original):
                        st.plotly_chart(fig, use_container_width=True)
                
                update_statistics(df_original, stats_container)
                error_text.text(f"Ошибок: {st.session_state.error_count}")
            
            progress = st.session_state.processed_count / total_records
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Обработано {st.session_state.processed_count}/{total_records} записей")
        
        if df_original := get_original_data():
            csv = df_original.to_csv(index=False).encode('utf-8')
            st.download_button("Скачать результаты", csv, "risk_results.csv", "text/csv")

if __name__ == "__main__":
    main()