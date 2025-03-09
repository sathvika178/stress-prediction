import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import time

# Load the trained model and scaler
model = joblib.load('stress_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit Page Configuration
st.set_page_config(page_title='Stress Level Prediction', page_icon='💊', layout='wide')

# Custom Page Design
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('💊 Stress Level Prediction App')
st.markdown('**Predict stress levels based on health parameters.**')

# Tabs for Single Prediction & Bulk Prediction
tabs = st.tabs(["🔮 Single Prediction", "📂 Bulk Prediction"])

with tabs[0]:
    st.subheader("Enter Health Parameters")
    col1, col2 = st.columns(2)
    with col1:
        heart_rate = st.number_input('💓 Heart Rate (bpm)', min_value=40, max_value=200)
        breathing_rate = st.number_input('💨 Breathing Rate (breaths/min)', min_value=10, max_value=50)
        temperature = st.number_input('🌡️ Body Temperature (°C)', min_value=30.0, max_value=45.0)
        movement = st.number_input('🚶‍♂️ Movement (m/s²)', min_value=0.0, max_value=5.0)
    with col2:
        sound_level = st.number_input('🔊 Sound Level (dB)', min_value=20.0, max_value=120.0)
        oxygen_saturation = st.number_input('🫁 Oxygen Saturation (%)', min_value=50.0, max_value=100.0)
        sleep_duration = st.number_input('💤 Sleep Duration (hours)', min_value=0.0, max_value=12.0)
        blood_pressure = st.number_input('🩸 Blood Pressure (mmHg)', min_value=80, max_value=200)
    
    if st.button('🚀 Predict Stress Level'):
        with st.spinner('Analyzing your data...'):
            time.sleep(2)
            input_data = scaler.transform([[heart_rate, breathing_rate, temperature, movement,
                                            sound_level, oxygen_saturation, sleep_duration, blood_pressure]])
            prediction = model.predict(input_data)
            stress_level = '😰 High Stress' if prediction[0] == 1 else '🙂 Low Stress'
            st.success(f'Predicted Stress Level: **{stress_level}**')

with tabs[1]:
    st.subheader("📂 Bulk Prediction via CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data_scaled = scaler.transform(data)
        predictions = model.predict(data_scaled)
        data['Stress Level'] = ['High' if p == 1 else 'Low' for p in predictions]
        st.dataframe(data)

        # Stress Level Distribution
        st.subheader('📊 Stress Level Distribution')
        fig = px.histogram(data, x='Stress Level', color='Stress Level', title="Stress Level Count",
                           color_discrete_map={'High': 'red', 'Low': 'green'})
        st.plotly_chart(fig)

        # Feature Correlation Heatmap
        st.subheader('🧱 Feature Correlation Heatmap')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Stress Level Over Time
        if 'Timestamp' in data.columns:
            st.subheader('📈 Stress Level Over Time')
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            data.sort_values(by='Timestamp', inplace=True)
            fig = px.line(data, x='Timestamp', y='Stress Level', title="Stress Level Trend Over Time")
            st.plotly_chart(fig)
        else:
            st.info('No timestamp column found. Upload data with timestamps for time-series analysis.')
