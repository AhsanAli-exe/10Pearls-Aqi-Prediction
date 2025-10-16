import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(
    page_title="10 Pearls AQI Predictor",
    page_icon="🌬️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_FILE = os.path.join(BASE_DIR,"static","style.css")

if os.path.exists(CSS_FILE):
    local_css(CSS_FILE)


FLASK_API_URL = "http://127.0.0.1:5000/predict"

def get_aqi_category(aqi):
    aqi = int(aqi)
    if aqi <= 50:
        return "Good", "#4CAF50", "White"
    elif aqi <= 100:
        return "Moderate", "#FFEB3B", "Black"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#FF9800", "White"
    elif aqi <= 200:
        return "Unhealthy", "#F44336", "White"
    elif aqi <= 300:
        return "Very Unhealthy", "#9C27B0", "White"
    else:
        return "Hazardous", "#795548", "White"

st.title("10 Pearls AQI Predictor 🌬️")
st.write("Predicting the Air Quality Index for the next 3 days in Karachi.")

if st.button("Get AQI Prediction"):
    try:
        with st.spinner("Fetching predictions from the model..."):
            response = requests.get(FLASK_API_URL)
            response.raise_for_status() 
            predictions = response.json()

        st.success("Prediction successful!")
        st.subheader("3-Day AQI Forecast")
        col1,col2,col3 = st.columns(3)
        days = ["Today","Tomorrow","Day After Tomorrow"]
        aqi_values = [predictions['day1_aqi'], predictions['day2_aqi'], predictions['day3_aqi']]

        for i, col in enumerate([col1, col2, col3]):
            with col:
                aqi = aqi_values[i]
                category, color, text_color = get_aqi_category(aqi)
                st.markdown(
                    f'''
                    <div class="metric-card" style="background-color: {color}; color: {text_color};">
                        <h4>{days[i]}</h4>
                        <h2>{int(aqi)}</h2>
                        <p>{category}</p>
                    </div>
                    ''', 
                    unsafe_allow_html=True
                )

        st.subheader("AQI Trend")
        chart_data = pd.DataFrame({
            'Day': days,
            'AQI': [int(val) for val in aqi_values]
        })
        st.line_chart(chart_data.set_index('Day'))

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the prediction API. Please ensure the Flask API is running. Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


