import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime,timedelta

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AQI Predictor",
    page_icon=":airplane:",
    layout="wide",
    initial_sidebar_state="expanded",
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_FILE = os.path.join(BASE_DIR,"static","style.css")

if os.path.exists(CSS_FILE):
    local_css(CSS_FILE)

# API Configuration
FLASK_API_URL = "http://127.0.0.1:5000"
CITIES_API_URL = f"{FLASK_API_URL}/cities"
PREDICT_API_URL = f"{FLASK_API_URL}/predict"
HISTORICAL_API_URL = f"{FLASK_API_URL}/historical"
CURRENT_WEATHER_URL = f"{FLASK_API_URL}/weather/current"

def get_aqi_category(aqi):
    aqi = int(aqi)
    if aqi <= 50:
        return "Good","#4CAF50","White"
    elif aqi <= 100:
        return "Moderate","#FFEB3B","Black"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups","#FF9800","White"
    elif aqi <= 200:
        return "Unhealthy","#F44336","White"
    elif aqi <= 300:
        return "Very Unhealthy","#9C27B0","White"
    else:
        return "Hazardous","#795548","White"

st.title("_AQI_ _Prediction_ :blue[Model] :sunglasses:")
st.header("Developed by: :blue[Ahsan Ali]")
st.write("Real-time Air Quality Index prediction optimized for Karachi, Pakistan")

with st.sidebar:
    st.title("🔧 System Info")
    st.title("ℹ️ About")
    st.info("""
    **Karachi-Optimized AQI Predictor**

    This application provides:
    - **Real-time AQI predictions** using live weather data
    - **Current conditions only** (no historical bias)
    - **Karachi-specific optimization** for maximum accuracy
    - **Interactive visualizations** for Karachi AQI trends
    """)

    st.markdown("---")
    st.markdown("[Ahsan Ali's LinkedIn](https://www.linkedin.com/in/ahsan--ali)")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Karachi, Pakistan")

    if st.button("Get Karachi AQI Prediction",type="primary",use_container_width=True):
        try:
            with st.spinner("🌐 Fetching live Karachi weather data and generating predictions..."):
                response = requests.get(f"{PREDICT_API_URL}")
                response.raise_for_status()
                predictions = response.json()

            st.success("Prediction successful!")

            st.subheader("3-Day AQI Forecast")

            pred_data = predictions['predictions']
            aqi_values = [pred_data['day1_aqi'],pred_data['day2_aqi'],pred_data['day3_aqi']]
            days = ["Today","Tomorrow","Day After Tomorrow"]

            cols = st.columns(3)
            for i,(col,day,aqi) in enumerate(zip(cols,days,aqi_values)):
                with col:
                    category,color,text_color = get_aqi_category(aqi)
                    st.markdown(
                        f"""
                        <div class="metric-card" style="background-color: {color}; color: {text_color};">
                            <h4>{day}</h4>
                            <h2>{int(aqi)}</h2>
                            <p>{category}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.subheader("📈 Prediction Trend")
            trend_df = pd.DataFrame({
                'Day': days,
                'AQI': [int(val) for val in aqi_values]
            })

            fig = px.line(
                trend_df,
                x='Day',
                y='AQI',
                markers=True,
                title='AQI Prediction Trend - Karachi',
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(
                xaxis_title="Day",
                yaxis_title="AQI Value",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📊 Prediction Details"):
                st.json(predictions)

        except requests.exceptions.RequestException as e:
            st.error(f"🌐 Could not connect to the prediction API. Please ensure the Flask API is running on port 5000. Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

with col2:
    st.subheader("🌤️ Current Weather Data")
    try:
        weather_response = requests.get(f"{CURRENT_WEATHER_URL}")
        if weather_response.status_code == 200:
            weather_data = weather_response.json()
            current_weather = weather_data['current_weather']

            st.info("**Real-time weather data used for predictions:**")

            col_a,col_b = st.columns(2)

            with col_a:
                st.metric("🌡️ Temperature",f"{current_weather['temperature']:.1f}°C")
                st.metric("💧 Humidity",f"{current_weather['humidity']:.1f}%")
                st.metric("💨 Wind Speed",f"{current_weather['wind_speed']:.1f} m/s")

            with col_b:
                st.metric("🏔️ Pressure",f"{current_weather['pressure']:.0f} hPa")
                st.metric("🌧️ Precipitation",f"{current_weather['precipitation']:.1f} mm")
                st.metric("🧭 Wind Direction",f"{current_weather['wind_direction']:.0f}°")

            st.subheader("🏭 Current Air Quality")
            aq_col1,aq_col2,aq_col3 = st.columns(3)

            with aq_col1:
                st.metric("🌫️ PM2.5",f"{current_weather['pm25']:.1f} µg/m³")
                st.metric("🏭 PM10",f"{current_weather['pm10']:.1f} µg/m³")

            with aq_col2:
                st.metric("⚡ O₃",f"{current_weather['o3']:.1f} µg/m³")
                st.metric("🔥 CO",f"{current_weather['co']:.1f} ppm")

            with aq_col3:
                st.metric("🚗 NO₂",f"{current_weather['no2']:.1f} µg/m³")
                st.metric("🏭 SO₂",f"{current_weather['so2']:.1f} µg/m³")
        else:
            st.info("🌤️ Current weather data not available")

    except Exception as e:
        st.info(f"Unable to fetch current weather data: {e}")

    st.subheader("🤖 Model Information")
    st.info("""
    **Karachi-Optimized Features:**
    - Real-time weather data from Open-Meteo API
    - Current air quality measurements (PM2.5,PM10,CO,NO₂,O₃,SO₂)
    - Time-based features (hour,day,month,weekday)
    - Weather-pollution interaction features
    - Ridge Regression model with enhanced sensitivity
    """)

    st.success("Predictions based purely on current conditions")

st.markdown("---")
st.caption("Made by Ahsan Ali")
st.caption("GitHub: [Ahsan Ali](https://github.com/AhsanAli-exe)")


