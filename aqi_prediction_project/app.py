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
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_FILE = os.path.join(BASE_DIR,"static","style.css")

if os.path.exists(CSS_FILE):
    local_css(CSS_FILE)


FLASK_API_URL = "http://127.0.0.1:5000/predict"
HEALTH_API_URL = "http://127.0.0.1:5000/health"

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
st.write("Predicting the Air Quality Index for the next 3 days in Karachi.")

with st.sidebar:
    st.title("[Ahsan Ali's LinkedIn](https://www.linkedin.com/in/ahsan--ali)")
    st.sidebar.title("ℹ️ How to Use")
    st.sidebar.write("Click the button below to get the AQI prediction for the next 3 days.")



# API Status Check
with st.expander("🔍 API Status", expanded=False):
    try:
        health_response = requests.get(HEALTH_API_URL, timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("✅ API is healthy and running")
            st.json(health_data)
        else:
            st.error(f"❌ API health check failed: {health_response.status_code}")
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {e}")

if st.button("Get AQI Prediction", type="primary"):
    try:
        with st.spinner("Fetching predictions from the model..."):
            response = requests.get(FLASK_API_URL, timeout=30)
            response.raise_for_status() 
            predictions = response.json()

        if predictions.get('status') == 'success':
            st.success("✅ Prediction successful!")
            
            # Display prediction metadata
            if 'timestamp' in predictions:
                st.caption(f"🕒 Generated at: {predictions['timestamp']}")
            
            st.subheader("3-Day AQI Forecast")
            col1,col2,col3 = st.columns(3)
            days = ["Today","Tomorrow","Day After Tomorrow"]
            aqi_values = [predictions['day1_aqi'],predictions['day2_aqi'],predictions['day3_aqi']]

            for i,col in enumerate([col1,col2,col3]):
                with col:
                    aqi = aqi_values[i]
                    category,color,text_color = get_aqi_category(aqi)
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
            
            # Additional insights
            st.subheader("📊 Insights")
            avg_aqi = sum(aqi_values) / len(aqi_values)
            max_aqi = max(aqi_values)
            min_aqi = min(aqi_values)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average AQI", f"{avg_aqi:.1f}")
            with col2:
                st.metric("Highest AQI", f"{max_aqi:.1f}")
            with col3:
                st.metric("Lowest AQI", f"{min_aqi:.1f}")
                
        else:
            st.error(f"❌ Prediction failed: {predictions.get('error', 'Unknown error')}")

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Could not connect to the prediction API. Please ensure the Flask API is running. Error: {e}")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")


