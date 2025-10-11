from unittest import removeResult
import requests
import pandas as pd
from datetime import datetime,timedelta
import numpy as np

OPENWEATHER_API_KEY = "dbe078e1e82116da23cb8e7fe77e1051"
AQICN_TOKEN = "190bb4c220d7cf012723c6a0affb13b166d163b7"

def fetch_aqi_data(city):
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def fetch_weather_data(city,lat,lon):
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'
    }
    
    try:
        response = requests.get(url,params=params)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def compute_time_features(timestamp):
    dt = datetime.fromtimestamp(timestamp)
    return {
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        'weekday': dt.weekday(),
        'is_weekend': dt.weekday() >= 5
    }
    
def process_raw_data(aqi_data,weather_data):
    if not aqi_data or not weather_data:
        return None
    
    features = {}
    
    timestamp = aqi_data['data']['time']['v']
    time_features = compute_time_features(timestamp)
    features.update(time_features)
    
    features['temperature'] = weather_data['main']['temp']
    features['humidity'] = weather_data['main']['humidity']
    features['pressure'] = weather_data['main']['pressure']
    features['wind_speed'] = weather_data['wind']['speed']
    features['wind_deg'] = weather_data['wind'].get('deg', 0)
    
    iaqi = aqi_data['data']['iaqi']
    pollutant_count = 0
    for pollutant in ['pm25','pm10','no2','so2','co','o3']:
        if pollutant in iaqi:
            features[pollutant] = iaqi[pollutant]['v']
            pollutant_count += 1
        else:
            features[pollutant] = 0
    print(f"Number of pollutants: {pollutant_count}")
            
    target = aqi_data['data']['aqi']
            
    return features,target 


city = "karachi"
lat,lon = 24.8607,67.0011

print("fetching current data...")
aqi_data = fetch_aqi_data(city)
weather_data = fetch_weather_data(city,lat,lon)

if aqi_data and weather_data:
    features,target = process_raw_data(aqi_data,weather_data)
    print(f"Target AQI: {target}")
    print("Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
else:
    print("Failed to fetch data - check your API keys!")