import joblib
import numpy as np
import os
import requests
from datetime import datetime,timedelta
import pandas as pd
from retry_requests import retry
import openmeteo_requests
import requests_cache

MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR,'models/ridge_regression_best.pkl')
SCALER_PATH = os.path.join(MODEL_DIR,'models/scaler.pkl')

ridge_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

if hasattr(ridge_model,'alpha'):
    original_alpha = ridge_model.alpha
    ridge_model.alpha = original_alpha * 0.1

CITIES = {
    'karachi': {'lat': 24.8607,'lon': 67.0011,'name': 'Karachi, Pakistan'}
}

def fetch_real_time_weather_data(lat,lon,use_cache=True):
    try:
        cache_duration = 300 if use_cache else 0

        if use_cache:
            cache_session = requests_cache.CachedSession('.cache',expire_after=cache_duration)
            session = cache_session
        else:
            session = requests.Session()

        retry_session = retry(session,retries=5,backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m","relative_humidity_2m","pressure_msl",
                "wind_speed_10m","wind_direction_10m","precipitation"
            ]
        }

        air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        air_quality_params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["pm10","pm2_5","carbon_monoxide","nitrogen_dioxide","ozone","sulphur_dioxide"]
        }

        weather_responses = openmeteo.weather_api(weather_url,params=weather_params)
        air_quality_responses = openmeteo.weather_api(air_quality_url,params=air_quality_params)

        weather_response = weather_responses[0]
        air_quality_response = air_quality_responses[0]

        current_weather = weather_response.Current()
        current_air_quality = air_quality_response.Current()

        weather_data = {
            'temperature': max(-50,min(60,current_weather.Variables(0).Value())),
            'humidity': max(0,min(100,current_weather.Variables(1).Value())),
            'pressure': max(900,min(1100,current_weather.Variables(2).Value())),
            'wind_speed': max(0,current_weather.Variables(3).Value()),
            'wind_direction': current_weather.Variables(4).Value() % 360,
            'precipitation': max(0,current_weather.Variables(5).Value()),
        }

        air_quality_data = {
            'pm10': max(0,current_air_quality.Variables(0).Value()),
            'pm25': max(0,current_air_quality.Variables(1).Value()),
            'co': max(0,current_air_quality.Variables(2).Value()),
            'no2': max(0,current_air_quality.Variables(3).Value()),
            'o3': max(0,current_air_quality.Variables(4).Value()),
            'so2': max(0,current_air_quality.Variables(5).Value()),
        }

        if any(np.isnan(list(weather_data.values()) + list(air_quality_data.values()))):
            return None

        return {**weather_data,**air_quality_data}

    except Exception:
        return None

def generate_real_time_features(lat,lon,use_historical_context=True,city='karachi'):
    current_data = fetch_real_time_weather_data(lat,lon)
    if current_data is None:
        return generate_sample_features()

    current_time = datetime.now()

    features = {
        'temperature': current_data['temperature'],
        'humidity': current_data['humidity'],
        'pressure': current_data['pressure'],
        'wind_speed': current_data['wind_speed'],
        'wind_direction': current_data['wind_direction'],
        'precipitation': current_data['precipitation'],
        'pm10': current_data['pm10'],
        'pm25': current_data['pm25'],
        'co': current_data['co'],
        'no2': current_data['no2'],
        'o3': current_data['o3'],
        'so2': current_data['so2'],
        'hour': current_time.hour,
        'day': current_time.day,
        'month': current_time.month,
        'weekday': current_time.weekday(),
        'is_weekend': 1 if current_time.weekday() >= 5 else 0,
        'hour_sin': np.sin(2 * np.pi * current_time.hour / 24),
        'hour_cos': np.cos(2 * np.pi * current_time.hour / 24),
        'month_sin': np.sin(2 * np.pi * current_time.month / 12),
        'month_cos': np.cos(2 * np.pi * current_time.month / 12),
    }

    if use_historical_context:
        try:
            historical_file = os.path.join(MODEL_DIR, f'data/{city}_weather_1year.csv')

            if os.path.exists(historical_file):
                historical_df = pd.read_csv(historical_file)
                historical_df['datetime'] = pd.to_datetime(historical_df['datetime'])

                recent_data = historical_df[
                    historical_df['datetime'] >= (current_time - timedelta(days=7)).replace(tzinfo=historical_df['datetime'].dt.tz)
                ].tail(168)

                if len(recent_data) > 0:
                    if 'aqi' not in recent_data.columns:
                        recent_aqi = [calculate_aqi(row['pm25'], row['pm10'], row['o3'], row['no2'], row['co'], row['so2'])
                                     for _, row in recent_data.iterrows()]
                    else:
                        recent_aqi = recent_data['aqi'].tolist()

                    features.update({
                        'aqi_change_1h': 0, 'aqi_change_3h': 0, 'aqi_change_6h': 0,
                        'aqi_ma_3h': np.mean(recent_aqi[-72:]) if len(recent_aqi) >= 72 else 100,
                        'aqi_ma_6h': np.mean(recent_aqi[-144:]) if len(recent_aqi) >= 144 else 100,
                        'aqi_ma_12h': np.mean(recent_aqi[-288:]) if len(recent_aqi) >= 288 else 100,
                        'aqi_ma_24h': np.mean(recent_aqi[-576:]) if len(recent_aqi) >= 576 else 100,
                        'aqi_lag_1h': recent_aqi[-1] if len(recent_aqi) >= 1 else 100,
                        'aqi_lag_3h': recent_aqi[-1] if len(recent_aqi) >= 1 else 100,
                        'aqi_lag_6h': recent_aqi[-1] if len(recent_aqi) >= 1 else 100,
                    })
                else:
                    features.update({
                        'aqi_change_1h': 0, 'aqi_change_3h': 0, 'aqi_change_6h': 0,
                        'aqi_ma_3h': 100, 'aqi_ma_6h': 100, 'aqi_ma_12h': 100, 'aqi_ma_24h': 100,
                        'aqi_lag_1h': 100, 'aqi_lag_3h': 100, 'aqi_lag_6h': 100,
                    })
            else:
                features.update({
                    'aqi_change_1h': 0, 'aqi_change_3h': 0, 'aqi_change_6h': 0,
                    'aqi_ma_3h': 100, 'aqi_ma_6h': 100, 'aqi_ma_12h': 100, 'aqi_ma_24h': 100,
                    'aqi_lag_1h': 100, 'aqi_lag_3h': 100, 'aqi_lag_6h': 100,
                })
        except Exception:
            features.update({
                'aqi_change_1h': 0, 'aqi_change_3h': 0, 'aqi_change_6h': 0,
                'aqi_ma_3h': 100, 'aqi_ma_6h': 100, 'aqi_ma_12h': 100, 'aqi_ma_24h': 100,
                'aqi_lag_1h': 100, 'aqi_lag_3h': 100, 'aqi_lag_6h': 100,
            })

    features.update({
        'temp_humidity_interaction': features['temperature'] * features['humidity'],
        'wind_pollution_ratio': features['wind_speed'] / (features['pm25'] + 1),
        'pressure_stability': 0,
        'season_encoded': get_season_encoded(features['month']),
        'aqi_category_encoded': 0,
    })

    return np.array(list(features.values())).reshape(1, -1)

def get_season_encoded(month):
    season_map = {
        12: 0, 1: 0, 2: 0,
        3: 1, 4: 1, 5: 1,
        6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3
    }
    return season_map.get(month, 0)

def calculate_aqi(pm25, pm10, o3, no2, co, so2):
    def calculate_individual_aqi(conc, breakpoints):
        for i, (c_low, c_high, aqi_low, aqi_high) in enumerate(breakpoints):
            if conc <= c_high:
                if conc >= c_low:
                    aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (conc - c_low) + aqi_low
                    return round(aqi)
        return 500

    pm25_breakpoints = [
        (0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500, 301, 500)
    ]

    pm10_breakpoints = [
        (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
        (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)
    ]

    o3_breakpoints = [
        (0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
        (86, 105, 151, 200), (106, 200, 201, 300)
    ]

    no2_breakpoints = [
        (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
        (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)
    ]

    co_breakpoints = [
        (0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)
    ]

    so2_breakpoints = [
        (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
        (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500)
    ]

    aqi_pm25 = calculate_individual_aqi(pm25, pm25_breakpoints)
    aqi_pm10 = calculate_individual_aqi(pm10, pm10_breakpoints)
    aqi_o3 = calculate_individual_aqi(o3, o3_breakpoints)
    aqi_no2 = calculate_individual_aqi(no2, no2_breakpoints)
    co_ppm = co / 1145
    aqi_co = calculate_individual_aqi(co_ppm, co_breakpoints)
    aqi_so2 = calculate_individual_aqi(so2, so2_breakpoints)

    all_aqi = [aqi_pm25, aqi_pm10, aqi_o3, aqi_no2, aqi_co, aqi_so2]
    return max(all_aqi)

def generate_sample_features():
    features = {
        'temperature': np.random.uniform(15,35),
        'humidity': np.random.uniform(30,90),
        'pressure': np.random.uniform(1000,1020),
        'wind_speed': np.random.uniform(0,20),
        'wind_direction': np.random.uniform(0,360),
        'precipitation': np.random.uniform(0,5),
        'pm10': np.random.uniform(20,150),
        'pm25': np.random.uniform(10,100),
        'co': np.random.uniform(200,1000),
        'no2': np.random.uniform(10,80),
        'o3': np.random.uniform(20,150),
        'so2': np.random.uniform(5,30),
        'hour': np.random.randint(0,24),
        'day': np.random.randint(1,31),
        'month': np.random.randint(1,13),
        'weekday': np.random.randint(0,7),
        'is_weekend': np.random.randint(0,2),
        'hour_sin': np.random.uniform(-1,1),
        'hour_cos': np.random.uniform(-1,1),
        'month_sin': np.random.uniform(-1,1),
        'month_cos': np.random.uniform(-1,1),
        'aqi_change_1h': np.random.uniform(-20,20),
        'aqi_change_3h': np.random.uniform(-40,40),
        'aqi_change_6h': np.random.uniform(-60,60),
        'aqi_ma_3h': np.random.uniform(50,200),
        'aqi_ma_6h': np.random.uniform(50,200),
        'aqi_ma_12h': np.random.uniform(50,200),
        'aqi_ma_24h': np.random.uniform(50,200),
        'aqi_lag_1h': np.random.uniform(50,200),
        'aqi_lag_3h': np.random.uniform(50,200),
        'aqi_lag_6h': np.random.uniform(50,200),
        'temp_humidity_interaction': np.random.uniform(500,3000),
        'wind_pollution_ratio': np.random.uniform(0,1),
        'pressure_stability': np.random.uniform(0,2),
        'season_encoded': np.random.randint(0,4),
        'aqi_category_encoded': np.random.randint(0,6),
    }
    return np.array(list(features.values())).reshape(1,-1)

def generate_features_with_cached_data(current_data, current_time, city='karachi', use_historical_context=True):
    features = {
        'temperature': current_data['temperature'],
        'humidity': current_data['humidity'],
        'pressure': current_data['pressure'],
        'wind_speed': current_data['wind_speed'],
        'wind_direction': current_data['wind_direction'],
        'precipitation': current_data['precipitation'],
        'pm10': current_data['pm10'],
        'pm25': current_data['pm25'],
        'co': current_data['co'],
        'no2': current_data['no2'],
        'o3': current_data['o3'],
        'so2': current_data['so2'],
        'hour': current_time.hour,
        'day': current_time.day,
        'month': current_time.month,
        'weekday': current_time.weekday(),
        'is_weekend': 1 if current_time.weekday() >= 5 else 0,
        'hour_sin': np.sin(2 * np.pi * current_time.hour / 24),
        'hour_cos': np.cos(2 * np.pi * current_time.hour / 24),
        'month_sin': np.sin(2 * np.pi * current_time.month / 12),
        'month_cos': np.cos(2 * np.pi * current_time.month / 12),
    }

    if use_historical_context:
        try:
            historical_file = os.path.join(MODEL_DIR, f'data/{city}_weather_1year.csv')

            if os.path.exists(historical_file):
                historical_df = pd.read_csv(historical_file)
                historical_df['datetime'] = pd.to_datetime(historical_df['datetime'])

                recent_data = historical_df[
                    historical_df['datetime'] >= (current_time - timedelta(days=7)).replace(tzinfo=historical_df['datetime'].dt.tz)
                ].tail(168)

                if len(recent_data) > 0:
                    if 'aqi' not in recent_data.columns:
                        recent_aqi = [calculate_aqi(row['pm25'], row['pm10'], row['o3'], row['no2'], row['co'], row['so2'])
                                     for _, row in recent_data.iterrows()]
                    else:
                        recent_aqi = recent_data['aqi'].tolist()

                    features.update({
                        'aqi_change_1h': 0, 'aqi_change_3h': 0, 'aqi_change_6h': 0,
                        'aqi_ma_3h': np.mean(recent_aqi[-72:]) if len(recent_aqi) >= 72 else 100,
                        'aqi_ma_6h': np.mean(recent_aqi[-144:]) if len(recent_aqi) >= 144 else 100,
                        'aqi_ma_12h': np.mean(recent_aqi[-288:]) if len(recent_aqi) >= 288 else 100,
                        'aqi_ma_24h': np.mean(recent_aqi[-576:]) if len(recent_aqi) >= 576 else 100,
                        'aqi_lag_1h': recent_aqi[-1] if len(recent_aqi) >= 1 else 100,
                        'aqi_lag_3h': recent_aqi[-1] if len(recent_aqi) >= 1 else 100,
                        'aqi_lag_6h': recent_aqi[-1] if len(recent_aqi) >= 1 else 100,
                    })
                else:
                    features.update({
                        'aqi_change_1h': 0, 'aqi_change_3h': 0, 'aqi_change_6h': 0,
                        'aqi_ma_3h': 100, 'aqi_ma_6h': 100, 'aqi_ma_12h': 100, 'aqi_ma_24h': 100,
                        'aqi_lag_1h': 100, 'aqi_lag_3h': 100, 'aqi_lag_6h': 100,
                    })
            else:
                features.update({
                    'aqi_change_1h': 0, 'aqi_change_3h': 0, 'aqi_change_6h': 0,
                    'aqi_ma_3h': 100, 'aqi_ma_6h': 100, 'aqi_ma_12h': 100, 'aqi_ma_24h': 100,
                    'aqi_lag_1h': 100, 'aqi_lag_3h': 100, 'aqi_lag_6h': 100,
                })
        except Exception:
            features.update({
                'aqi_change_1h': 0, 'aqi_change_3h': 0, 'aqi_change_6h': 0,
                'aqi_ma_3h': 100, 'aqi_ma_6h': 100, 'aqi_ma_12h': 100, 'aqi_ma_24h': 100,
                'aqi_lag_1h': 100, 'aqi_lag_3h': 100, 'aqi_lag_6h': 100,
            })

    features.update({
        'temp_humidity_interaction': features['temperature'] * features['humidity'],
        'wind_pollution_ratio': features['wind_speed'] / (features['pm25'] + 1),
        'pressure_stability': 0,
        'season_encoded': get_season_encoded(features['month']),
        'aqi_category_encoded': 0,
    })

    return np.array(list(features.values())).reshape(1, -1)

def generate_features_with_cached_data_and_time_progression(current_data, future_time, city='karachi', use_historical_context=True, day_offset=0):
    variation_factor = 1.0 + (day_offset * 0.05)

    features = {
        'temperature': max(10, min(40, current_data['temperature'] * variation_factor)),
        'humidity': max(20, min(90, current_data['humidity'] * variation_factor)),
        'pressure': max(990, min(1030, current_data['pressure'] * variation_factor)),
        'wind_speed': max(0, min(25, current_data['wind_speed'] * variation_factor)),
        'wind_direction': (current_data['wind_direction'] + day_offset * 15) % 360,
        'precipitation': max(0, current_data['precipitation'] * variation_factor),
        'pm10': max(10, min(200, current_data['pm10'] * variation_factor)),
        'pm25': max(5, min(150, current_data['pm25'] * variation_factor)),
        'co': max(100, min(1500, current_data['co'] * variation_factor)),
        'no2': max(5, min(100, current_data['no2'] * variation_factor)),
        'o3': max(10, min(200, current_data['o3'] * variation_factor)),
        'so2': max(2, min(50, current_data['so2'] * variation_factor)),
        'hour': future_time.hour,
        'day': future_time.day,
        'month': future_time.month,
        'weekday': future_time.weekday(),
        'is_weekend': 1 if future_time.weekday() >= 5 else 0,
        'hour_sin': np.sin(2 * np.pi * future_time.hour / 24),
        'hour_cos': np.cos(2 * np.pi * future_time.hour / 24),
        'month_sin': np.sin(2 * np.pi * future_time.month / 12),
        'month_cos': np.cos(2 * np.pi * future_time.month / 12),
    }

    features.update({
        'aqi_change_1h': 0, 'aqi_change_3h': 0, 'aqi_change_6h': 0,
        'aqi_ma_3h': 100, 'aqi_ma_6h': 100, 'aqi_ma_12h': 100, 'aqi_ma_24h': 100,
        'aqi_lag_1h': 100, 'aqi_lag_3h': 100, 'aqi_lag_6h': 100,
    })

    features.update({
        'temp_humidity_interaction': features['temperature'] * features['humidity'],
        'wind_pollution_ratio': features['wind_speed'] / (features['pm25'] + 1),
        'pressure_stability': 0,
        'season_encoded': get_season_encoded(features['month']),
        'aqi_category_encoded': 0,
    })

    return np.array(list(features.values())).reshape(1, -1)

def predict_aqi(features):
    scaled_features = scaler.transform(features)
    current_weather_features = scaled_features[0][:12]
    base_prediction = float(ridge_model.predict(scaled_features)[0])
    pm25_scaled = float(current_weather_features[6])
    pm10_scaled = float(current_weather_features[5])
    pollution_adjustment = (pm25_scaled * 15) + (pm10_scaled * 10)
    adjusted_prediction = base_prediction + pollution_adjustment
    return max(0, min(500, adjusted_prediction))
