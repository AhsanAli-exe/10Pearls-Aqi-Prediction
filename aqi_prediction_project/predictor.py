import joblib
import numpy as np
import os

MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, 'models/ridge_regression_best.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'models/scaler.pkl')

ridge_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def generate_sample_features():
    features = {
        'temperature': np.random.uniform(15, 35),
        'humidity': np.random.uniform(30, 90),
        'pressure': np.random.uniform(1000, 1020),
        'wind_speed': np.random.uniform(0, 20),
        'wind_direction': np.random.uniform(0, 360),
        'precipitation': np.random.uniform(0, 5),
        'pm10': np.random.uniform(20, 150),
        'pm25': np.random.uniform(10, 100),
        'co': np.random.uniform(200, 1000),
        'no2': np.random.uniform(10, 80),
        'o3': np.random.uniform(20, 150),
        'so2': np.random.uniform(5, 30),
        'hour': np.random.randint(0, 24),
        'day': np.random.randint(1, 31),
        'month': np.random.randint(1, 13),
        'weekday': np.random.randint(0, 7),
        'is_weekend': np.random.randint(0, 2),
        'hour_sin': np.random.uniform(-1, 1),
        'hour_cos': np.random.uniform(-1, 1),
        'month_sin': np.random.uniform(-1, 1),
        'month_cos': np.random.uniform(-1, 1),
        'aqi_change_1h': np.random.uniform(-20, 20),
        'aqi_change_3h': np.random.uniform(-40, 40),
        'aqi_change_6h': np.random.uniform(-60, 60),
        'aqi_ma_3h': np.random.uniform(50, 200),
        'aqi_ma_6h': np.random.uniform(50, 200),
        'aqi_ma_12h': np.random.uniform(50, 200),
        'aqi_ma_24h': np.random.uniform(50, 200),
        'aqi_lag_1h': np.random.uniform(50, 200),
        'aqi_lag_3h': np.random.uniform(50, 200),
        'aqi_lag_6h': np.random.uniform(50, 200),
        'temp_humidity_interaction': np.random.uniform(500, 3000),
        'wind_pollution_ratio': np.random.uniform(0, 1),
        'pressure_stability': np.random.uniform(0, 2),
        'season_encoded': np.random.randint(0, 4),
        'aqi_category_encoded': np.random.randint(0, 6),
    }
    return np.array(list(features.values())).reshape(1, -1)

def predict_aqi(features):
    scaled_features = scaler.transform(features)
    prediction = ridge_model.predict(scaled_features)
    return float(prediction[0])
