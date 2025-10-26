import pandas as pd
import numpy as np
import hopsworks
import os
import warnings
warnings.filterwarnings("ignore")


def calculate_aqi(pm25,pm10,o3,no2,co,so2):
    def calculate_individual_aqi(conc,breakpoints):
        for i, (c_low,c_high,aqi_low,aqi_high) in enumerate(breakpoints):
            if conc<=c_high:
                if conc >= c_low:
                    aqi =  ((aqi_high - aqi_low) / (c_high - c_low)) * (conc - c_low) + aqi_low
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
        (0, 53, 0, 50), 
        (54, 100, 51, 100), 
        (101, 360, 101, 150),
        (361, 649, 151, 200), 
        (650, 1249, 201, 300), 
        (1250, 2049, 301, 500)
    ]
    

    co_breakpoints = [
        (0, 4.4, 0, 50),      # 0-4.4 ppm
        (4.5, 9.4, 51, 100),  # 4.5-9.4 ppm
        (9.5, 12.4, 101, 150), # 9.5-12.4 ppm
        (12.5, 15.4, 151, 200), # 12.5-15.4 ppm
        (15.5, 30.4, 201, 300), # 15.5-30.4 ppm
        (30.5, 50.4, 301, 500)  # 30.5-50.4 ppm
    ]
    
    so2_breakpoints = [
        (0, 35, 0, 50), 
        (36, 75, 51, 100), 
        (76, 185, 101, 150),
        (186, 304, 151, 200), 
        (305, 604, 201, 300), 
        (605, 1004, 301, 500)
    ]
    
    aqi_pm25 = calculate_individual_aqi(pm25,pm25_breakpoints)
    aqi_pm10 = calculate_individual_aqi(pm10,pm10_breakpoints)
    aqi_o3 = calculate_individual_aqi(o3,o3_breakpoints)
    aqi_no2 = calculate_individual_aqi(no2,no2_breakpoints)
    co_ppm = co/1145
    aqi_co = calculate_individual_aqi(co_ppm,co_breakpoints)
    aqi_so2 = calculate_individual_aqi(so2,so2_breakpoints)
    all_aqi = [aqi_pm25,aqi_pm10,aqi_o3,aqi_no2,aqi_co,aqi_so2]
    return max(all_aqi)



# Check if data files exist
if not os.path.exists("data/karachi_weather_1year.csv") or not os.path.exists("data/karachi_air_quality_1year.csv"):
    print("❌ Data files not found. Please run data collection first.")
    print("📝 Expected files: data/karachi_weather_1year.csv and data/karachi_air_quality_1year.csv")
    exit(1)

weather_df = pd.read_csv("data/karachi_weather_1year.csv")
air_quality_df = pd.read_csv("data/karachi_air_quality_1year.csv")
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
air_quality_df['datetime'] = pd.to_datetime(air_quality_df['datetime'])
merged_df = pd.merge(weather_df,air_quality_df,on='datetime',how='inner')
merged_df['aqi'] = merged_df.apply(
    lambda row: calculate_aqi(row['pm25'],row['pm10'],row['o3'],row['no2'],row['co'],row['so2']),axis=1
)
def get_aqi_category(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'
merged_df['aqi_category'] = merged_df['aqi'].apply(get_aqi_category)
merged_df['hour'] = merged_df['datetime'].dt.hour
merged_df['day'] = merged_df['datetime'].dt.day
merged_df['month'] = merged_df['datetime'].dt.month
merged_df['weekday'] = merged_df['datetime'].dt.weekday
merged_df['is_weekend'] = (merged_df['weekday'] >= 5).astype(int)
merged_df['season'] = merged_df['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})
merged_df['hour_sin'] = np.sin(2 * np.pi * merged_df['hour'] / 24)
merged_df['hour_cos'] = np.cos(2 * np.pi * merged_df['hour'] / 24)
merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12)

merged_df = merged_df.sort_values(by='datetime').reset_index(drop=True)
merged_df['aqi_change_1h'] = merged_df['aqi'].diff()
merged_df['aqi_change_3h'] = merged_df['aqi'].diff(3)
merged_df['aqi_change_6h'] = merged_df['aqi'].diff(6)
merged_df['aqi_ma_3h'] = merged_df['aqi'].rolling(window=3).mean()
merged_df['aqi_ma_6h'] = merged_df['aqi'].rolling(window=6).mean()
merged_df['aqi_ma_12h'] = merged_df['aqi'].rolling(window=12).mean()
merged_df['aqi_ma_24h'] = merged_df['aqi'].rolling(window=24).mean()
merged_df['aqi_lag_1h'] = merged_df['aqi'].shift(1)
merged_df['aqi_lag_3h'] = merged_df['aqi'].shift(3)
merged_df['aqi_lag_6h'] = merged_df['aqi'].shift(6)

merged_df['temp_humidity_interaction'] = merged_df['temperature'] * merged_df['humidity']
merged_df['wind_pollution_ratio'] = merged_df['wind_speed'] / (merged_df['pm25'] + 1)
merged_df['pressure_stability'] = merged_df['pressure'].rolling(window=6).std()
merged_df = merged_df.sort_values(by='datetime').reset_index(drop=True)

lag_rolling_cols = [
    'aqi_change_1h','aqi_change_3h','aqi_change_6h',
    'aqi_ma_3h','aqi_ma_6h','aqi_ma_12h','aqi_ma_24h',
    'aqi_lag_1h','aqi_lag_3h','aqi_lag_6h',
    'pressure_stability'
]

merged_df = merged_df.dropna(subset=[c for c in lag_rolling_cols if c in merged_df.columns])



feature_df = merged_df.copy()
feature_df['timestamp'] = pd.to_datetime(feature_df['datetime'])
feature_df = feature_df.drop('datetime',axis=1)
feature_df['season_encoded' ] = feature_df['season'].map(
{'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3})

feature_df = feature_df.drop('season', axis=1)
feature_df['aqi_category_encoded' ] = feature_df['aqi_category' ].map({
'Good': 0, 'Moderate': 1, 'Unhealthy for Sensitive Groups': 2,
'Unhealthy': 3, 'Very Unhealthy': 4, 'Hazardous': 5
})
feature_df = feature_df.drop('aqi_category', axis=1)

y = feature_df['aqi'].copy()
X = feature_df.drop(columns=['aqi'])

ts = pd.to_datetime(X['timestamp'], utc=True, errors='coerce')
X['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)

# Creating integer primary key
X['ts_epoch_ms'] = (ts.view('int64') // 10**6).astype('int64')

X.columns = [c.strip().lower().replace(' ', '_') for c in X.columns]
for col in X.columns:
    if col not in ('timestamp','ts_epoch_ms') and X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')


X = X.dropna()
X = X.sort_values('timestamp').drop_duplicates(subset=['ts_epoch_ms'],keep='last').reset_index(drop=True)
labels_df = feature_df[['timestamp','aqi']].copy()
labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'], utc=True, errors='coerce').dt.tz_convert('UTC').dt.tz_localize(None)
labels_df = labels_df.dropna().sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)

try:
    # Check if Hopsworks API key exists
    if not os.path.exists("hopsworks.key"):
        print("❌ Hopsworks API key not found")
        print("📝 Please add HOPSWORKS_API_KEY to GitHub Secrets")
        exit(1)

    print("🔗 Connecting to Hopsworks...")
    project = hopsworks.login(
        project='aqi_prediction72',
        api_key_file="hopsworks.key"
    )
    fs = project.get_feature_store()
    print("✅ Connected to Hopsworks Feature Store")

    FG_NAME = "aqi_features_on"
    FG_VER = 1

    aqi_fg = fs.create_feature_group(
        name=FG_NAME,
        version=FG_VER,
        description="AQI features with int PK for online serving",
        primary_key=["ts_epoch_ms"],
        event_time="timestamp",
        online_enabled=True
    )
    print("📦 Created feature group for features")
    aqi_fg.insert(X,write_options={"wait_for_job": False})

    LABEL_FG_NAME = "aqi_labels_on"
    LABEL_FG_VER = 1
    aqi_labels = fs.create_feature_group(
        name=LABEL_FG_NAME,
        version=LABEL_FG_VER,
        description="AQI labels aligned by timestamp",
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False
    )
    print("📦 Created feature group for labels")
    aqi_labels.insert(labels_df,write_options={"wait_for_job": False})

    fv_query = aqi_fg.select_all().join(
        aqi_labels.select(['timestamp','aqi']),
        on=['timestamp']
    )

    FV_NAME = "aqi_prediction_online"
    FV_VER = 1

    feature_view = fs.create_feature_view(
        name=FV_NAME,
        version=FV_VER,
        description="Online features (int PK) joined with offline labels on timestamp",
        query=fv_query,
        labels=["aqi"]
    )
    print("✅ Created feature view successfully")

except Exception as e:
    print(f"❌ Hopsworks error: {e}")
    print("📝 This is expected if running locally without Hopsworks setup")
    print("✅ Local data processing completed successfully")




    






