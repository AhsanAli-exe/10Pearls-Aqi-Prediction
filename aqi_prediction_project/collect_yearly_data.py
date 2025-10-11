import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime,timedelta
import os
import time

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def collect_weather_data(lat,lon,start_date,end_date):
    print(f"Collecting weather data for {start_date} to {end_date}")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": [
            "temperature_2m", 
            "relative_humidity_2m", 
            "pressure_msl", 
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation"
        ]
    }
    try:
        responses = openmeteo.weather_api(url,params=params)
        response = responses[0]
        
        hourly = response.Hourly()
        hourly_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "humidity": hourly.Variables(1).ValuesAsNumpy(),
            "pressure": hourly.Variables(2).ValuesAsNumpy(),
            "wind_speed": hourly.Variables(3).ValuesAsNumpy(),
            "wind_direction": hourly.Variables(4).ValuesAsNumpy(),
            "precipitation": hourly.Variables(5).ValuesAsNumpy(),
        }
        
        df = pd.DataFrame(hourly_data)
        print(f"✅ Collected {len(df)} hourly weather records")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def collect_air_quality_data(lat,lon,start_date,end_date):
    print(f"Collecting air quality data for {start_date} to {end_date}")
    
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone","sulphur_dioxide"]
    }
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        hourly = response.Hourly()
        air_quality_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "pm10": hourly.Variables(0).ValuesAsNumpy(),
            "pm25": hourly.Variables(1).ValuesAsNumpy(),
            "co": hourly.Variables(2).ValuesAsNumpy(),
            "no2": hourly.Variables(3).ValuesAsNumpy(),
            "o3": hourly.Variables(4).ValuesAsNumpy(),
            "so2": hourly.Variables(5).ValuesAsNumpy(),
        }
        
        df = pd.DataFrame(air_quality_data)
        print(f"✅ Collected {len(df)} hourly air quality records")
        
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def collect_yearly_data_in_chunks(lat,lon,city_name="karachi"):
    os.makedirs("data", exist_ok=True)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    print(f"COLLECTING 1 YEAR OF DATA FOR {city_name.upper()}")
    print(f"Date range: {start_date} to {end_date}")
    print("="*60)
    
    all_weather_data = []
    all_air_quality_data = []
    
    current_date = start_date
    month_count = 0
    while current_date <= end_date:
        month_count += 1
        if current_date.month == 12:
            next_month = current_date.replace(month=1, year=current_date.year + 1)
        else:
            next_month = current_date.replace(month=current_date.month + 1)
        
        chunk_end = min(next_month - timedelta(days=1),end_date)
        
        weather_df = collect_weather_data(lat,lon,current_date,chunk_end)
        if weather_df is not None:
            all_weather_data.append(weather_df)
        time.sleep(1)
        
        air_quality_df = collect_air_quality_data(lat,lon,current_date,chunk_end)
        if air_quality_df is not None:
            all_air_quality_data.append(air_quality_df)
        
        time.sleep(1)
        
        current_date = next_month
    
    print("\n" + "="*60)
    print("Combining data...")
    
    if all_weather_data:
        combined_weather = pd.concat(all_weather_data, ignore_index=True)
        combined_weather.to_csv(f"data/{city_name}_weather_1year.csv", index=False)
        print(f"Weather data saved: {len(combined_weather)} records")
    
    if all_air_quality_data:
        combined_air_quality = pd.concat(all_air_quality_data, ignore_index=True)
        combined_air_quality.to_csv(f"data/{city_name}_air_quality_1year.csv", index=False)
        print(f"Air quality data saved: {len(combined_air_quality)} records")
    
    print(f"\n🎉 SUCCESS! 1 year of data collected for {city_name}")
    return combined_weather,combined_air_quality

lat,lon = 24.8607,67.0011
city = "karachi"

print("STARTING 1-YEAR DATA COLLECTION")
print("This will take several minutes...")
print("="*60)

    
weather_data,air_quality_data = collect_yearly_data_in_chunks(lat,lon,city)
    
    
        
        
