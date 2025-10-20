#!/usr/bin/env python3
"""
Real-time data fetcher for AQI prediction
Fetches current weather and air quality data from Open-Meteo API
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeDataFetcher:
    def __init__(self, lat=24.8607, lon=67.0011, city_name="Karachi"):
        """
        Initialize the real-time data fetcher for Karachi
        
        Args:
            lat (float): Latitude of the city
            lon (float): Longitude of the city
            city_name (str): Name of the city
        """
        self.lat = lat
        self.lon = lon
        self.city_name = city_name
        self.base_url = "https://api.open-meteo.com/v1"
        
    def fetch_current_weather(self):
        """
        Fetch current weather data from Open-Meteo API
        
        Returns:
            dict: Current weather data
        """
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'latitude': self.lat,
                'longitude': self.lon,
                'current': [
                    'temperature_2m',
                    'relative_humidity_2m',
                    'surface_pressure',
                    'wind_speed_10m',
                    'wind_direction_10m',
                    'precipitation'
                ],
                'timezone': 'Asia/Karachi'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            current = data['current']
            weather_data = {
                'temperature': current['temperature_2m'],
                'humidity': current['relative_humidity_2m'],
                'pressure': current['surface_pressure'],
                'wind_speed': current['wind_speed_10m'],
                'wind_direction': current['wind_direction_10m'],
                'precipitation': current['precipitation'],
                'timestamp': current['time']
            }
            
            logger.info(f"Successfully fetched current weather data for {self.city_name}")
            return weather_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in fetch_current_weather: {e}")
            return None
    
    def fetch_current_air_quality(self):
        """
        Fetch current air quality data from Open-Meteo API
        Note: Air quality data may not be available for all locations
        
        Returns:
            dict: Current air quality data or None if not available
        """
        try:
            url = f"{self.base_url}/air-quality"
            params = {
                'latitude': self.lat,
                'longitude': self.lon,
                'current': [
                    'european_aqi',
                    'pm10',
                    'pm2_5',
                    'carbon_monoxide',
                    'nitrogen_dioxide',
                    'sulphur_dioxide',
                    'ozone'
                ],
                'timezone': 'Asia/Karachi'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            current = data['current']
            air_quality_data = {
                'pm10': current['pm10'],
                'pm25': current['pm2_5'],
                'co': current['carbon_monoxide'],
                'no2': current['nitrogen_dioxide'],
                'o3': current['ozone'],
                'so2': current['sulphur_dioxide'],
                'timestamp': current['time']
            }
            
            logger.info(f"Successfully fetched current air quality data for {self.city_name}")
            return air_quality_data
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Air quality data not available for {self.city_name}: {e}")
            # Return default values for air quality data
            return self._get_default_air_quality_data()
        except Exception as e:
            logger.error(f"Unexpected error in fetch_current_air_quality: {e}")
            return self._get_default_air_quality_data()
    
    def _get_default_air_quality_data(self):
        """
        Get default air quality data when API is not available
        
        Returns:
            dict: Default air quality data
        """
        return {
            'pm10': 50.0,  # Moderate level
            'pm25': 25.0,  # Moderate level
            'co': 500.0,   # Moderate level
            'no2': 30.0,   # Moderate level
            'o3': 60.0,    # Moderate level
            'so2': 15.0,   # Moderate level
            'timestamp': datetime.now().isoformat()
        }
    
    def get_current_data(self):
        """
        Fetch both weather and air quality data
        
        Returns:
            dict: Combined current data
        """
        weather_data = self.fetch_current_weather()
        air_quality_data = self.fetch_current_air_quality()
        
        if weather_data is None:
            logger.error("Failed to fetch weather data")
            return None
        
        # Air quality data might be None, use defaults
        if air_quality_data is None:
            air_quality_data = self._get_default_air_quality_data()
            logger.warning("Using default air quality data")
        
        # Combine the data
        current_data = {**weather_data, **air_quality_data}
        
        # Add timestamp
        current_data['fetched_at'] = datetime.now().isoformat()
        
        return current_data

def test_real_time_data():
    """Test function to verify real-time data fetching"""
    fetcher = RealTimeDataFetcher()
    data = fetcher.get_current_data()
    
    if data:
        print("✅ Real-time data fetched successfully!")
        print(f"Temperature: {data['temperature']}°C")
        print(f"Humidity: {data['humidity']}%")
        print(f"PM2.5: {data['pm25']} μg/m³")
        print(f"PM10: {data['pm10']} μg/m³")
        print(f"Timestamp: {data['timestamp']}")
    else:
        print("❌ Failed to fetch real-time data")

if __name__ == "__main__":
    test_real_time_data()