#!/usr/bin/env python3
"""
Real-time feature engineering for AQI prediction
Processes real-time data to create features for model prediction
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RealTimeFeatureEngineer:
    def __init__(self):
        """Initialize the real-time feature engineer"""
        self.feature_names = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
            'precipitation', 'pm10', 'pm25', 'co', 'no2', 'o3', 'so2',
            'hour', 'day', 'month', 'weekday', 'is_weekend',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'aqi_change_1h', 'aqi_change_3h', 'aqi_change_6h',
            'aqi_ma_3h', 'aqi_ma_6h', 'aqi_ma_12h', 'aqi_ma_24h',
            'aqi_lag_1h', 'aqi_lag_3h', 'aqi_lag_6h',
            'temp_humidity_interaction', 'wind_pollution_ratio', 'pressure_stability',
            'season_encoded', 'aqi_category_encoded'
        ]
    
    def calculate_aqi(self, pm25, pm10, o3, no2, co, so2):
        """
        Calculate AQI based on pollutant concentrations using US EPA standards
        
        Args:
            pm25, pm10, o3, no2, co, so2: Pollutant concentrations
            
        Returns:
            float: Calculated AQI value
        """
        def calculate_individual_aqi(conc, breakpoints):
            for i, (c_low, c_high, aqi_low, aqi_high) in enumerate(breakpoints):
                if conc <= c_high:
                    if conc >= c_low:
                        aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (conc - c_low) + aqi_low
                        return round(aqi)
            return 500
        
        # US EPA AQI breakpoints
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
        co_ppm = co / 1145  # Convert μg/m³ to ppm
        aqi_co = calculate_individual_aqi(co_ppm, co_breakpoints)
        aqi_so2 = calculate_individual_aqi(so2, so2_breakpoints)
        
        all_aqi = [aqi_pm25, aqi_pm10, aqi_o3, aqi_no2, aqi_co, aqi_so2]
        return max(all_aqi)
    
    def get_time_features(self, timestamp):
        """
        Extract time-based features from timestamp
        
        Args:
            timestamp: datetime object or ISO string
            
        Returns:
            dict: Time-based features
        """
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = timestamp
        
        # Basic time features
        hour = dt.hour
        day = dt.day
        month = dt.month
        weekday = dt.weekday()
        is_weekend = 1 if weekday >= 5 else 0
        
        # Cyclical encoding for hour and month
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Season encoding (0: Winter, 1: Spring, 2: Summer, 3: Fall)
        if month in [12, 1, 2]:
            season_encoded = 0  # Winter
        elif month in [3, 4, 5]:
            season_encoded = 1  # Spring
        elif month in [6, 7, 8]:
            season_encoded = 2  # Summer
        else:
            season_encoded = 3  # Fall
        
        return {
            'hour': hour,
            'day': day,
            'month': month,
            'weekday': weekday,
            'is_weekend': is_weekend,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'season_encoded': season_encoded
        }
    
    def calculate_derived_features(self, current_data, historical_data=None):
        """
        Calculate derived features from current and historical data
        
        Args:
            current_data: Current weather and air quality data
            historical_data: Historical data for lag features (optional)
            
        Returns:
            dict: Derived features
        """
        # For now, we'll use placeholder values for lag features
        # In production, these would come from a database or cache
        derived_features = {
            'aqi_change_1h': 0.0,  # Placeholder - would need historical data
            'aqi_change_3h': 0.0,  # Placeholder
            'aqi_change_6h': 0.0,  # Placeholder
            'aqi_ma_3h': current_data.get('pm25', 0),  # Using PM2.5 as proxy
            'aqi_ma_6h': current_data.get('pm25', 0),  # Using PM2.5 as proxy
            'aqi_ma_12h': current_data.get('pm25', 0),  # Using PM2.5 as proxy
            'aqi_ma_24h': current_data.get('pm25', 0),  # Using PM2.5 as proxy
            'aqi_lag_1h': current_data.get('pm25', 0),  # Using PM2.5 as proxy
            'aqi_lag_3h': current_data.get('pm25', 0),  # Using PM2.5 as proxy
            'aqi_lag_6h': current_data.get('pm25', 0),  # Using PM2.5 as proxy
        }
        
        return derived_features
    
    def calculate_interaction_features(self, current_data):
        """
        Calculate interaction features
        
        Args:
            current_data: Current weather and air quality data
            
        Returns:
            dict: Interaction features
        """
        temp = current_data.get('temperature', 0)
        humidity = current_data.get('humidity', 0)
        wind_speed = current_data.get('wind_speed', 0)
        pressure = current_data.get('pressure', 0)
        pm25 = current_data.get('pm25', 0)
        
        # Temperature-humidity interaction
        temp_humidity_interaction = temp * humidity
        
        # Wind-pollution ratio
        wind_pollution_ratio = wind_speed / (pm25 + 1) if pm25 > 0 else 0
        
        # Pressure stability (simplified)
        pressure_stability = abs(pressure - 1013.25)  # Deviation from standard pressure
        
        return {
            'temp_humidity_interaction': temp_humidity_interaction,
            'wind_pollution_ratio': wind_pollution_ratio,
            'pressure_stability': pressure_stability
        }
    
    def get_aqi_category_encoded(self, aqi):
        """
        Get AQI category encoding
        
        Args:
            aqi: AQI value
            
        Returns:
            int: Encoded AQI category (0-5)
        """
        if aqi <= 50:
            return 0  # Good
        elif aqi <= 100:
            return 1  # Moderate
        elif aqi <= 150:
            return 2  # Unhealthy for Sensitive Groups
        elif aqi <= 200:
            return 3  # Unhealthy
        elif aqi <= 300:
            return 4  # Very Unhealthy
        else:
            return 5  # Hazardous
    
    def engineer_features(self, current_data):
        """
        Main function to engineer all features from current data
        
        Args:
            current_data: Current weather and air quality data
            
        Returns:
            np.array: Engineered features array
        """
        try:
            # Calculate current AQI
            aqi = self.calculate_aqi(
                current_data.get('pm25', 0),
                current_data.get('pm10', 0),
                current_data.get('o3', 0),
                current_data.get('no2', 0),
                current_data.get('co', 0),
                current_data.get('so2', 0)
            )
            
            # Get time features
            time_features = self.get_time_features(current_data.get('timestamp', datetime.now()))
            
            # Get derived features
            derived_features = self.calculate_derived_features(current_data)
            
            # Get interaction features
            interaction_features = self.calculate_interaction_features(current_data)
            
            # Get AQI category
            aqi_category_encoded = self.get_aqi_category_encoded(aqi)
            
            # Combine all features
            features = {
                # Basic weather and air quality
                'temperature': current_data.get('temperature', 0),
                'humidity': current_data.get('humidity', 0),
                'pressure': current_data.get('pressure', 0),
                'wind_speed': current_data.get('wind_speed', 0),
                'wind_direction': current_data.get('wind_direction', 0),
                'precipitation': current_data.get('precipitation', 0),
                'pm10': current_data.get('pm10', 0),
                'pm25': current_data.get('pm25', 0),
                'co': current_data.get('co', 0),
                'no2': current_data.get('no2', 0),
                'o3': current_data.get('o3', 0),
                'so2': current_data.get('so2', 0),
                
                # Time features
                **time_features,
                
                # Derived features
                **derived_features,
                
                # Interaction features
                **interaction_features,
                
                # AQI category
                'aqi_category_encoded': aqi_category_encoded
            }
            
            # Convert to numpy array in the correct order
            feature_array = np.array([features.get(name, 0) for name in self.feature_names])
            
            logger.info(f"Successfully engineered {len(feature_array)} features")
            return feature_array.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            # Return zero array as fallback
            return np.zeros((1, len(self.feature_names)))

def test_feature_engineering():
    """Test function to verify feature engineering"""
    from real_time_data import RealTimeDataFetcher
    
    # Fetch real data
    fetcher = RealTimeDataFetcher()
    current_data = fetcher.get_current_data()
    
    if current_data:
        # Engineer features
        engineer = RealTimeFeatureEngineer()
        features = engineer.engineer_features(current_data)
        
        print("✅ Feature engineering successful!")
        print(f"Feature shape: {features.shape}")
        print(f"Sample features: {features[0][:10]}")
    else:
        print("❌ Failed to fetch data for feature engineering")

if __name__ == "__main__":
    test_feature_engineering()