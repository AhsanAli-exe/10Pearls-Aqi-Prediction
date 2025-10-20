#!/usr/bin/env python3
"""
Test script for real-time data integration
Tests the complete pipeline from data fetching to prediction
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from real_time_data import RealTimeDataFetcher
from real_time_features import RealTimeFeatureEngineer
from predictor import predict_aqi_real_time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_fetching():
    """Test real-time data fetching"""
    print("=" * 50)
    print("Testing Real-Time Data Fetching")
    print("=" * 50)
    
    fetcher = RealTimeDataFetcher()
    data = fetcher.get_current_data()
    
    if data:
        print("✅ Data fetching successful!")
        print(f"Temperature: {data.get('temperature', 'N/A')}°C")
        print(f"Humidity: {data.get('humidity', 'N/A')}%")
        print(f"PM2.5: {data.get('pm25', 'N/A')} μg/m³")
        print(f"PM10: {data.get('pm10', 'N/A')} μg/m³")
        print(f"Wind Speed: {data.get('wind_speed', 'N/A')} m/s")
        print(f"Timestamp: {data.get('timestamp', 'N/A')}")
        return True
    else:
        print("❌ Data fetching failed!")
        return False

def test_feature_engineering():
    """Test feature engineering"""
    print("\n" + "=" * 50)
    print("Testing Feature Engineering")
    print("=" * 50)
    
    # Fetch data
    fetcher = RealTimeDataFetcher()
    data = fetcher.get_current_data()
    
    if not data:
        print("❌ Cannot test feature engineering - no data available")
        return False
    
    # Engineer features
    engineer = RealTimeFeatureEngineer()
    features = engineer.engineer_features(data)
    
    if features is not None:
        print("✅ Feature engineering successful!")
        print(f"Feature shape: {features.shape}")
        print(f"Number of features: {features.shape[1]}")
        print(f"Sample features (first 10): {features[0][:10]}")
        return True
    else:
        print("❌ Feature engineering failed!")
        return False

def test_prediction():
    """Test AQI prediction"""
    print("\n" + "=" * 50)
    print("Testing AQI Prediction")
    print("=" * 50)
    
    prediction = predict_aqi_real_time()
    
    if prediction is not None:
        print("✅ Prediction successful!")
        print(f"Predicted AQI: {prediction:.2f}")
        
        # Categorize AQI
        if prediction <= 50:
            category = "Good"
        elif prediction <= 100:
            category = "Moderate"
        elif prediction <= 150:
            category = "Unhealthy for Sensitive Groups"
        elif prediction <= 200:
            category = "Unhealthy"
        elif prediction <= 300:
            category = "Very Unhealthy"
        else:
            category = "Hazardous"
        
        print(f"AQI Category: {category}")
        return True
    else:
        print("❌ Prediction failed!")
        return False

def test_complete_pipeline():
    """Test the complete pipeline"""
    print("\n" + "=" * 50)
    print("Testing Complete Pipeline")
    print("=" * 50)
    
    # Test data fetching
    data_success = test_data_fetching()
    
    # Test feature engineering
    feature_success = test_feature_engineering()
    
    # Test prediction
    prediction_success = test_prediction()
    
    # Summary
    print("\n" + "=" * 50)
    print("Pipeline Test Summary")
    print("=" * 50)
    print(f"Data Fetching: {'✅ PASS' if data_success else '❌ FAIL'}")
    print(f"Feature Engineering: {'✅ PASS' if feature_success else '❌ FAIL'}")
    print(f"Prediction: {'✅ PASS' if prediction_success else '❌ FAIL'}")
    
    overall_success = data_success and feature_success and prediction_success
    print(f"Overall Pipeline: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    return overall_success

if __name__ == "__main__":
    print("Starting Real-Time Integration Tests...")
    print("This will test the complete pipeline from data fetching to prediction.")
    print("Make sure you have an internet connection for API calls.\n")
    
    success = test_complete_pipeline()
    
    if success:
        print("\n🎉 All tests passed! Your real-time integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)