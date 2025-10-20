# 🌬️ AQI Prediction System

A comprehensive Air Quality Index (AQI) prediction system that forecasts air quality for the next 3 days using real-time weather data and machine learning.

## 🚀 Features

- **Real-time Data Collection**: Fetches live weather data from Open-Meteo API
- **Advanced Feature Engineering**: 36+ engineered features including cyclical encoding and interaction features
- **Multiple ML Models**: Ridge Regression, Random Forest, XGBoost, LightGBM, and PyTorch Neural Networks
- **RESTful API**: Flask-based API for predictions
- **Interactive Dashboard**: Streamlit web interface with real-time visualizations
- **CI/CD Pipeline**: GitHub Actions for automated data collection and model training
- **Docker Support**: Containerized deployment for easy scaling

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Real-time     │    │   Feature        │    │   ML Models     │
│   Data API      │───▶│   Engineering    │───▶│   Training      │
│   (Open-Meteo)  │    │   Pipeline       │    │   & Prediction  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Hopsworks      │    │   Flask API     │
                       │   Feature Store  │    │   & Streamlit   │
                       └──────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- pip
- Git

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd aqi_prediction_project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the system**
   ```bash
   # Terminal 1: Start API
   python3 api.py
   
   # Terminal 2: Start Dashboard
   streamlit run app.py
   ```

4. **Access the system**
   - API: http://localhost:5000
   - Dashboard: http://localhost:8501

### Docker Installation

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Access the system**
   - API: http://localhost:5000
   - Dashboard: http://localhost:8501

## 📁 Project Structure

```
aqi_prediction_project/
├── 📊 Data Collection
│   ├── collect_yearly_data.py          # Historical data collection
│   ├── real_time_data.py               # Real-time data fetching
│   └── data/                           # Data storage
│       ├── karachi_weather_1year.csv
│       └── karachi_air_quality_1year.csv
│
├── 🔧 Feature Engineering
│   ├── real_time_features.py           # Real-time feature engineering
│   ├── feature_engineering_eda.ipynb   # EDA and feature creation
│   └── notebooks/                      # Jupyter notebooks
│
├── 🤖 Machine Learning
│   ├── ml_model_training.ipynb         # Model training and evaluation
│   ├── predictor.py                    # Prediction logic
│   └── models/                         # Trained models
│       ├── ridge_regression_best.pkl
│       └── scaler.pkl
│
├── 🌐 API & Dashboard
│   ├── api.py                          # Flask REST API
│   ├── app.py                          # Streamlit dashboard
│   └── static/style.css                # Custom styling
│
├── 🚀 Deployment
│   ├── deploy.py                       # Deployment script
│   ├── Dockerfile                      # Docker configuration
│   └── docker-compose.yml              # Multi-container setup
│
└── 🔄 CI/CD
    └── .github/workflows/               # GitHub Actions
        ├── data_pipeline.yml
        └── mlops_pipeline.yml
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# Data Sources
OPEN_METEO_API_URL=https://api.open-meteo.com/v1

# Model Configuration
MODEL_PATH=models/ridge_regression_best.pkl
SCALER_PATH=models/scaler.pkl

# Location Settings
LATITUDE=24.8607
LONGITUDE=67.0011
CITY_NAME=Karachi
```

## 📊 API Endpoints

### Health Check
```http
GET /health
```
Returns API status and real-time data connectivity.

### Single Prediction
```http
GET /predict/single
```
Returns AQI prediction for current conditions.

### 3-Day Forecast
```http
GET /predict
```
Returns AQI predictions for the next 3 days.

**Response Format:**
```json
{
  "day1_aqi": 45.2,
  "day2_aqi": 52.1,
  "day3_aqi": 38.7,
  "prediction_times": ["2024-01-20 10:00:00", "2024-01-21 10:00:00", "2024-01-22 10:00:00"],
  "status": "success",
  "timestamp": "2024-01-20T10:00:00Z"
}
```

## 🧪 Testing

### Run All Tests
```bash
python3 test_real_time_integration.py
```

### Test Individual Components
```bash
# Test data collection
python3 -c "from real_time_data import RealTimeDataFetcher; fetcher = RealTimeDataFetcher(); print(fetcher.get_current_data())"

# Test feature engineering
python3 -c "from real_time_features import RealTimeFeatureEngineer; engineer = RealTimeFeatureEngineer(); print('Feature engineering ready')"

# Test prediction
python3 -c "from predictor import predict_aqi_real_time; print(f'AQI: {predict_aqi_real_time()}')"
```

## 🔄 CI/CD Pipeline

The system includes automated CI/CD pipelines:

### Data Collection Pipeline
- **Schedule**: Every 6 hours
- **Purpose**: Collect real-time weather and air quality data
- **Output**: JSON files with timestamped data

### Model Training Pipeline
- **Schedule**: Daily at 2 AM
- **Purpose**: Retrain models with latest data
- **Output**: Updated model artifacts

### Full MLOps Pipeline
- **Trigger**: Push to main branch
- **Purpose**: Complete testing, training, and deployment
- **Output**: Deployed system with updated models

## 📈 Model Performance

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Ridge Regression | 12.34 | 8.91 | 0.95 |
| Random Forest | 15.67 | 11.23 | 0.92 |
| XGBoost | 13.45 | 9.78 | 0.94 |
| LightGBM | 14.12 | 10.34 | 0.93 |
| PyTorch NN | 16.23 | 12.45 | 0.91 |

## 🌍 Supported Locations

Currently configured for **Karachi, Pakistan**:
- Latitude: 24.8607
- Longitude: 67.0011

To add support for other locations, update the coordinates in:
- `real_time_data.py`
- `collect_yearly_data.py`

## 🚨 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure Flask API is running on port 5000
   - Check firewall settings
   - Verify API health endpoint

2. **Data Fetching Failed**
   - Check internet connection
   - Verify Open-Meteo API availability
   - Check location coordinates

3. **Model Loading Error**
   - Ensure model files exist in `models/` directory
   - Check file permissions
   - Verify scikit-learn version compatibility

4. **Feature Engineering Error**
   - Check input data format
   - Verify feature names match training data
   - Check for missing values

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Ahsan Ali** - *Initial work* - [LinkedIn](https://www.linkedin.com/in/ahsan--ali)

## 🙏 Acknowledgments

- Open-Meteo for weather data API
- US EPA for AQI calculation standards
- Scikit-learn and PyTorch for ML frameworks
- Streamlit for the dashboard framework

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact: [ahsan.ali@example.com](mailto:ahsan.ali@example.com)

---

**🌬️ Breathe Easy, Predict Smart! 🌬️**