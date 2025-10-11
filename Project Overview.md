### **Project Overview**



**Let's predict the Air Quality Index (AQI) in your city in the next 3 days, using a 100% serverless stack. This project involves building an end-to-end machine learning pipeline for AQI forecasting with automated data collection, feature engineering, model training, and real-time predictions through a web dashboard**



**Required technologies and tools:**



**Python**

**Scikit-learn**

**TensorFlow**

**Hopsworks or Vertex AI**

**Apache Airflow or GitHub Actions**

**Streamlit**

**Flask**

**AQICN or OpenWeather APIs**

**SHAP**

**Git**



**Key Features**



**Feature Pipeline Development**



* **Fetch raw weather and pollutant data from external APIs like AQICN or OpenWeather**
* **Compute features from raw data including time-based features (hour, day, month) and derived features like AQI change rate**
* **Store processed features in Feature Store (Hopsworks or Vertex AI)**





**Historical Data Backfill**



* **Run feature pipeline for past dates to generate training data**
* **Create comprehensive dataset for model training and evaluation**





**Training Pipeline Implementation**



* **Fetch historical features and targets from Feature Store**
* **Experiment with various ML models (Random Forest, Ridge Regression, TensorFlow/PyTorch)**
* **Evaluate performance using RMSE, MAE, and R² metrics**
* **Store trained models in Model Registry**



**Automated CI/CD Pipeline**



* **Feature pipeline runs every hour automatically**
* **Training pipeline runs daily for model updates**
* **Use Apache Airflow, GitHub Actions, or similar tools**



**Web Application Dashboard**

**•**

**Load models and features from Feature Store**

**•**

**Compute real-time predictions for next 3 days**

**•**

**Display interactive dashboard with Streamlit/Gradio and Flask/FastAPI**

**•**

**Advanced Analytics Features**

**•**

**Perform Exploratory Data Analysis (EDA) to identify trends**

**•**

**Use SHAP or LIME for feature importance explanations**

**•**

**Implement alerts for hazardous AQI levels**

**•**

**Support multiple forecasting models from statistical to deep learning**

