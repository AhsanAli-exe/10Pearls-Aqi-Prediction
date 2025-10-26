import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import hopsworks
import warnings
warnings.filterwarnings('ignore')

class AQIPredictor(nn.Module):
    def __init__(self, input_size):
        super(AQIPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def evaluate_model(y_true,y_pred,model_name):
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    mae = mean_absolute_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)
    print(f'{model_name} Performance:')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R2 Score: {r2:.4f}')
    print('\n')
    return {'RMSE': rmse,'MAE': mae,'R2': r2}


project = hopsworks.login(project="aqi_prediction72",api_key_file="hopsworks.key")
fs = project.get_feature_store()

try:
    feature_view = fs.get_feature_view("aqi_prediction_online",version=1)

    try:
        td_version, td_job = feature_view.create_train_test_split(
            test_size=0.2,
            random_state=42,
            description="AQI Prediction Training dataset"
        )
    except:
        print("⚠️ Feature store not ready yet - no data available for training")
        print("📝 Please run data collection and feature engineering first")
        exit(0)

    X_train,X_test,y_train,y_test = feature_view.get_train_test_split(
    training_dataset_version=td_version,
    test_size=0.2,
    random_state=42
    )

except Exception as e:
    print(f"❌ Feature store error: {e}")
    print("📝 Please ensure data collection and feature engineering have run successfully")
    exit(1)

drop_cols = [c for c in ['timestamp','aqi','ts_epoch_ms'] if c in X_train.columns]
feature_columns = [c for c in X_train.columns if c not in drop_cols]

X_train_features = X_train[feature_columns].copy()
X_test_features = X_test[feature_columns].copy()

X_train_clean = X_train_features.fillna(X_train_features.mean(numeric_only=True))
X_test_clean = X_test_features.fillna(X_test_features.mean(numeric_only=True))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test_clean)

model_results = {}
trained_models = {}

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled,y_train)
ridge_predictions = ridge.predict(X_test_scaled)
ridge_results = evaluate_model(y_test, ridge_predictions,"Ridge Regression")
model_results['Ridge Regression'] = ridge_results
trained_models['Ridge Regression'] = ridge

comparison_df = pd.DataFrame(model_results).T
comparison_df = comparison_df.round(4)
best_rmse = comparison_df['RMSE'].min()
best_mae = comparison_df['MAE'].min()
best_r2 = comparison_df['R2'].max()
best_rmse_model = comparison_df[comparison_df['RMSE'] == best_rmse].index[0]
best_mae_model = comparison_df[comparison_df['MAE'] == best_mae].index[0]
best_r2_model = comparison_df[comparison_df['R2'] == best_r2].index[0]

comparison_df.to_csv('models/model_comparison.csv')

mr = project.get_model_registry()
best_model = trained_models['Ridge Regression']
joblib.dump(best_model, 'models/ridge_regression_best.pkl')
joblib.dump(scaler, "models/scaler.pkl")

try:
    ridge_model_registry = mr.sklearn.create_model(
        name="aqi_ridge_regression",
        version=3,
        description="Best Performing AQI Prediction Model - Ridge Regression"
    )
    ridge_model_registry.save("models")
except Exception as e:
    print(f"Error saving model to Hopsworks Model Registry: {e}")


