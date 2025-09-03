import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

start = time.perf_counter()

print('loading model_data.csv..')
df = pd.read_csv('./processed_data/model_data.csv')
y = df['popularity']
x = df.drop('popularity', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print('scaling train and test data..')
scaler = StandardScaler()

x_train_scaled = x_train.copy()
x_test_scaled = x_test.copy()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

n_est = 100
model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)

print('random forest regressor model training..')
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"model performance => n_estimators: {n_est}, rmse: {rmse}, r2: {r2}")

os.makedirs('saved_models', exist_ok=True)
joblib.dump(model, os.path.join('saved_models', 'random_forest_model.joblib'))
joblib.dump(scaler, os.path.join('saved_models', 'scaler.joblib'))
print(f'model and scaler saved at ./saved_models..')
print(f'[fin in {time.perf_counter()-start:.3f} s..]')
