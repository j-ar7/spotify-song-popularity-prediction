import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./processed_data/model_data.csv')

y = df['track_popularity']
x = df.drop('track_popularity', axis=1)

x = pd.get_dummies(x, columns=['primary_genre'], drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

start = time.perf_counter()
n_est = 60
model = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
# model = lgb.LGBMRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
# model = LinearRegression()

print('model training..')
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)
end = time.perf_counter()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"n_est: {n_est}, rmse: {rmse}, r2: {r2}, time: {end-start}")

os.makedirs('saved_models', exist_ok=True)
joblib.dump(model, os.path.join('saved_models', 'random_forest_model.joblib'))
joblib.dump(scaler, os.path.join('saved_models', 'scaler.joblib'))