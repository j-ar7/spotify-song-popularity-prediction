import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import time

start = time.perf_counter()

print('loading model and scaler..')
model = joblib.load('saved_models/random_forest_model.joblib')
scaler = joblib.load('saved_models/scaler.joblib')

print('loading model_data.csv..')
df = pd.read_csv('processed_data/model_data.csv')

output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)

heatmap_features = [
	'popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness',
	'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
	'valence', 'tempo', 'time_signature', 'combined_artist_popularity',
	'combined_following', 'release_year', 'mode', 'explicit'
]

heatmap_df = df[heatmap_features]
correlation_matrix = heatmap_df.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Feature Correlation Matrix', fontsize=18)

output_path = os.path.join(output_dir, 'feature_correlation_heatmap.png')
plt.savefig(output_path, bbox_inches='tight')
print(f"heatmap saved successfully to: {output_path}..")
plt.close()

y = df['popularity']
X = df.drop('popularity', axis=1)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

N = 15
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(N)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title(f'{N} most important features for popularity', fontsize=16)
plt.xlabel('Importance')
plt.ylabel('Feature')

plt.savefig(os.path.join(output_dir, 'feature_importance.png'), bbox_inches='tight')
print(f"feature importance plot saved to: {output_dir}/feature_importance.png..")
plt.close()

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

plt.figure(figsize=(10, 10))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.plot([0, 100], [0, 100], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs. Predicted Popularity', fontsize=16)
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.xlim(0, 100)
plt.ylim(0, 100)

plt.savefig(os.path.join(output_dir, 'prediction_scatter_plot.png'), bbox_inches='tight')
print(f"prediction scatter plot saved to: {output_dir}/prediction_scatter_plot.png..")
plt.close()
print(f'[fin in {time.perf_counter()-start:.3f} s..]')
