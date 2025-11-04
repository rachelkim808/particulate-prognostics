import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

csv_file = "FortLeeWinterMinus1"

# Load the datasets
X_train_scaled_m1 = joblib.load('X_train_scaled_m1.pkl')
y_train_m1 = joblib.load('y_train_m1.pkl')
X_test_scaled_m1 = joblib.load('X_test_scaled_m1.pkl')
y_test_m1 = joblib.load('y_test_m1.pkl')


# Assuming these are the features in the order they were scaled
features = ['pm25-1', 'temp', 'visibility', 'winddir', 'windspeed', 'precip', 'solarradiation', 'cloudcover', 'humidity']

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a base model
rf = RandomForestRegressor(random_state=30)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

# Fit the grid search to the data
grid_search.fit(X_train_scaled_m1, y_train_m1)

# Best estimator
best_rf = grid_search.best_estimator_

# Print best parameters
print("Best Parameters:", grid_search.best_params_)

# Prediction and evaluation
y_pred = best_rf.predict(X_test_scaled_m1)

# Evaluation Metrics
mae = mean_absolute_error(y_test_m1, y_pred)
mse = mean_squared_error(y_test_m1, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_m1, y_pred)
accuracy_like_metric = np.mean(np.abs(y_test_m1 - y_pred) <= 5)
print(f'Accuracy-like metric (with tolerance ±5): {accuracy_like_metric:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared Score: {r2:.2f}')

# Feature Importance Plot
feature_importance = best_rf.feature_importances_
sorted_idx = feature_importance.argsort()
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
plt.xlabel('Importance')
plt.title('Feature Importance from Random Forest')
plt.show()

# Density Heatmap
plt.figure(figsize=(10, 6))
sns.kdeplot(x=y_test_m1, y=y_pred, cmap="Blues", fill=True, cbar=True)
plt.xlabel('Actual PM2.5')
plt.ylabel('Predicted PM2.5')
plt.title('Density Heatmap from Random Forest')
plt.xlim(-5, 50)
plt.ylim(-5, 50)
plt.show()

# 2D Histogram
plt.figure(figsize=(10, 6))
plt.hist2d(y_test_m1, y_pred, bins=(30, 30), cmap='Blues')
plt.colorbar(label='Frequency')
plt.xlabel('Actual PM2.5')
plt.ylabel('Predicted PM2.5')
plt.title('Actual vs Predicted PM2.5 2D Histogram from Random Forest')
plt.show()

# Save the best model
joblib.dump(best_rf, f'random_forest_model_{csv_file.split(".")[0]}_updated.pkl')

# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# csv_file = "FortLeeWinterMinus1"

# # Load the datasets
# X_train_scaled = joblib.load('X_train_scaled.pkl')
# y_train = joblib.load('y_train.pkl')
# X_test_scaled = joblib.load('X_test_scaled.pkl')
# y_test = joblib.load('y_test.pkl')

# # Assuming these are the features in the order they were scaled
# features = ['pm25-1', 'pm25-2', 'temp', 'visibility', 'winddir', 'windspeed', 'precip', 'solarradiation', 'cloudcover', 'humidity']

# # Define parameter grid for Random Forest
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Create a base model
# rf = RandomForestRegressor(random_state=30)

# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')

# # Fit the grid search to the data
# grid_search.fit(X_train_scaled, y_train)

# # Best estimator
# best_rf = grid_search.best_estimator_

# # Print best parameters
# print("Best Parameters:", grid_search.best_params_)

# # Prediction and evaluation
# y_pred = best_rf.predict(X_test_scaled)

# # Evaluation Metrics
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
# accuracy_like_metric = np.mean(np.abs(y_test - y_pred) <= 5)
# print(f'Accuracy-like metric (with tolerance ±5): {accuracy_like_metric:.2f}')
# print(f'Mean Absolute Error (MAE): {mae:.2f}')
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
# print(f'R-squared Score: {r2:.2f}')

# # Feature Importance Plot
# feature_importance = best_rf.feature_importances_
# sorted_idx = feature_importance.argsort()
# plt.figure(figsize=(10, 6))
# plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
# plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
# plt.xlabel('Importance')
# plt.title('Feature Importance from Random Forest')
# plt.show()

# # Density Heatmap
# plt.figure(figsize=(10, 6))
# sns.kdeplot(x=y_test, y=y_pred, cmap="Blues", fill=True, cbar=True)
# plt.xlabel('Actual PM2.5')
# plt.ylabel('Predicted PM2.5')
# plt.title('Density Heatmap from Random Forest')
# plt.xlim(-5, 50)
# plt.ylim(-5, 50)
# plt.show()

# # 2D Histogram
# plt.figure(figsize=(10, 6))
# plt.hist2d(y_test, y_pred, bins=(30, 30), cmap='Blues')
# plt.colorbar(label='Frequency')
# plt.xlabel('Actual PM2.5')
# plt.ylabel('Predicted PM2.5')
# plt.title('Actual vs Predicted PM2.5 2D Histogram from Random Forest')
# plt.show()

# # Save the best model
# joblib.dump(best_rf, f'random_forest_model_{csv_file.split(".")[0]}_updated.pkl')