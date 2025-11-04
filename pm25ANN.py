import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from tensorflow.keras import backend as K

# Assuming csv_file is provided for naming purposes only
csv_file = "FortLeeWinterMinus1"

# Load the datasets
X_train_scaled_m1 = joblib.load('X_train_scaled_m1.pkl')
y_train_m1 = joblib.load('y_train_m1.pkl')
X_test_scaled_m1 = joblib.load('X_test_scaled_m1.pkl')
y_test_m1 = joblib.load('y_test_m1.pkl')


# Assuming these are the features in the order they were scaled
features = ['pm25-1', 'temp', 'visibility', 'winddir', 'windspeed', 'precip', 'humidity', 'solarradiation', 'cloudcover']

# Function to create model
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # Compile the model with a specified optimizer
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
    return model

# Create and compile the model
model = create_model(input_dim=len(features))

# Train the model
model.fit(X_train_scaled_m1, y_train_m1, epochs=100, batch_size=20, verbose=1)

# Prediction and Evaluation
y_pred = model.predict(X_test_scaled_m1).squeeze()  # Use .squeeze() to convert y_pred to 1D
mae = mean_absolute_error(y_test_m1, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_m1, y_pred))
r2 = r2_score(y_test_m1, y_pred)
accuracy_like_metric = np.mean(np.abs(y_test_m1 - y_pred) <= 5)
print(f'Accuracy-like metric (with tolerance ±5): {accuracy_like_metric:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared Score: {r2:.2f}')

# No permutation importance calculation is included due to its complexity and dependency on the model's predict method

# Density Heatmap
plt.figure(figsize=(10, 6))
ax = sns.kdeplot(x=y_test_m1, y=y_pred, cmap="Blues", fill=True, cbar=True)
ax.set_xlabel('Actual PM2.5')
ax.set_ylabel('Predicted PM2.5')
ax.set_title('Density Heatmap from ANN')
plt.xlim(-5, 50)
plt.ylim(-5, 50)
plt.show()

# 2D Histogram
plt.figure(figsize=(10, 6))
plt.hist2d(y_test_m1, y_pred, bins=(30, 30), cmap='Blues')
plt.colorbar(label='Frequency')
plt.xlabel('Actual PM2.5')
plt.ylabel('Predicted PM2.5')
plt.title('Actual vs Predicted PM2.5 2D Histogram from ANN')
plt.show()

# Save the model
model.save(f'ann_model_{csv_file}.keras')

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam, RMSprop
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import joblib
# from tensorflow.keras import backend as K

# # Assuming csv_file is provided for naming purposes only
# csv_file = "FortLeeWinterMinus1"

# # Load the datasets
# X_train_scaled = joblib.load('X_train_scaled.pkl')
# y_train = joblib.load('y_train.pkl')
# X_test_scaled = joblib.load('X_test_scaled.pkl')
# y_test = joblib.load('y_test.pkl')

# # Assuming these are the features in the order they were scaled
# features = ['pm25-1', 'pm25-2', 'temp', 'visibility', 'winddir', 'windspeed', 'precip', 'humidity', 'solarradiation', 'cloudcover']

# # Function to create model
# def create_model(input_dim):
#     model = Sequential()
#     model.add(Dense(12, input_dim=input_dim, activation='relu'))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dense(1, activation='linear'))
#     # Compile the model with a specified optimizer
#     model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))
#     return model

# # Create and compile the model
# model = create_model(input_dim=len(features))

# # Train the model
# model.fit(X_train_scaled, y_train, epochs=100, batch_size=20, verbose=1)

# # Prediction and Evaluation
# y_pred = model.predict(X_test_scaled).squeeze()  # Use .squeeze() to convert y_pred to 1D
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
# accuracy_like_metric = np.mean(np.abs(y_test - y_pred) <= 5)
# print(f'Accuracy-like metric (with tolerance ±5): {accuracy_like_metric:.2f}')
# print(f'Mean Absolute Error (MAE): {mae:.2f}')
# print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
# print(f'R-squared Score: {r2:.2f}')

# # No permutation importance calculation is included due to its complexity and dependency on the model's predict method

# # Density Heatmap
# plt.figure(figsize=(10, 6))
# ax = sns.kdeplot(x=y_test, y=y_pred, cmap="Blues", fill=True, cbar=True)
# ax.set_xlabel('Actual PM2.5')
# ax.set_ylabel('Predicted PM2.5')
# ax.set_title('Density Heatmap from ANN')
# plt.xlim(-5, 50)
# plt.ylim(-5, 50)
# plt.show()

# # 2D Histogram
# plt.figure(figsize=(10, 6))
# plt.hist2d(y_test, y_pred, bins=(30, 30), cmap='Blues')
# plt.colorbar(label='Frequency')
# plt.xlabel('Actual PM2.5')
# plt.ylabel('Predicted PM2.5')
# plt.title('Actual vs Predicted PM2.5 2D Histogram from ANN')
# plt.show()

# # Save the model
# model.save(f'ann_model_{csv_file}.keras')
