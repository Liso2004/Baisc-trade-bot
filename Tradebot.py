import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import math

# Load dataset
data = pd.read_csv("all_stock_5yr.csv")

# Extract available stock names
all_stock_names = data['name'].unique()
print(all_stock_names)

# Ask user for stock name
stock_name = input("Enter a Stock Price Name: ")

# Filter rows for selected stock
final_data = data[data['name'] == stock_name]

# Preview data
print(final_data.head())

# Plot full history
final_data.plot(x='date', y='close', color="red", title="Stock Close Price")
plt.show()

# Plot first 60 days
New_data = final_data.head(60)
New_data.plot(x='date', y='close', color="green", title="First 60 Days Close Price")
plt.show()

# Use only close column
close_data = final_data.filter(['close'])
dataset = close_data.values

# Scaling
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# 70% training size
training_data_len = math.ceil(len(dataset) * 0.7)

train_data = scaled_data[0:training_data_len]

# Creating training X and Y
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert to numpy
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile
model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(x_train, y_train, batch_size=1, epochs=1)

# ----- TEST DATA -----
test_data = scaled_data[training_data_len - 60:]
x_test = []
y_test = dataset[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print("RMSE:", rmse)

# ----- PLOT RESULTS -----
train = final_data[:training_data_len]
valid = final_data[training_data_len:]

valid.loc[:, 'Predictions'] = predictions

plt.figure(figsize=(14,7))
plt.title("Model Prediction")
plt.xlabel("Date")
plt.ylabel("Close Price")

plt.plot(train['close'], label='Train')
plt.plot(valid['close'], label='Validation')
plt.plot(valid['Predictions'], label='Predictions')

plt.legend()
plt.show()
