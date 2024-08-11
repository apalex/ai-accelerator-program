import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Fetch stock data
stock_name = yf.Ticker('AAPL')
today = datetime.now().strftime('%Y-%m-%d')
data = stock_name.history(start='2016-01-01', end=today)

# Adding Moving Averages
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA100'] = data['Close'].rolling(window=100).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# Calculate RSI
def calculate_RSI(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

data['RSI'] = calculate_RSI(data, window=14)
data = data.dropna()  # Drop rows with NaN values due to moving average and RSI calculation

# Preprocessing
features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA100', 'MA200', 'RSI']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Prepare data
def prepare_data(data, n_steps):
    x, y = [], []
    for i in range(len(data) - n_steps):
        x.append(data[i:(i + n_steps), :])
        y.append(data[i + n_steps, 3])  # Using 'Close' price as the target
    return np.array(x), np.array(y)

n_steps = 50
x, y = prepare_data(scaled_features, n_steps)

# Split the data into train, validation, and test sets
train_size = int(len(x) * 0.7)
val_size = int(len(x) * 0.2)
test_size = len(x) - train_size - val_size

x_train, y_train = x[:train_size], y[:train_size]
x_val, y_val = x[train_size:train_size + val_size], y[train_size:train_size + val_size]
x_test, y_test = x[train_size + val_size:], y[train_size + val_size:]

# Model
def LSTM_Model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(units=50, activation='relu', return_sequences=False))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    return model

model = LSTM_Model((x_train.shape[1], x_train.shape[2]))
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping])

# Predictions
train_predictions = model.predict(x_train)
val_predictions = model.predict(x_val)
test_predictions = model.predict(x_test)

# Rescale predictions back to original scale
def rescale_predictions(predictions, original_data, scaler, feature_index):
    scaled_predictions = np.zeros((predictions.shape[0], original_data.shape[1]))
    scaled_predictions[:, feature_index] = predictions[:, 0]
    return scaler.inverse_transform(scaled_predictions)[:, feature_index]

train_predictions = rescale_predictions(train_predictions, scaled_features, scaler, 3)
val_predictions = rescale_predictions(val_predictions, scaled_features, scaler, 3)
test_predictions = rescale_predictions(test_predictions, scaled_features, scaler, 3)

# Calculate MSE for each set
train_mse = mean_squared_error(data['Close'][n_steps:n_steps + train_size], train_predictions, squared=False)
val_mse = mean_squared_error(data['Close'][n_steps + train_size:n_steps + train_size + val_size], val_predictions, squared=False)
test_mse = mean_squared_error(data['Close'][n_steps + train_size + val_size:], test_predictions, squared=False)

print(f'Mean Squared Error on Training Data: {train_mse}')
print(f'Mean Squared Error on Validation Data: {val_mse}')
print(f'Mean Squared Error on Test Data: {test_mse}')

# Calculate R-squared for each set
train_r2 = r2_score(data['Close'][n_steps:n_steps + train_size], train_predictions)
val_r2 = r2_score(data['Close'][n_steps + train_size:n_steps + train_size + val_size], val_predictions)
test_r2 = r2_score(data['Close'][n_steps + train_size + val_size:], test_predictions)

print(f'R-squared on Training Data: {train_r2}')
print(f'R-squared on Validation Data: {val_r2}')
print(f'R-squared on Test Data: {test_r2}')

# Calculate MAE for each set
train_mae = mean_absolute_error(data['Close'][n_steps:n_steps + train_size], train_predictions)
val_mae = mean_absolute_error(data['Close'][n_steps + train_size:n_steps + train_size + val_size], val_predictions)
test_mae = mean_absolute_error(data['Close'][n_steps + train_size + val_size:], test_predictions)

print(f'Mean Absolute Error on Training Data: {train_mae}')
print(f'Mean Absolute Error on Validation Data: {val_mae}')
print(f'Mean Absolute Error on Test Data: {test_mae}')

# Function to predict the next n days
def predict_next_days(model, last_data, n_days, scaler, feature_index):
    predictions = []
    current_input = last_data[-n_steps:].reshape(1, n_steps, last_data.shape[1])
    
    for _ in range(n_days):
        next_pred = model.predict(current_input)
        predictions.append(next_pred[0, 0])
        
        next_input = np.zeros((1, n_steps, last_data.shape[1]))
        next_input[0, :-1, :] = current_input[0, 1:, :]
        next_input[0, -1, feature_index] = next_pred
        
        current_input = next_input
        
    scaled_predictions = np.zeros((n_days, last_data.shape[1]))
    scaled_predictions[:, feature_index] = predictions
    return scaler.inverse_transform(scaled_predictions)[:, feature_index]

# Prediction for the next 5 days
next_input = x_test[-1].reshape(1, n_steps, x_test.shape[2])
next_5_days_predictions = []

for _ in range(5):
    next_pred = model.predict(next_input)
    next_5_days_predictions.append(next_pred[0, 0])
    
    # Update next_input with the new prediction
    next_input = np.roll(next_input, -1, axis=1)
    next_input[0, -1, 3] = next_pred

next_5_days_predictions = np.array(next_5_days_predictions)
next_5_days_predictions = rescale_predictions(next_5_days_predictions.reshape(-1, 1), scaled_features, scaler, 3)

print(f'Predictions for the next 5 days: {next_5_days_predictions}')

print(model.summary())

# Extend the index to include the next 5 days
future_dates = pd.date_range(start=data.index[-1], periods=6, freq='D')[1:]

plt.figure(figsize=(12, 6))
plt.plot(data.index[n_steps:n_steps + train_size], data['Close'][n_steps:n_steps + train_size], label='Train Actual Prices', color='blue')
plt.plot(data.index[n_steps + train_size:n_steps + train_size + val_size], data['Close'][n_steps + train_size:n_steps + train_size + val_size], label='Validation Actual Prices', color='orange')
plt.plot(data.index[n_steps + train_size + val_size:], data['Close'][n_steps + train_size + val_size:], label='Test Actual Prices', color='green')

plt.plot(data.index[n_steps:n_steps + train_size], train_predictions, label='Train Predicted Prices', color='red')
plt.plot(data.index[n_steps + train_size:n_steps + train_size + val_size], val_predictions, label='Validation Predicted Prices', color='purple')
plt.plot(data.index[n_steps + train_size + val_size:], test_predictions, label='Test Predicted Prices', color='brown')

# Plot the next 5 days predictions
plt.plot(future_dates, next_5_days_predictions, label='Next 5 Days Predictions', color='orange', linestyle='--')

plt.title(f'{stock_name.info["symbol"]} Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()
