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

# 40s MSE

# Fetch stock data
stock_name = yf.Ticker('META')
today = datetime.now().strftime('%Y-%m-%d')
data = stock_name.history(start='2014-01-01', end=today)

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
x_train, y_train = prepare_data(scaled_features, n_steps)

# Model
def LSTM_Model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.15))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.15))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model

model = LSTM_Model((x_train.shape[1], x_train.shape[2]))
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Predictions
train_predictions = model.predict(x_train)
scaled_predictions = np.zeros((train_predictions.shape[0], scaled_features.shape[1]))
scaled_predictions[:, 3] = train_predictions[:, 0]
train_predictions = scaler.inverse_transform(scaled_predictions)[:, 3]

mse = mean_squared_error(data['Close'][n_steps:], train_predictions)
print(f'Mean Squared Error on Training Data: {mse}')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(data.index[n_steps:], data['Close'][n_steps:], label='Actual Prices', color='blue')
plt.plot(data.index[n_steps:], train_predictions, label='Predicted Prices', color='red')
plt.plot(data.index, data['MA50'], label='MA50', color='green')
plt.plot(data.index, data['MA100'], label='MA100', color='orange')
plt.plot(data.index, data['MA200'], label='MA200', color='purple')
plt.title(f'{stock_name.info["symbol"]} Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()
