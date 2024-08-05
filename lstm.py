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

# LSTM Model

stock_name = yf.Ticker('META')
today = datetime.now().strftime('%Y-%m-%d')
data = stock_name.history(start='2014-01-01', end=today)

# Preprocessing
closing_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
closing_prices_scaled = scaler.fit_transform(closing_prices)

# Model
def prepare_data(data, n_steps):
    x, y = [], []
    for i in range(len(data) - n_steps):
        x.append(data[i:(i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(x), np.array(y)

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


n_steps = 50
x_train, y_train = prepare_data(closing_prices_scaled, n_steps)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = LSTM_Model((x_train.shape[1], 1))
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1)


train_predictions = model.predict(x_train)
train_predictions = scaler.inverse_transform(train_predictions)
mse = mean_squared_error(closing_prices[n_steps:], train_predictions)
print(f'Mean Squared Error on Training Data: {mse}')

plt.figure(figsize=(12, 6))
plt.plot(data.index[n_steps:], closing_prices[n_steps:], label='Actual Prices', color='blue')
plt.plot(data.index[n_steps:], train_predictions, label='Predicted Prices', color='red')
plt.title(f'{stock_name.info["symbol"]} Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()
