import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from dateutil.relativedelta import relativedelta as rd

class LSTMStockPredictor:
    
    # Constructor for Lstm model
    def __init__(self, ticker, start_date, n_steps):
        self.ticker = ticker
        self.n_steps = n_steps
        self.data = self.fetch_data(start_date)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
    
    # method to fetch data from yfinance based on ticker and start date
    def fetch_data(self, start_date):
        stock_name = yf.Ticker(self.ticker)
        today = datetime.now().strftime('%Y-%m-%d')
        data = stock_name.history(start=start_date, end=today)

        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA100'] = data['Close'].rolling(window=100).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        data['RSI'] = self.calculate_RSI(data, window=14)
        data = data.dropna()
        return data

    #method to calculate RSI 
    @staticmethod
    def calculate_RSI(data, window):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        RS = gain / loss
        RSI = 100 - (100 / (1 + RS))
        return RSI

    # method to prepare the data
    def prepare_data(self):
        features = self.data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA100', 'MA200', 'RSI']]
        scaled_features = self.scaler.fit_transform(features)
        
        x, y = [], []
        for i in range(len(scaled_features) - self.n_steps):
            x.append(scaled_features[i:(i + self.n_steps), :])
            y.append(scaled_features[i + self.n_steps, 3])  # 'Close' price as target
        
        x, y = np.array(x), np.array(y)
        
        train_size = int(len(x) * 0.7)
        val_size = int(len(x) * 0.2)
        
        self.x_train, self.y_train = x[:train_size], y[:train_size]
        self.x_val, self.y_val = x[train_size:train_size + val_size], y[train_size:train_size + val_size]
        self.x_test, self.y_test = x[train_size + val_size:], y[train_size + val_size:]
    
    # method to build the model
    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, activation='relu', input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
        self.model.add(LSTM(units=50, activation='relu', return_sequences=False))
        self.model.add(Dense(units=1))
        
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    # method to train the model with default parameters
    def train_model(self, epochs=1000, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.x_val, self.y_val), callbacks=[early_stopping])
    
    # method to predict based on the self.dataset of the self.lstm_model object
    def predict(self):
        train_predictions = self.model.predict(self.x_train)
        val_predictions = self.model.predict(self.x_val)
        test_predictions = self.model.predict(self.x_test)
        
        train_predictions = self.rescale_predictions(train_predictions)
        val_predictions = self.rescale_predictions(val_predictions)
        test_predictions = self.rescale_predictions(test_predictions)
        
        return train_predictions, test_predictions, val_predictions
    
    # method to rescale predictions accordingly
    def rescale_predictions(self, predictions):
        # Create an array with the same number of columns as the original scaled features (9 columns, as expected by the scaler)
        scaled_predictions = np.zeros((predictions.shape[0], 9))
        
        # Place the predictions in the 'Close' price column (assuming it's the 4th column, index 3)
        scaled_predictions[:, 3] = predictions[:, 0]
        
        # Now perform the inverse transformation on the entire array
        rescaled_predictions = self.scaler.inverse_transform(scaled_predictions)
        
        # Return only the 'Close' price column
        return rescaled_predictions[:, 3]

    # method to return error metrics: MAE, MSE, R^2 score
    def calculate_error_metrics(self, train_predictions, val_predictions, test_predictions):
        train_size = len(self.x_train)
        val_size = len(self.x_val)
        
        train_mse = mean_squared_error(self.data['Close'][self.n_steps:self.n_steps + train_size], train_predictions, squared=False)
        val_mse = mean_squared_error(self.data['Close'][self.n_steps + train_size:self.n_steps + train_size + val_size], val_predictions, squared=False)
        test_mse = mean_squared_error(self.data['Close'][self.n_steps + train_size + val_size:], test_predictions, squared=False)
        
        train_r2 = r2_score(self.data['Close'][self.n_steps:self.n_steps + train_size], train_predictions)
        val_r2 = r2_score(self.data['Close'][self.n_steps + train_size:self.n_steps + train_size + val_size], val_predictions)
        test_r2 = r2_score(self.data['Close'][self.n_steps + train_size + val_size:], test_predictions)
        
        train_mae = mean_absolute_error(self.data['Close'][self.n_steps:self.n_steps + train_size], train_predictions)
        val_mae = mean_absolute_error(self.data['Close'][self.n_steps + train_size:self.n_steps + train_size + val_size], val_predictions)
        test_mae = mean_absolute_error(self.data['Close'][self.n_steps + train_size + val_size:], test_predictions)
        
        return {
            'Training set MSE': round(train_mse, 3),
            'Validation set MSE': round(val_mse, 3),
            'Test set MSE': round(test_mse, 3),
            'Training set R^2 score': round(train_r2, 3),
            'Validation set R^2 score': round(val_r2, 3),
            'Test set R^2 score': round(test_r2, 3),
            'Training set MAE': round(train_mae, 3),
            'Validation set MAE': round(val_mae, 3),
            'Test set MAE': round(test_mae, 3)
        }
    
    # method to predict n amount of future days
    def predict_next_days(self, n_days):
        next_input = self.x_test[-1].reshape(1, self.n_steps, self.x_test.shape[2])
        next_days_predictions = []
        
        for _ in range(n_days):
            next_pred = self.model.predict(next_input)
            next_days_predictions.append(next_pred[0, 0])
            
            next_input = np.roll(next_input, -1, axis=1)
            next_input[0, -1, 3] = next_pred
        
        next_days_predictions = np.array(next_days_predictions).reshape(-1, 1)
        next_days_predictions = self.rescale_predictions(next_days_predictions)
        
        return next_days_predictions
    
    """
    def plot_results(self, train_predictions, val_predictions, test_predictions, next_days_predictions=None):
        future_dates = pd.date_range(start=self.data.index[-1], periods=len(next_days_predictions) + 1, freq='D')[1:]
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index[self.n_steps:self.n_steps + len(self.x_train)], self.data['Close'][self.n_steps:self.n_steps + len(self.x_train)], label='Train Actual Prices', color='blue')
        plt.plot(self.data.index[self.n_steps + len(self.x_train):self.n_steps + len(self.x_train) + len(self.x_val)], self.data['Close'][self.n_steps + len(self.x_train):self.n_steps + len(self.x_train) + len(self.x_val)], label='Validation Actual Prices', color='orange')
        plt.plot(self.data.index[self.n_steps + len(self.x_train) + len(self.x_val):], self.data['Close'][self.n_steps + len(self.x_train) + len(self.x_val):], label='Test Actual Prices', color='green')

        plt.plot(self.data.index[self.n_steps:self.n_steps + len(self.x_train)], train_predictions, label='Train Predicted Prices', color='red')
        plt.plot(self.data.index[self.n_steps + len(self.x_train):self.n_steps + len(self.x_train) + len(self.x_val)], val_predictions, label='Validation Predicted Prices', color='purple')
        plt.plot(self.data.index[self.n_steps + len(self.x_train) + len(self.x_val):], test_predictions, label='Test Predicted Prices', color='brown')

        if next_days_predictions is not None:
            plt.plot(future_dates, next_days_predictions, label='Next Days Predictions', color='orange', linestyle='--')
        
        plt.title(f'{self.ticker} Stock Price Prediction using LSTM')
        plt.xlabel('Date')
        plt.ylabel('Stock Price (USD)')
        plt.legend()
        plt.show()
    """

# Sample Usage
def main():
    today = datetime.now()
    time_frame = today - rd(years=10)
    predictor = LSTMStockPredictor(ticker='AAPL', start_date=time_frame)
    predictor.prepare_data()
    predictor.build_model()
    predictor.train_model()
    train_preds, val_preds, test_preds = predictor.predict()
    metrics = predictor.calculate_error_metrics(train_preds, val_preds, test_preds)
    next_days_preds = predictor.predict_next_days(5)
    predictor.plot_results(train_preds, val_preds, test_preds, next_days_preds)