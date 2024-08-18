import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

class TSFStockPredictor:

    def __init__(self, ticker, start_date):
        self.ticker = ticker
        self.df = self.fetch_data(start_date)
        self.model = None

    def fetch_data(self, start_date):
        ticker = yf.Ticker(self.ticker)
        today = dt.now().strftime('%Y-%m-%d')
        df = ticker.history(start_date, end=today)

        return df

    def prepare_data(self):
        self.df.reset_index(inplace=True)
        self.df['ds'] = self.df['Date'].dt.strftime('%Y-%m-%d')
        self.df[['y']] = self.df[['Close']]  

    """
    # real data visualization
    fig = px.line(df, x='ds', y='y')
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()
    """
    
    def build_model(self):
        model = Prophet()
        model.fit(self.df)
        self.model = model

    def train_model(self):
        train_data = self.df.sample(frac=0.8, random_state=0)
        test_data = self.df.drop(train_data.index)

    """
    maybe
    def build_and_train_model(self):
        self.train_data = self.df.sample(frac=0.8, random_state=0)
        self.test_data = self.df.drop(self.train_data.index)
        
        self.model = Prophet()
        self.model.fit(self.train_data)
    """

    # making future predictions
    def predict(self):
        future = self.model.make_future_dataframe(periods=30)
        forecast = self.model.predict(future)
        return forecast

    """
    # plotting predictions
    model.plot(forecast)
    plt.title(f"Predicted Stock Price of {ticker} Using Prophet")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.show()
    """

    def calc_error_metrics(self, forecast):
        y_actual = self.test_data['y']
        y_predicted = forecast[forecast['ds'].isin(self.test_data['ds'])]['yhat']
        
        mae = mean_absolute_error(y_actual, y_predicted)
        mse = mean_squared_error(y_actual, y_predicted)
        
        print(f'Mean Absolute Error: {mae}')
        print(f'Mean Squared Error: {mse}')