import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px

# fetching stock data
stock_name = yf.Ticker('META')
today = dt.now().strftime('%Y-%m-%d')
df = stock_name.history(start='2014-01-01', end=today)

# preparing data for prophet
df.reset_index(inplace=True)
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
df[['ds','y']] = df[['Date','Close']]  
print(df)

# data visualization
fig = px.line(df, x='ds', y='y')
fig.update_xaxes(rangeslider_visible=True)
fig.show()

# create and fit prophet model
model = Prophet()
model.fit(df)

