import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# fetching stock data
ticker = yf.Ticker('META')
today = dt.now().strftime('%Y-%m-%d')
df = ticker.history(start='2014-01-01', end=today)

# preparing data for prophet
df.reset_index(inplace=True)
df['ds'] = df['Date'].dt.strftime('%Y-%m-%d')
df[['y']] = df[['Close']]  

# real data visualization
fig = px.line(df, x='ds', y='y')
fig.update_xaxes(rangeslider_visible=True)
fig.show()

# create and fit prophet model
model = Prophet()
model.fit(df)

# train and test data tests
train_data = df.sample(frac=0.8, random_state=0)
test_data = df.drop(train_data.index)
print(f'training data size : {train_data.shape}')
print(f'testing data size : {test_data.shape}')

# making future predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# plotting predictions
model.plot(forecast)
plt.title(f"Predicted Stock Price of {ticker} Using Prophet")
plt.xlabel("Date")
plt.ylabel("Close")
plt.show()

# MAE
y_actual = test_data['y']
y_predicted = forecast[forecast['ds'].isin(test_data['ds'])]['yhat']
mae = mean_absolute_error(y_actual, y_predicted)
print(f'Mean Absolute Error: {mae}')

#MSE
mse = mean_squared_error(y_actual, y_predicted)
print(f'Mean squared error: {mse}')