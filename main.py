import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

stock_name = yf.Ticker('META')
# data = stock_name.history(period="max")
today = datetime.now().strftime('%Y-%m-%d')
data = stock_name.history(start='2014-01-01', end=today)

# Price Info
print(data)
# General Info
print("Stock Info") # Non-trading days are not included
print(stock_name.info)
# Dividends and Stock Splits
print("Dividends and Stock Splits")
print(stock_name.actions)
# Dividends
print("Dividends")
print(stock_name.dividends)
# Splits
print("Splits")
print(stock_name.splits)
# Quaterly Financials
print("Quarterly Financials")
print(stock_name.quarterly_financials)
# Major Holders
# print("Major Holders")
# print(stock_name.major_holders)
# Institutional Holders
# print("Institutional Holders")
# print(stock_name.institutional_holders)
# Recommendations
# print("Recommendations")
# print(stock_name.recommendations)
# print(stock_name.recommendations['Grade'].value_counts())

# Data Cleaning
# If is an index not an individual stock
# del stock_name["Dividends"]
# del stock_name["Stock Splits"]

# Drop rows with missing values, no need to fill
# data.dropna(inplace=True)

# Outliers
# z_scores = np.abs(stats.zscore(data))
# data = data[(z_scores < 3).all(axis=1)]


# Data Visualization
# Subplots of Price Info
data.plot(kind="line", figsize=(12,12), subplots=True)
plt.show()

# Open and Close Prices
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Close', color='blue')
plt.plot(data.index, data['Open'], label='Open', color='orange')
plt.title('Open and Close Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot with candles along with volume and moving average
days = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
month = stock_name.history(start=days, end=today)
mpf.plot(month, type='candle', volume=True, mav=(3,6,9))

# Heatmap correlation
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            linewidths=0.5, linecolor='black', cbar_kws={'shrink': 0.75, 'label': 'Correlation Coefficient'})
plt.title('Correlation Heatmap of Stock Data', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Splitting data into training and test set

high_prices = data.loc[:, 'High'].values.reshape(-1, 1)
low_prices = data.loc[:, 'Low'].values.reshape(-1, 1)
mid_prices = (high_prices + low_prices) / 2.0

# length of mid_prices is 2664

train_data = mid_prices[:2000]
test_data = mid_prices[2000:]

#normalize data

scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)