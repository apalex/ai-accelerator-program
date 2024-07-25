import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler

stock_name = yf.Ticker('META')
data = stock_name.history(period="max")

# Drop rows with missing values
# data.dropna(inplace=True)

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

# # stock_name.plot.line(y="Close", use_index=True)
# # plt.show()

# Data Cleaning
# If is an index not an individual stock
# del stock_name["Dividends"]
# del stock_name["Stock Splits"]


