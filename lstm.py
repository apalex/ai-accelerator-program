import streamlit as st
import lstm_model as lstm
import yfinance as yf
import pandas as pd
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta as rd

# C:\Users\ducky\Desktop\financebro\ai-accelerator-program\LSTM.py

st.title("Finance Bro")
st.markdown("An AI financial assistant to help you envision the market!")

st.divider()

st.subheader("Select some parameters to get started!")

# Columns (side by side) for parameters of graph pre model prediction
col1, col2, = st.columns(2)

with col1:
    def load_tickers():
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tickers = pd.read_html(url)
        return tickers[0]['Symbol'].tolist()

    tickers = load_tickers()

    selected_ticker = st.selectbox(
        "Stock Ticker:",
        tickers,
    )

with col2:
    selected_time_frame = st.selectbox (
        "Time Frame:",
        ["5d", "1mo", "6mo", "1y", "5y"],
    )

# Fetching data set from inputs into graph
ticker = yf.Ticker(selected_ticker)
today = dt.now()
start_date = None

if selected_time_frame == "5d":
    start_date = today - rd(days=5)

elif selected_time_frame == "1mo":
    start_date = today - rd(months=1)

elif selected_time_frame == "6mo":
    start_date = today - rd(months=6)

elif selected_time_frame == "1y":
    start_date = today - rd(years=1)

elif selected_time_frame == "5y":
    start_date = today - rd(years=5)

graph_df = ticker.history(start=start_date, end=today)

st.line_chart(
    data=graph_df['Close'],
    x_label="Date",
    y_label="Close Price"
)

st.divider()

