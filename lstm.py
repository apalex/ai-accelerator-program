import streamlit as st
import lstm_model as lstm
import yfinance as yf
import pandas as pd
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta as rd

st.title("Finance Bro")
st.markdown("An AI financial assistant to help you envision the market!")

st.divider()

st.subheader("Select a stock to get started!")
st.write("")

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
time_frame = None

if selected_time_frame == "5d":
    time_frame = today - rd(days=5)

elif selected_time_frame == "1mo":
    time_frame = today - rd(months=1)

elif selected_time_frame == "6mo":
    time_frame = today - rd(months=6)

elif selected_time_frame == "1y":
    time_frame = today - rd(years=1)

elif selected_time_frame == "5y":
    time_frame = today - rd(years=5)

st.write("")
st.write("")
st.write("")

graph_df = ticker.history(start=time_frame, end=today)
st.line_chart(
    data=graph_df['Close'],
    y_label="Close Price"
)

st.divider()

st.subheader("Now let's do some prediction!")
num_days = st.slider("Number of days to be predicted:", min_value=1, max_value=10)

st.write("")

if st.button("Predict", type="primary"):
    st.divider()
    date = today - rd(years=10)
    time_frame = date.strftime('%Y-%m-%d')
    predictor = lstm.LSTMStockPredictor(ticker=selected_ticker, start_date=time_frame, n_steps=50)
    predictor.prepare_data()
    predictor.build_model()
    predictor.train_model()
    train_pred, test_pred, val_pred = predictor.predict()
    err_metrics = predictor.calculate_error_metrics(train_predictions=train_pred, test_predictions=test_pred, val_predictions=val_pred)
    print(err_metrics)
    n_days_preds = predictor.predict_next_days(num_days)
