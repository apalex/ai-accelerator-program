import streamlit as st
import lstm_model as lstm
import yfinance as yf
import pandas as pd
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta as rd
import numpy as np
import matplotlib.pyplot as plt

st.title("Finance Bro")
st.markdown("An AI financial assistant to help you envision the market!")

st.divider()

st.subheader("Select a stock to get started!")
st.write("")

# Columns (side by side) for parameters of graph pre-model prediction
col1, col2 = st.columns(2)

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
    selected_time_frame = st.selectbox(
        "Time Frame:",
        ["5d", "1mo", "6mo", "1y", "5y"],
    )

# Fetching dataset from inputs into graph
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
    x_label="Date",
    y_label="Close Price (USD)"
)

st.divider()

st.subheader("Now let's do some prediction!")
num_days = st.slider("Number of days to be predicted:", min_value=1, max_value=10)

st.write("")

if st.button("Predict", type="primary"):
    st.divider()

    # Progress bar during model training
    progress_bar = st.progress(0)
    
    date = today - rd(years=10)
    time_frame = date.strftime('%Y-%m-%d')
    predictor = lstm.LSTMStockPredictor(ticker=selected_ticker, start_date=time_frame, n_steps=50)
    
    predictor.prepare_data()
    predictor.build_model()

    # Updating progress during training
    for i in range(1, 101):
        predictor.train_model(epochs=1, batch_size=32)  # Incremental training
        progress_bar.progress(i)
    
    train_pred, test_pred, val_pred = predictor.predict()

    # Combine predictions
    combined_predictions = pd.concat([
        pd.Series(train_pred.flatten(), index=predictor.data.index[predictor.n_steps:len(train_pred) + predictor.n_steps]),
        pd.Series(val_pred.flatten(), index=predictor.data.index[len(train_pred) + predictor.n_steps:len(train_pred) + len(val_pred) + predictor.n_steps]),
        pd.Series(test_pred.flatten(), index=predictor.data.index[len(train_pred) + len(val_pred) + predictor.n_steps:])
    ])
    
    err_metrics = predictor.calculate_error_metrics(train_predictions=train_pred, test_predictions=test_pred, val_predictions=val_pred)
    
    # Display error metrics
    st.write("### Error Metrics")
    st.write(err_metrics)

    n_days_preds = predictor.predict_next_days(num_days)
    
    # Final graph with actual data and predictions
    st.write("### Prediction vs Actual Data")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predictor.data.index, predictor.data['Close'], label='Actual Close Price', color='blue')
    ax.plot(combined_predictions.index, combined_predictions.values, label='Predicted Close Price', color='red')
    
    # Plot future predictions
    future_dates = pd.date_range(start=predictor.data.index[-1], periods=num_days + 1, freq='D')[1:]
    ax.plot(future_dates, n_days_preds, label='Future Predictions', color='orange', linestyle='--')
    
    ax.set_title(f'{selected_ticker} Stock Price Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price (USD)')
    ax.legend()
    st.pyplot(fig)