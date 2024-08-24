import streamlit as st
import lstm_model as lstm
import yfinance as yf
import pandas as pd
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta as rd

# Basic information and formatting
st.title("Finance Bro")
st.markdown("An AI financial assistant to help you envision the market!")

st.divider()

st.subheader("Select a stock to get started!")
st.write("")

# Columns (side by side) for parameters of graph pre-model prediction
col1, col2 = st.columns(2)

# col1 contains a select box containing tickers of the S&P500
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

# col2 allows user to select time frame to visualize the real data
with col2:
    selected_time_frame = st.selectbox(
        "Time Frame:",
        ["5d", "1mo", "6mo", "1y", "5y"],
    )

# Fetching dataset from inputs into graph
ticker = yf.Ticker(selected_ticker)
today = dt.now()
time_frame = None

# based on the option selected for timeframe of graph, it will update the graph's domain
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

# Spacers 
st.write("")
st.write("")
st.write("")

# line graph for real stock data
graph_df = ticker.history(start=time_frame, end=today)
st.line_chart(
    data=graph_df['Close'],
    x_label="Date",
    y_label="Close Price (USD)"
)

# More text and formatting
st.divider()

st.subheader("Now let's do some prediction!")
num_days = st.slider("Number of days to be predicted:", min_value=1, max_value=10)

st.write("")

# When predict button is clicked, model is built, trained, and is shown on a graph
if st.button("Predict", type="primary"):
    st.divider()
    
    # time frame for df of model
    date = today - rd(years=10)
    time_frame = date.strftime('%Y-%m-%d')
    
    # Constructing lstm model based on the lstm_model class
    lstm_model = lstm.LSTMStockPredictor(ticker=selected_ticker, start_date=time_frame, n_steps=50)
    
    # preparing data and building model methods from lstm_model class
    lstm_model.prepare_data()
    lstm_model.build_model()

    # Spinner with model is training
    with st.spinner("Finance Bro is working hard..."):
        lstm_model.train_model()
    
    st.subheader("Finance Bro's Predictions!:")
        
    # 3 sets of values are returned for lstm_model's predict method based on its predictions on
    # the train, test, and val dataframes
    train_pred, test_pred, val_pred = lstm_model.predict()
    
    # Model's prediction for n days
    n_days_preds = lstm_model.predict_next_days(num_days)

    # Concatenate predictions for train, test, val for graphing purposes
    combined_predictions = pd.concat([
        pd.Series(train_pred.flatten(), index=lstm_model.data.index[lstm_model.n_steps:len(train_pred) + lstm_model.n_steps]),
        pd.Series(val_pred.flatten(), index=lstm_model.data.index[len(train_pred) + lstm_model.n_steps:len(train_pred) + len(val_pred) + lstm_model.n_steps]),
        pd.Series(test_pred.flatten(), index=lstm_model.data.index[len(train_pred) + len(val_pred) + lstm_model.n_steps:])
    ])
    
    # Combined dates (including future n_days) for graphing purposes
    graph_df = ticker.history(start=time_frame, end=today)
    original_dates = graph_df.index
    last_date = original_dates[-1]  # Get the last date from original data
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days, freq='D')
    
    combined_dates = original_dates.union(future_dates)
    
    # Create a Series for n_days_preds with future dates as the index
    future_predictions_series = pd.Series(n_days_preds.flatten(), index=future_dates)

    # Concatenate future predictions with the combined predictions
    combined_predictions = pd.concat([combined_predictions, future_predictions_series])
    
    # Extend real data with NaNs to match the length of the predictions
    extended_real_data = graph_df['Close'].reindex(combined_dates)
    
    # New dataframe with combined dates and predictions to be graphed
    preds_df = pd.DataFrame({
        "Real Data": extended_real_data,
        "Predictions": combined_predictions
    })

    # Graph of all predictions alongside real data for comparison
    st.line_chart(preds_df)
    
    # columns to show predicted results and error metrics
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("### Predicted Prices")
        st.write(f"for the next {num_days} days")
        predicted_values_df = pd.DataFrame({
            "Date": future_dates.strftime('%Y-%m-%d'),
            "Predicted Price (USD)": n_days_preds.flatten()
        })
        st.write(predicted_values_df)
    
    with col4:
        # Error metrics to measure performance (MAE, MSE, R squared)
        err_metrics = lstm_model.calculate_error_metrics(train_predictions=train_pred, test_predictions=test_pred, val_predictions=val_pred)
        
        # Display error metrics
        st.write("### Error Metrics")
        st.write(err_metrics)
