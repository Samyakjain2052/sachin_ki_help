import streamlit as st
import requests
import yfinance as yf
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import timedelta

# Function to fetch news from NewsAPI
def fetch_news(company_name):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': company_name,
        'apiKey': 'c29e4dda9ea540e8935663f392d2bb2b'  # Replace with your actual NewsAPI key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        st.error(f"Failed to fetch news. Status code: {response.status_code}")
        st.error(f"Response content: {response.content}")
        return []

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to fetch stock data and predict prices using LSTM
def predict_stock_prices_lstm(ticker):
    stock_data = yf.download(ticker, period='2y')
    if stock_data.empty:
        st.error("No data fetched. Please check the ticker symbol.")
        return None, None, None, None, None
    
    stock_data.reset_index(inplace=True)
    stock_data = stock_data[['Date', 'Close']]
    
    # Prepare data for LSTM
    data = stock_data['Close'].values
    data = data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step):
            a = dataset[i:(i + time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)
    
    time_step = 30
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    st.write(f"X_train shape: {X_train.shape}")
    st.write(f"y_train shape: {y_train.shape}")
    st.write(f"X_test shape: {X_test.shape}")
    st.write(f"y_test shape: {y_test.shape}")
    
    if X_test.shape[0] == 0 or y_test.shape[0] == 0:
        st.error("Not enough data to create test set. Please try a different stock ticker or adjust the time step.")
        return None, None, None, None, None
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
    
    # Predict and inverse transform
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Ensure predictions are 1D
    train_predict = scaler.inverse_transform(train_predict).flatten()
    test_predict = scaler.inverse_transform(test_predict).flatten()
    
    # Inverse transform the actual data
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Prepare dates for predictions
    train_dates = stock_data['Date'].iloc[time_step:train_size].values
    test_dates = stock_data['Date'].iloc[train_size + time_step:].values
    
    # Create DataFrames for plotting
    train_pred_df = pd.DataFrame({'Date': train_dates, 'Prediction': train_predict})
    test_pred_df = pd.DataFrame({'Date': test_dates, 'Prediction': test_predict})
    
    # Actual data up to the last prediction
    actual_df = stock_data.iloc[time_step:].reset_index(drop=True)
    
    # Generate 5-day future predictions
    future_predictions = []
    future_dates = []
    last_date = stock_data['Date'].iloc[-1]
    
    # Get the last `time_step` data points for prediction
    last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
    
    current_sequence = last_sequence.copy()
    
    for i in range(5):
        next_pred_scaled = model.predict(current_sequence)
        next_pred = scaler.inverse_transform(next_pred_scaled)[0][0]
        future_predictions.append(next_pred)
        next_date = last_date + timedelta(days=1)
        # Ensure the next_date is a trading day (Monday to Friday)
        while next_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            next_date += timedelta(days=1)
        future_dates.append(next_date)
        # Update the last_date for the next iteration
        last_date = next_date
        # Update the current_sequence
        next_pred_scaled = scaler.transform(np.array([[next_pred]]))
        # Reshape to (1, 1, 1) for appending
        next_pred_scaled = next_pred_scaled.reshape(1, 1, 1)
        current_sequence = np.append(current_sequence[:, 1:, :], next_pred_scaled, axis=1)
    
    future_pred_df = pd.DataFrame({'Date': future_dates, 'Prediction': future_predictions})
    
    return stock_data, train_pred_df, test_pred_df, actual_df, future_pred_df

# Streamlit app
st.title("Company News and Stock Price Prediction")

# Sidebar for navigation
option = st.sidebar.selectbox("Choose a section", ["News Sentiment Analysis", "Stock Price Prediction"])

if option == "News Sentiment Analysis":
    st.header("News Sentiment Analysis")
    company_name = st.text_input("Enter a company name:", "Apple")

    if company_name:
        news_data = fetch_news(company_name)

        if news_data:
            for news in news_data:
                title = news.get('title', 'No title')
                summary = news.get('description', 'No description')

                st.subheader(title)
                st.write(summary)

                sentiment = analyze_sentiment(summary)
                if sentiment > 0:
                    st.write("Sentiment: Positive")
                elif sentiment < 0:
                    st.write("Sentiment: Negative")
                else:
                    st.write("Sentiment: Neutral")

elif option == "Stock Price Prediction":
    st.header("Stock Price Prediction")
    ticker = st.text_input("Enter the stock ticker:", "AAPL")

    if ticker:
        stock_data, train_pred_df, test_pred_df, actual_df, future_pred_df = predict_stock_prices_lstm(ticker)

        if stock_data is not None:
            st.subheader("Stock Price Prediction")
            
            # Convert actual data back to a flat array for plotting
            original_data = actual_df['Close'].values
            
            fig, ax = plt.subplots(figsize=(14,7))
            ax.plot(actual_df['Date'], original_data, label='Actual Price', color='blue')
            
            if train_pred_df is not None and not train_pred_df.empty:
                ax.plot(train_pred_df['Date'], train_pred_df['Prediction'], label='Train Prediction', color='green')
            if test_pred_df is not None and not test_pred_df.empty:
                ax.plot(test_pred_df['Date'], test_pred_df['Prediction'], label='Test Prediction', color='red')
            if future_pred_df is not None and not future_pred_df.empty:
                ax.plot(future_pred_df['Date'], future_pred_df['Prediction'], label='Future Prediction', color='orange', linestyle='--')
            
            ax.legend()
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(f'Stock Price Prediction for {ticker.upper()}')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Display 5-day future predictions
            if future_pred_df is not None and not future_pred_df.empty:
                st.subheader("5-Day Future Predictions")
                st.table(future_pred_df.set_index('Date'))