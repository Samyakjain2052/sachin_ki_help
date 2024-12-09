import streamlit as st
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

# Function to fetch AQI data from the API
def fetch_aqi_data(city):
    token = 'afe3727768bcfe9daaa1dd62dec00d44b9ff0786'  # Replace with your API token
    url = f'https://api.waqi.info/feed/{city}/?token={token}'
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'ok':
        aqi = data['data']['aqi']
        return aqi
    else:
        return None

# Streamlit app
st.title('Air Quality Index (AQI) Prediction for Indian Cities')

# Input city name
city = st.text_input('Enter the city name:', 'Delhi')

# Fetch AQI data
aqi = fetch_aqi_data(city)
if aqi is not None:
    st.write(f'The current AQI for {city} is {aqi}')
else:
    st.write('Could not fetch AQI data. Please check the city name or try again later.')

# Generate sample data for prediction
days = np.array(range(1, 31)).reshape(-1, 1)
aqi_values = np.random.randint(50, 200, size=(30,))  # Random AQI values for demonstration
data = pd.DataFrame({'Day': days.flatten(), 'AQI': aqi_values})

# Train the model
model = LinearRegression()
model.fit(data[['Day']], data['AQI'])

# Make predictions for the next 30 days
future_days = np.array(range(31, 61)).reshape(-1, 1)
predictions = model.predict(future_days)

# Generate dates for the next 30 days
start_date = datetime.date.today()
future_dates = [start_date + datetime.timedelta(days=i) for i in range(1, 31)]

# Display the predictions with dates
st.write("## AQI Predictions for the Next 30 Days")
predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted AQI': predictions})
st.write(predictions_df)

# Plot the predictions
st.line_chart(predictions_df.set_index('Date'))