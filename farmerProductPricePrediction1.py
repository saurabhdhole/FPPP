from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import requests
#import plotly.express as px
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
#from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ta
from ta.trend import SMAIndicator


#@st.cache
@st.experimental_singleton
# Load pre-existing data from folder
def load_pre_existing_data_from_folder(folder_path):
    try:
        data_frames = []

        for file_name in os.listdir(folder_path):

            # Check if the file is a CSV file
            if file_name.endswith(".csv"):
            
                # Construct the full file path
                file_path = os.path.join(folder_path, file_name)
            
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
            
                # Append the DataFrame to the list
                data_frames.append(df)

        combined_df = pd.concat(data_frames, ignore_index=True)

        return combined_df

    except Exception as e:
        st.error(f"Error loading pre-existing data: {e}")
        return pd.DataFrame()


# Load pre-existing data
@st.cache_data
def load_pre_existing_data(file_path):
    try:
        data = pd.read_excel(file_path)
        # Rename columns to clean up encoded names
        data.rename(columns={
            'Min_x0020_Price': 'Min_Price',
            'Max_x0020_Price': 'Max_Price',
            'Modal_x0020_Price': 'Modal_Price'
        }, inplace=True)
        data['Arrival_Date'] = pd.to_datetime(data['Arrival_Date'], errors='coerce')
        return data
    except Exception as e:
        st.error(f"Error loading pre-existing data: {e}")
        return pd.DataFrame()


# Filter markets based on selected state
def get_markets_for_state(df, state):
    return df[df['State'] == state]['Market'].unique()

def generate_future_dates(start_date, days=365):
    """
    Generate a list of future dates starting from `start_date` for the next `days` days.
    """
    return [start_date + timedelta(days=i) for i in range(days)]

# Fetch data from pre-existing file
def get_pre_existing_data(pre_existing_df, state, market, commodity, start_date, end_date):
        # Convert 'Arrival_Date' to datetime if it's not already
    if pre_existing_df['Arrival_Date'].dtype != 'datetime64[ns]':
        pre_existing_df['Arrival_Date'] = pd.to_datetime(pre_existing_df['Arrival_Date'], errors='coerce')

    # Drop rows with missing critical columns
    required_columns = ['State', 'Market', 'Commodity', 'Modal_Price', 'Arrival_Date']
    for col in required_columns:
        if col not in pre_existing_df.columns:
            raise KeyError(f"The input DataFrame does not contain the column '{col}'.")

    pre_existing_df = pre_existing_df.dropna(subset=['Arrival_Date', 'Modal_Price'])

    # Convert start_date and end_date to datetime
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    filtered_data = pre_existing_df[
        (pre_existing_df['State'] == state) &
        (pre_existing_df['Market'] == market) &
        (pre_existing_df['Commodity'] == commodity) &
        (pre_existing_df['Arrival_Date'] >= start_date) &
        (pre_existing_df['Arrival_Date'] <= end_date)
    ].copy()
    
    if filtered_data.empty:
        raise ValueError("No data found for the given filters.")

    # Further processing
    filtered_data['Modal_Price'] = pd.to_numeric(filtered_data['Modal_Price'], errors='coerce')
    filtered_data = filtered_data.dropna()  # Drop rows where 'Modal_Price' couldn't be converted
    filtered_data['Day'] = filtered_data['Arrival_Date'].dt.day
    filtered_data['Month'] = filtered_data['Arrival_Date'].dt.month
    filtered_data['Year'] = filtered_data['Arrival_Date'].dt.year

    # Encode 'Commodity' column
    le = LabelEncoder()
    filtered_data['Commodity_Code'] = le.fit_transform(filtered_data['Commodity'])

    return filtered_data

# Function to convert date from year-month-day format to day-month-year format
def convert_date_format(date_input):
    # Check if the input is a datetime.date object
    if isinstance(date_input, datetime):
        date_str = date_input.strftime('%Y-%m-%d')
    else:
        date_str = date_input
    # Parse the input date string
    date_obj = datetime.strptime(str(date_str), '%Y-%m-%d')
    # Convert to the desired format
    return date_obj.strftime('%d-%m-%Y')


# Function to fetch daily market prices from API
def fetch_market_prices(api_key, state, district, commodity, start_date, end_date, limit=999999999999):
    # st.write("Start Date:", convert_date_format(start_date))
    # st.write("End Date:",convert_date_format( end_date))
    # st.write("State:", state)
    # st.write("District:", district)
    resource_id = "35985678-0d79-46b4-9ed6-6f13308a1d24"  # Replace with actual resource ID
    url = f"https://api.data.gov.in/resource/{resource_id}?api-key={api_key}&format=json"
    url += f"&filters[State.keyword]={state}&filters[Market.keyword]={district}&filters[Commodity.keyword]={commodity}"
    url += f"&range[Arrival_Date][gte]={convert_date_format(start_date)}&range[Arrival_Date][lte]={convert_date_format(end_date)}"
    # url += f"&filters[Arrival_Date]<={convert_date_format(end_date)}&limit={limit}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'records' in data:
            return pd.DataFrame(data['records'])
        else:
            st.error("No records found for the selected criteria.")
            return pd.DataFrame()
    else:
        st.error("Failed to fetch data. Check your API key or query parameters.")
        return pd.DataFrame()

# Function to generate predictions (dummy implementation)
# def generate_predictions(prices_df):

#     column_name = 'Modal_Price'
#     if column_name not in prices_df.columns:
#         st.error(f"Column '{column_name}' not found in the data.")
#         return pd.DataFrame()  # Return an empty DataFrame if the column is missing

# #    prices_df['Modal_Price'] = pd.to_numeric(prices_df['Modal_Price'], errors='coerce')
# #    dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
# #    trend = np.linspace(prices_df['Modal_Price'].mean(), prices_df['Modal_Price'].mean() * 1.1, len(dates))
# #    prediction_df = pd.DataFrame({
# #        'date': dates,
# #        'predicted_price': trend
# #    })

#     features = ['Day', 'Month', 'Year', 'Commodity_Code']
#     target = 'Modal_Price'

#     # Train-test split
#     X = prices_df[features]
#     y = prices_df[target]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Generate future dates for prediction
#     future_dates = generate_future_dates(datetime.now(), 365)
#     future_data = pd.DataFrame({
#         'Day': [date.day for date in future_dates],
#         'Month': [date.month for date in future_dates],
#         'Year': [date.year for date in future_dates],
#         'Commodity_Code': [0] * 365  # Assume a default commodity
#     })

#     # Store predictions from different models
#     predictions = {}

#     # Linear Regression
#     lin_reg = LinearRegression()
#     lin_reg.fit(X_train, y_train)
#     predictions['Linear Regression'] = lin_reg.predict(future_data)

#     # Random Forest
#     rf = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf.fit(X_train, y_train)
#     predictions['Random Forest'] = rf.predict(future_data)

#     # XGBoost
#     xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
#     xgb.fit(X_train, y_train)
#     predictions['XGBoost'] = xgb.predict(future_data)

#     # ARIMA (for time series)
#     arima = ARIMA(prices_df['Modal_Price'], order=(1, 1, 0))  # Adjust order based on data
#     arima_model = arima.fit()
#     predictions['ARIMA'] = arima_model.forecast(steps=365)

#     # Combine results into a DataFrame
#     prediction_df = pd.DataFrame({
#         'Date': future_dates,
#         'Linear Regression': predictions['Linear Regression'],
#         'Random Forest': predictions['Random Forest'],
#         'XGBoost': predictions['XGBoost'],
#         'ARIMA': predictions['ARIMA']
#     })
    
#     return prediction_df


def generate_predictions1(prices_df):
    # Ensure 'date' and 'Modal_Price' columns exist
    required_columns = ['Arrival_Date', 'Modal_Price']
    if not all(col in prices_df.columns for col in required_columns):
        raise ValueError(f"Required columns {required_columns} not found in the data.")

    # Convert 'date' to datetime and set as index
    prices_df['Arrival_Date'] = pd.to_datetime(prices_df['Arrival_Date'])
    prices_df.set_index('Arrival_Date', inplace=True)

    # Handle missing values
    prices_df['Modal_Price'] = pd.to_numeric(prices_df['Modal_Price'], errors='coerce')
    prices_df.dropna(subset=['Modal_Price'], inplace=True)
    #prices_df.dropna()

    # Feature engineering
    prices_df['Day'] = prices_df.index.day
    prices_df['Month'] = prices_df.index.month
    prices_df['Year'] = prices_df.index.year
    prices_df['DayOfWeek'] = prices_df.index.dayofweek

    # Add technical indicators
    prices_df['SMA_5'] = SMAIndicator(close=prices_df['Modal_Price'], window=5).sma_indicator()
    prices_df['SMA_20'] = SMAIndicator(close=prices_df['Modal_Price'], window=20).sma_indicator()

    prices_df['RSI'] = ta.momentum.rsi(prices_df['Modal_Price'], window=14)
    prices_df['MACD'] = ta.trend.macd_diff(prices_df['Modal_Price'])

    # Add lag features
    for lag in [1, 3, 7]:
        prices_df[f'Price_Lag_{lag}'] = prices_df['Modal_Price'].shift(lag)

    # Drop rows with NaN values after feature engineering
    prices_df.dropna(inplace=True)



    # Define features and target
    features = ['Day', 'Month', 'Year', 'DayOfWeek', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 
                'Price_Lag_1', 'Price_Lag_3', 'Price_Lag_7']
    target = 'Modal_Price'

    # Create lag features
    # prices_df['Price_Lag_1'] = prices_df['Modal_Price'].shift(1)
    # prices_df['Price_Lag_3'] = prices_df['Modal_Price'].shift(3)
    # prices_df['Price_Lag_7'] = prices_df['Modal_Price'].shift(7)

    # Train-test split
    X = prices_df[features]
    y = prices_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Generate future dates for prediction
    last_date = prices_df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=365)
    future_data = pd.DataFrame(index=future_dates)
    future_data['Day'] = future_data.index.day
    future_data['Month'] = future_data.index.month
    future_data['Year'] = future_data.index.year
    future_data['DayOfWeek'] = future_data.index.dayofweek

    future_data = future_data.fillna(0)  

    # Fill NaN values in future_data with appropriate values
    for feature in features:
        if feature not in future_data.columns:
            future_data[feature] = 0  # or use an appropriate default value

    st.dataframe(future_data.isnull().sum())  

    # Initialize predictions dictionary
    predictions = {}

    st.write(X_train.isnull().sum())

    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)

    
    predictions['Linear Regression'] = lin_reg.predict(scaler.transform(future_data[features]))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    predictions['Random Forest'] = rf.predict(scaler.transform(future_data[features]))

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train_scaled, y_train)
    predictions['XGBoost'] = xgb.predict(scaler.transform(future_data[features]))

    # ARIMA
    arima = ARIMA(prices_df['Modal_Price'], order=(1, 1, 0))
    arima_model = arima.fit()
    predictions['ARIMA'] = arima_model.forecast(steps=365)

    # LSTM
    def create_lstm_model(input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    X_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    lstm_model = create_lstm_model((1, X_train_scaled.shape[1]))
    lstm_model.fit(X_lstm, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    future_data_lstm = scaler.transform(future_data[features]).reshape((future_data.shape[0], 1, len(features)))
    predictions['LSTM'] = lstm_model.predict(future_data_lstm).flatten()

    # Prophet
    try:
        from fbprophet import Prophet
        prophet_data = prices_df.reset_index()[['Arrival_Date', 'Modal_Price']]
        prophet_data.rename(columns={'Arrival_Date': 'ds', 'Modal_Price': 'y'}, inplace=True)
        prophet_model = Prophet()
        prophet_model.fit(prophet_data)
        
        # Future data for Prophet
        future_dates_prophet = prophet_model.make_future_dataframe(periods=365)
        forecast = prophet_model.predict(future_dates_prophet)
        
        # Add Prophet predictions
        predictions['Prophet'] = forecast[forecast['ds'].isin(future_dates)]['yhat'].values
    except ImportError:
        st.warning("Prophet is not installed. Skipping Prophet predictions.")

    # Evaluate models
    models = ['Linear Regression', 'Random Forest', 'XGBoost', 'LSTM']
    model_scores = {}
    for model in models:
        y_pred = globals()[model.lower().replace(' ', '_')].predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        model_scores[model] = (mse, mae)

    # Calculate weights based on inverse MSE
    total_inverse_mse = sum(1/score[0] for score in model_scores.values())
    weights = {model: (1/score[0])/total_inverse_mse for model, score in model_scores.items()}

    # Create weighted ensemble prediction
    ensemble_prediction = sum(weights[model] * predictions[model] for model in models)
    predictions['Weighted Ensemble'] = ensemble_prediction

    # Combine results into a DataFrame
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Linear Regression': predictions['Linear Regression'],
        'Random Forest': predictions['Random Forest'],
        'XGBoost': predictions['XGBoost'],
        'ARIMA': predictions['ARIMA'],
        'LSTM': predictions['LSTM'],
        #'Prophet': predictions['Prophet'],
        'Weighted Ensemble': predictions['Weighted Ensemble']
    })
    
    if 'Prophet' in predictions:
        prediction_df['Prophet'] = predictions['Prophet']
    
    return prediction_df, model_scores


@st.cache
def generate_predictions(prices_df):
    # Ensure required columns exist
    required_columns = ['Arrival_Date', 'Modal_Price']
    if not all(col in prices_df.columns for col in required_columns):
        raise ValueError(f"Required columns {required_columns} not found in the data.")

    # Convert 'Arrival_Date' to datetime and set as index
    prices_df['Arrival_Date'] = pd.to_datetime(prices_df['Arrival_Date'])
    prices_df.set_index('Arrival_Date', inplace=True)

    # Handle missing values
    prices_df['Modal_Price'] = pd.to_numeric(prices_df['Modal_Price'], errors='coerce')
    prices_df.dropna(subset=['Modal_Price'], inplace=True)

    # Feature engineering
    prices_df['Day'] = prices_df.index.day
    prices_df['Month'] = prices_df.index.month
    prices_df['Year'] = prices_df.index.year
    prices_df['DayOfWeek'] = prices_df.index.dayofweek

    # Add technical indicators
    prices_df['SMA_5'] = SMAIndicator(close=prices_df['Modal_Price'], window=5).sma_indicator()
    prices_df['SMA_20'] = SMAIndicator(close=prices_df['Modal_Price'], window=20).sma_indicator()
    prices_df['RSI'] = ta.momentum.rsi(prices_df['Modal_Price'], window=14)
    prices_df['MACD'] = ta.trend.macd_diff(prices_df['Modal_Price'])

    # Add lag features
    for lag in [1, 3, 7]:
        prices_df[f'Price_Lag_{lag}'] = prices_df['Modal_Price'].shift(lag)

    # Drop rows with NaN values after feature engineering
    prices_df.dropna(inplace=True)

    # Define features and target
    features = ['Day', 'Month', 'Year', 'DayOfWeek', 'SMA_5', 'SMA_20', 'RSI', 'MACD',
                'Price_Lag_1', 'Price_Lag_3', 'Price_Lag_7']
    target = 'Modal_Price'

    # Train-test split
    X = prices_df[features]
    y = prices_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Generate future dates for prediction
    last_date = prices_df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=365)
    future_data = pd.DataFrame(index=future_dates)
    future_data['Day'] = future_data.index.day
    future_data['Month'] = future_data.index.month
    future_data['Year'] = future_data.index.year
    future_data['DayOfWeek'] = future_data.index.dayofweek

    # Fill missing columns with historical mean or zeros
    for feature in features:
        if feature not in future_data.columns:
            future_data[feature] = prices_df[feature].mean() if feature in prices_df else 0

    future_data_scaled = scaler.transform(future_data[features])

    # Initialize models
    lin_reg = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Train models
    lin_reg.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)
    xgb.fit(X_train_scaled, y_train)

    # Predictions dictionary
    predictions = {
        'Linear Regression': lin_reg.predict(future_data_scaled),
        'Random Forest': rf.predict(future_data_scaled),
        'XGBoost': xgb.predict(future_data_scaled)
    }

    # ARIMA model
    arima = ARIMA(prices_df['Modal_Price'], order=(1, 1, 0))
    arima_model = arima.fit()
    predictions['ARIMA'] = arima_model.forecast(steps=365)

    # LSTM model
    def create_lstm_model(input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    X_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    lstm_model = create_lstm_model((1, X_train_scaled.shape[1]))
    lstm_model.fit(X_lstm, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

    future_data_lstm = future_data_scaled.reshape((future_data_scaled.shape[0], 1, future_data_scaled.shape[1]))
    predictions['LSTM'] = lstm_model.predict(future_data_lstm).flatten()

    # Evaluate models
    trained_models = {
        'Linear Regression': lin_reg,
        'Random Forest': rf,
        'XGBoost': xgb
    }
    model_scores = {}
    for model_name, model_instance in trained_models.items():
        y_pred = model_instance.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        model_scores[model_name] = {'MSE': mse, 'MAE': mae}

    # Ensemble prediction using inverse MSE weights
    epsilon = 1e-5
    total_inverse_mse = sum(1 / (score['MSE'] + epsilon) for score in model_scores.values())
    weights = {model: (1 / (score['MSE'] + epsilon)) / total_inverse_mse for model, score in model_scores.items()}

    ensemble_prediction = sum(weights[model] * predictions[model] for model in trained_models.keys())
    predictions['Weighted Ensemble'] = ensemble_prediction

    # Combine results into a DataFrame
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Linear Regression': predictions['Linear Regression'],
        'Random Forest': predictions['Random Forest'],
        'XGBoost': predictions['XGBoost'],
        'ARIMA': predictions['ARIMA'],
        'LSTM': predictions['LSTM'],
        'Weighted Ensemble': predictions['Weighted Ensemble']
    })

    return prediction_df, model_scores

@st.experimental_singleton
def populate_dropdowns(pre_existing_data):
  """
  Populates Streamlit sidebar with dropdown menus for State, Market, and Commodity.

  Args:
    pre_existing_data: DataFrame containing the data for the dropdowns.

  Returns:
    A tuple containing the selected State, Market, and Commodity.
  """

  if not pre_existing_data.empty:
    states = pre_existing_data['State'].dropna().unique().tolist()
    state = st.sidebar.selectbox("Select State", states)

    # Helper function to get markets for a given state
    def get_markets_for_state(df, state):
      return df[df['State'] == state]['Market'].dropna().unique().tolist()

    markets = get_markets_for_state(pre_existing_data, state)
    market = st.sidebar.selectbox("Select Market", markets)

    commodities = pre_existing_data['Commodity'].dropna().unique().tolist()
    commodity = st.sidebar.selectbox("Select Commodity", commodities)

  else:
    state, market, commodity = None, None, None

  return state, market, commodity



# Usage example:
# prediction_df, model_scores = generate_predictions(prices_df)
# print(prediction_df)
# print(model_scores)


# Function to display graphs
def display_price_graphs(filtered_data, commodity, market, state):
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data['Arrival_Date'], filtered_data['Min_Price'], label='Min Price', marker='o')
    plt.plot(filtered_data['Arrival_Date'], filtered_data['Max_Price'], label='Max Price', marker='o')
    plt.plot(filtered_data['Arrival_Date'], filtered_data['Modal_Price'], label='Modal Price', marker='o')
    plt.xlabel('Arrival Date')
    plt.ylabel('Price')
    plt.title(f'{commodity} Prices in {market} ({state})')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Load data from the uploaded file
#pre_existing_file = "Date-Wise-Prices-all-Commodity.xlsx"
#pre_existing_data = load_pre_existing_data(pre_existing_file)

pre_existing_data = load_pre_existing_data_from_folder("data")

# Streamlit App
st.title("Market Price Viewer and Predictor")

# Sidebar Inputs
st.sidebar.header("User Inputs")

state, market, commodity = populate_dropdowns(pre_existing_data) 


# Date range input
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=7))
end_date = st.sidebar.date_input("End Date", datetime.now())

# API Key for fetching new data
api_key = st.sidebar.text_input("Enter your Data.gov.in API Key", type="password")

# Data fetching options
data_source = st.sidebar.radio("Select Data Source", options=["Pre-Existing Data", "Fetch New Data"])

# Fetch and Display Data
if st.sidebar.button("Fetch Data"):
    if data_source == "Pre-Existing Data":
        st.header(f"Market Prices (From Pre-Existing Data)")
        filtered_data = get_pre_existing_data(pre_existing_data, state, market, commodity, start_date, end_date)
        if not filtered_data.empty:
            st.dataframe(filtered_data)
            
            # Display graphs for Min, Max, and Modal prices
            st.subheader("Price Trends")
            display_price_graphs(filtered_data, commodity, market, state)
            
            predictions, model_scores = generate_predictions(filtered_data)
            
            st.subheader("Predicted Prices")
            st.write("Prediction (Tabular Format)")
            st.dataframe(predictions)
            

            # Display the model scores
            # st.subheader("Model Performance")
            # st.write(model_scores)

            st.write("Prediction (Chart Format)")
            #fig = px.line(predictions, x='date', y='predicted_price', title="Price Prediction Chart")
            #st.plotly_chart(fig)
            chart_data = predictions.set_index('Date')
            st.line_chart(chart_data)
            
            # st.download_button(
            #     label="Download Filtered Data as CSV",
            #     data=filtered_data.to_csv(index=False),
            #     file_name=f"{commodity}_market_prices_filtered.csv",
            #     mime="text/csv"
            # )
        else:
            st.warning("No data available for the selected criteria in pre-existing data.")
    
    elif data_source == "Fetch New Data":
        api_key = "579b464db66ec23bdd000001d74c0d0c21a44b0572ae636312525017"
        if api_key:
            st.header(f"Market Prices (From API)")
            st.write("Current limit on API is to pull 10 records only.");
            fetched_data = fetch_market_prices(api_key, state, market, commodity, start_date, end_date)
            if not fetched_data.empty:
                filtered_data = get_pre_existing_data(fetched_data, state, market, commodity, start_date, end_date)
                st.dataframe(filtered_data)
                
                # Display graphs for Min, Max, and Modal prices
                st.subheader("Price Trends")
                display_price_graphs(filtered_data, commodity, market, state)
                
                predictions = generate_predictions(filtered_data)
                
                st.subheader("Predicted Prices")
                st.write("Prediction (Tabular Format)")
                st.dataframe(predictions)
                
                st.write("Prediction (Chart Format)")
                #fig = px.line(predictions, x='Date', y='predicted_price', title="Price Prediction Chart")
                fig = st.line(predictions, x='Arrival_Date', y=['Linear Regression', 'Random Forest', 'XGBoost', 'ARIMA'], title="Price Prediction Chart")
                st.plotly_chart(fig)
                
                # st.download_button(
                #     label="Download API Data as CSV",
                #     data=fetched_data.to_csv(index=False),
                #     file_name=f"{commodity}_market_prices_api.csv",
                #     mime="text/csv"
                # )
            else:
                st.warning("No data available for the selected criteria from API.")
        else:
            st.error("Please enter a valid API Key.")
