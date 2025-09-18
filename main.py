import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf  
import os
from plotly.offline import plot
import plotly.graph_objs as go
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Set TensorFlow logging level to suppress detailed logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '3'
def get_stock_data(ticker, start_date, end_date):
    """
    This function is designed to fetch historical stock data
    @param ticker: Stock ticker symbol
    @param start_date: Start date for fetching data (YYYY-MM-DD)
    @param end_date: End date for fetching data (YYYY-MM-DD)
    @return: DataFrame containing historical stock data
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    data = yf.Ticker(ticker)
    new_data = data.history(start=start_date, end=end_date)
    if new_data.empty:
        raise ValueError("No data found for the given ticker and date range.")
    return new_data

def transformed_data(data):
    """
    This function transforms the index of the DataFrame to datetime format
    it also drops unnecessary columns
    @param data: DataFrame containing stock data
    @return: DataFrame with datetime index
    """
    data = data.drop(columns=['Dividends', 'Stock Splits'], axis=1)
    data.index = pd.to_datetime(data.index)
    return data

def plot_stock_data(data, title):
    """
    This function is built to plot the opening and closing price data of the stock
    @param data: DataFrame containing stock data
    @param title: The title prefered for the plot
    """
    data.plot(subplots=True, figsize=(12, 14))
    plt.figure(figsize=(12, 8))
    plt.plot(data['Close'], label="Closing Price", color='blue', lw=1.5, linestyle="--")
    plt.plot(data['Open'], label="Opening Price", color='red', lw=1.3)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def normalise_data(data):
    """
    This function normalises the close price of the stock data for model development
    @param data: DataFrame containing stock data
    @return: Scaled data object
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = data.copy()
    scaled_data["Close"] = scaler.fit_transform(data[["Close"]])
    scaled_data["High"] = scaler.fit_transform(data[["High"]])
    scaled_data["Low"] = scaler.fit_transform(data[["Low"]])
    scaled_data["Open"] = scaler.fit_transform(data[["Open"]])
    scaled_data["Volume"] = scaler.fit_transform(data[["Volume"]])
    return scaled_data, scaler

def plotting_with_plotly(data):
    """
    This function plots the stock data using plotly so that we can see the price movements
    using candlestick charts
    The color indicates whether the stock closed higher (green) or lower (red) than it opened
    @param data: DataFrame containing stock data
    @param title: The title prefered for the plot
    """
    trace = go.Ohlc(x=data.index, 
                    open=data['Open'], 
                    high=data['High'], 
                    low=data['Low'], 
                    close=data['Close'], 
                    name='OHLC Charts of the price movements')
    fig = go.Figure(data=[trace])
    plot(fig)

def moving_average(data, window_size=20):
    """
    This function calculates the moving average of the stock's closing price
    @param data: DataFrame containing stock data
    @param window_size: The window size for calculating the moving average (default is 50)
    @return: Series containing the moving average
    """
    ma = data['Close'].rolling(window=window_size).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(data["Close"], label='Closing Price', color = "red")
    plt.plot(ma, label=f'{window_size}-Days Moving Average', color='blue')
    plt.title(f'Plot of the Closing price and its {window_size}-Days Moving Average')
    plt.legend()
    plt.show()

def data_train_test(data):
    """
    This function separates the data into training and testing sets
    @param data: DataFrame containing stock data
    @param train_size: Proportion of data to be used for training (default is 0.8)
    @return: Training and testing datasets
    """
    Close = data[["Close"]]
    train = Close[Close.index < '2022-01-01']
    test = Close[Close.index >= '2022-01-01']
    print(f"The shape of the training data is: {train.shape}")
    print(f"The shape of the testing data is: {test.shape}")
    return train, test

def create_timeseries_generator(train, test, n_input=30):
    """
    This function creates a time series generator for the stock data
    @param train: DataFrame containing the train stock data
    @param test: DataFrame containing the test stock data
    @return: TimeseriesGenerator object
    """
    train_array = train.values
    test_array = test.values

    train_gen = TimeseriesGenerator(train_array, 
                                    train_array, 
                                    length=n_input, 
                                    batch_size=32, 
                                    shuffle=False,
                                    sampling_rate=1,
                                    stride=1)

    test_gen = TimeseriesGenerator(test_array, 
                                   test_array,
                                   length=n_input,
                                   batch_size=32,
                                   shuffle=False,
                                   sampling_rate=1)
    return train_gen, test_gen

def lstm_model(n_neuron = 8, n_input=30):
    n_features = 1
    n_input = n_input
    model = Sequential()
    model.add(LSTM(n_neuron,  
                   input_shape=(n_input, n_features))
                   )
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def run_model(model, train_gen, epochs=200):
    """
    This function runs the LSTM model on the training and testing data
    @param model: The LSTM model to be trained
    @param train_gen: TimeseriesGenerator object for training data
    @param epochs: Number of epochs to train the model (default is 50)
    @return: History object containing training history
    """
    history = model.fit(train_gen, 
                        epochs=epochs, 
                        verbose=1)
    return history



def example_with_Microsoft_stock(ticker = 'MSFT'):
    """
    This function is an example of how to use the above functions with Microsoft stock data
    @param ticker: Stock ticker symbol (default is 'MSFT' for Microsoft)
    """
    data = get_stock_data(ticker, '2000-01-03', '2025-09-01')
    new_data = data.copy()
    new_data = transformed_data(new_data)
    #plot_stock_data(new_data, f"{ticker} Stock Price Data")
    data_scaled, scaler = normalise_data(new_data)
    train, test = data_train_test(data_scaled)
    train_gen, test_gen = create_timeseries_generator(train, test, n_input=30)
    model = lstm_model(n_neuron=4, n_input=30)
    history = run_model(model, train_gen, epochs=100)
    #print(type(data_scaled))
    #print(data_scaled.head())
    return history, model, test, test_gen, scaler

if __name__ == "__main__":
    history, model, test, test_gen, scaler = example_with_Microsoft_stock('MSFT')
    #data.describe()
    #print(f"The total number of null values per variable is:\n {data.isnull().sum()}")
    #print(f"The shape of the dataset is: {data.shape}")
    #print(f"The columns of the dataset are: {data.columns}")
    #plotting_with_plotly(data, title="Microsoft Stock Price Data")
    #moving_average(data, window_size=50)
    predictions = model.predict(test_gen)
    test['Close'] = scaler.inverse_transform(test[['Close']])
    test_predictions   = scaler.inverse_transform(predictions)
    print(model.summary())
    plt.figure(figsize=(14, 12))
    plt.plot(test.index, test['Close'], label='Actual Price', color='blue')
    plt.plot(test.index[30:], test_predictions, label='Predicted Price', color='red')
    plt.title('Stock Price Predictions vs Actual Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    plt.savefig('stock_price_predictions.png')











