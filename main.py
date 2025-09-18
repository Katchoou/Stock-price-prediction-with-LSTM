import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf  
import os
from plotly.offline import iplot
import plotly.graph_objs as go

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Uncomment only if you need to disable oneDNN optimizations for troubleshooting


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
    return scaled_data

def example_with_Microsoft_stock(ticker = 'MSFT'):
    """
    This function is an example of how to use the above functions with Microsoft stock data
    @param ticker: Stock ticker symbol (default is 'MSFT' for Microsoft)
    """
    data = get_stock_data(ticker, '2000-01-03', '2025-09-01')
    new_data = data.copy()
    new_data = transformed_data(new_data)
    plot_stock_data(new_data, f"{ticker} Stock Price Data")
    data_scaled = normalise_data(new_data)
    print(type(data_scaled))
    print(data_scaled.head())
    return data_scaled



if __name__ == "__main__":
    data = example_with_Microsoft_stock('MSFT')
    data.describe()
    print(f"The total number of null values per variable is:\n {data.isnull().sum()}")
    print(f"The shape of the dataset is: {data.shape}")
    print(f"The columns of the dataset are: {data.columns}")










