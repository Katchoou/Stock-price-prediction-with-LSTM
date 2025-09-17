import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf  
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def get_stock_data(ticker, start_date, end_date):
    """
    This function is designed to fetch historical stock data
    @param ticker: Stock ticker symbol
    @param start_date: Start date for fetching data (YYYY-MM-DD)
    @param end_date: End date for fetching data (YYYY-MM-DD)
    @return: DataFrame containing historical stock data
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def plot_stock_data(data, title):
    """
    This function is built to plot the opening and closing price data of the stock
    :param data: DataFrame containing stock data
    :param title: The title prefered for the plot
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label="Closing Price", color='red', lw=1.3)
    plt.plot(data['Open'], label="Opening Price", color='blue', lw=1.3)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def preprocessing(data):
    """
    This function preprocesses the stock data for model development
    @param data: DataFrame containing stock data
    @return: Scaled data and scaler object
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return data_scaled, scaler

ticker = 'MSFT'
data = get_stock_data(ticker, '2019-01-01', '2025-09-01')
data = pd.read_csv(
    f"E:\\Formations\\Africa Techup 2025\\projets\\Stock-price-forecasting-\\{ticker}_data.csv",
    index_col=0, skiprows=2
)
data.to_csv(f"{ticker}_data.csv", index=True)
new_data = data.copy()
new_data.reset_index(inplace=True)
new_data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
print(new_data.head())
print(new_data.columns)

plot_stock_data(new_data, f"{ticker} Stock Price Data")
data_scaled, scaler = preprocessing(new_data)
print(data_scaled)






