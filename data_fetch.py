import yfinance as yf
import pandas as pd

def get_stock_data(symbol, period='1y', interval='1d'):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period, interval=interval)
    df.reset_index(inplace=True)
    df['Volume'] = df['Volume'].fillna(0)
    return df