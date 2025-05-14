import pandas as pd

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def compute_bollinger_bands(series, window=20):
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return upper, lower

def compute_fibonacci_retracement(series, window=20):
    max_price = series.rolling(window=window).max()
    min_price = series.rolling(window=window).min()
    level_0_618 = max_price - 0.618 * (max_price - min_price)
    return level_0_618

def compute_obv(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - volume[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

def add_technical_indicators(df):
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])
    df['Bollinger_Upper'], df['Bollinger_Lower'] = compute_bollinger_bands(df['Close'])
    df['Fibonacci_0618'] = compute_fibonacci_retracement(df['Close'])
    df['OBV'] = compute_obv(df['Close'], df['Volume'])
    return df.dropna().reset_index(drop=True)
