from data_fetch import get_stock_data
from feature_engineering import add_technical_indicators
from model_train import label_data
from predictor import predict_action
import joblib
import os
import matplotlib.pyplot as plt

MODEL_PATH = 'model/rf_model.pkl'

def run_backtest(symbol):
    if not os.path.exists(MODEL_PATH):
        print("❌ 請先訓練模型再執行回測。")
        return

    df = get_stock_data(symbol)
    df = add_technical_indicators(df)
    df = label_data(df)

    model = joblib.load(MODEL_PATH)
    features = ['RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower', 'Fibonacci_0618', 'OBV']
    df['Prediction'] = model.predict(df[features])

    capital = 100000
    position = 0
    equity_curve = []

    for i in range(len(df) - 1):
        price_today = df.iloc[i]['Close']
        price_tomorrow = df.iloc[i + 1]['Close']
        prediction = df.iloc[i]['Prediction']

        # 模型建議買入
        if prediction == 1 and capital > 0:
            position = capital / price_today
            capital = 0
        # 模型建議賣出
        elif prediction == 0 and position > 0:
            capital = position * price_today
            position = 0

        total_equity = capital + position * price_today
        equity_curve.append(total_equity)

    # 最終結算
    final_price = df.iloc[-1]['Close']
    if position > 0:
        capital = position * final_price

    final_value = capital
    total_return = (final_value - 100000) / 100000 * 100
    print(f"\n💰 最終資產：{final_value:.2f} 元")
    print(f"📈 總報酬率：{total_return:.2f}%")

    # 畫資產曲線
    plt.plot(equity_curve)
    plt.title(f"{symbol} 策略回測資產變化")
    plt.xlabel("時間 (交易日)")
    plt.ylabel("總資產")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    symbol = input("請輸入股票代碼進行策略回測：").strip()
    run_backtest(symbol)