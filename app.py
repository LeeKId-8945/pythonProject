# app.py
from flask import Flask, render_template, request, jsonify
from data_fetch import get_stock_data
from feature_engineering import add_technical_indicators
from predictor import predict_action
from model_train import label_data
from ml_evaluation import evaluate_model
import os
import joblib
import pandas as pd
from model_compare import compare_models

app = Flask(__name__)
MODEL_PATH = 'model/rf_model.pkl'

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    chart_data = {}
    backtest_curve = []
    report_html = None
    symbol = ""
    model_reports = None

    if request.method == 'POST':
        symbol = request.form['symbol'].strip()
        if symbol:
            try:
                df = get_stock_data(symbol)
                df = add_technical_indicators(df)
                df = df.tail(100).reset_index()

                if not os.path.exists(MODEL_PATH):
                    result = '⚠️ 尚未訓練模型，請先執行完整訓練。'
                else:
                    result = predict_action(df, model_path=MODEL_PATH)

                    chart_data = {
                        'Date': df['Date'].astype(str).tolist(),
                        'Close': df['Close'].tolist(),
                        'RSI': df['RSI'].tolist(),
                        'MACD': df['MACD'].tolist(),
                        'MACD_Signal': df['MACD_Signal'].tolist(),
                        'Bollinger_Upper': df['Bollinger_Upper'].tolist(),
                        'Bollinger_Lower': df['Bollinger_Lower'].tolist(),
                        'Fibonacci_0618': df['Fibonacci_0618'].tolist(),
                        'OBV': df['OBV'].tolist()
                    }

                    df = label_data(df)
                    model = joblib.load(MODEL_PATH)
                    features = ['RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower', 'Fibonacci_0618', 'OBV']
                    X = df[features]
                    y = df['Target']

                    report_html, y_pred = evaluate_model(model, X, y)
										
                    model_reports = compare_models(X, y)

                    # 回測資產曲線
                    capital = 100000
                    position = 0
                    for i in range(len(df) - 1):
                        price = df.iloc[i]['Close']
                        if y_pred[i] == 1 and capital > 0:
                            position = capital / price
                            capital = 0
                        elif y_pred[i] == 0 and position > 0:
                            capital = position * price
                            position = 0
                        total = capital + position * price
                        backtest_curve.append(total)

                    if position > 0:
                        capital = position * df.iloc[-1]['Close']
                        backtest_curve.append(capital)

            except Exception as e:
                result = f"❌ 錯誤：{e}"

    return render_template(
    'index_plotly.html',
    result=result,
    symbol=symbol,
    chart_data=chart_data,
    backtest_curve=backtest_curve,
    report_html=report_html,
    model_reports=model_reports
)


@app.route('/api/chart_data', methods=['POST'])
def api_chart_data():
    symbol = request.json.get("symbol")
    if not symbol:
        return jsonify({"error": "missing symbol"}), 400

    df = get_stock_data(symbol)
    df = add_technical_indicators(df)
    df = df.tail(100).reset_index()

    data = {
        'Date': df['Date'].astype(str).tolist(),
        'Close': df['Close'].tolist(),
        'RSI': df['RSI'].tolist(),
        'MACD': df['MACD'].tolist(),
        'MACD_Signal': df['MACD_Signal'].tolist(),
        'Bollinger_Upper': df['Bollinger_Upper'].tolist(),
        'Bollinger_Lower': df['Bollinger_Lower'].tolist(),
        'Fibonacci_0618': df['Fibonacci_0618'].tolist(),
        'OBV': df['OBV'].tolist()
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
