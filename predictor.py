import joblib
import pandas as pd

def predict_action(df, model_path='model/rf_model.pkl'):
    features = ['RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower', 'Fibonacci_0618', 'OBV']
    latest = df[features].iloc[-1:]
    model = joblib.load(model_path)

    proba = model.predict_proba(latest)[0]
    predicted_class = model.predict(latest)[0]
    confidence = proba[predicted_class] * 100

    if predicted_class == 1:
        return f"✅ 建議：買入（信心 {confidence:.1f}％）"
    else:
        return f"❌ 建議：賣出（信心 {confidence:.1f}％）"
