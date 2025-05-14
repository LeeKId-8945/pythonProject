from data_fetch import get_stock_data
from feature_engineering import add_technical_indicators
from predictor import predict_action
import joblib

def quick_predict(symbol, model_path='model/rf_model.pkl'):
    df = get_stock_data(symbol)
    df = add_technical_indicators(df)
    action = predict_action(df, model_path=model_path)
    print(f"\n{symbol} 的快速預測結果：{action}")
    return action