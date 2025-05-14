from data_fetch import get_stock_data
from feature_engineering import add_technical_indicators
from model_train import label_data, train_model_with_gridsearch
from predictor import predict_action
import os
import joblib


MODEL_PATH = 'model/rf_model.pkl'

def full_train_mode(symbol):
    print(f"\n[完整訓練] 下載 {symbol} 並訓練模型...")
    df = get_stock_data(symbol)
    df = add_technical_indicators(df)
    df = label_data(df)
    model = train_model_with_gridsearch(df, model_path=MODEL_PATH)
    action = predict_action(df, model_path=MODEL_PATH)
    print(f"\n📈 {symbol} 操作建議：{action}")

def quick_predict_mode(symbol):
    print(f"\n[快速預測] 載入模型預測 {symbol} ...")
    df = get_stock_data(symbol)
    df = add_technical_indicators(df)
    action = predict_action(df, model_path=MODEL_PATH)
    print(f"\n⚡ {symbol} 快速預測建議：{action}")

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)

    while True:
        symbol = input("\n請輸入股票代碼（或輸入 'exit' 離開）：").strip()
        if symbol.lower() == 'exit':
            break

        try:
            if os.path.exists(MODEL_PATH):
                quick_predict_mode(symbol)
            else:
                full_train_mode(symbol)
        except Exception as e:
            print(f"[錯誤] 處理 {symbol} 時發生錯誤：{e}")

