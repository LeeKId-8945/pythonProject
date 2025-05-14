from sklearn.metrics import classification_report

def evaluate_model(model, X, y):
    try:
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=False)
        return f"<pre>{report}</pre>", y_pred
    except Exception as e:
        return f"<pre>模型評估錯誤：{e}</pre>", None