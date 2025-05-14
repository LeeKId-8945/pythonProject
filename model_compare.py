from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


def compare_models(X, y):
	models = {
		"RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
		"XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
		"LogisticRegression": LogisticRegression(max_iter=1000)
	}

	html_reports = {}

	for name, model in models.items():
		model.fit(X, y)
		# 切分資料
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		# 訓練模型
		model.fit(X_train, y_train)

		# 預測測試資料
		y_pred = model.predict(X_test)

		# 用正確的真實值來比對預測結果
		report = classification_report(y_test, y_pred)
		html_reports[name] = f"<pre>{report}</pre>"

	return html_reports
