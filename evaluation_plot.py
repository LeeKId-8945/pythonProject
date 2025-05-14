# evaluation_plot.py
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def plot_prediction_vs_actual(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred, y_true, alpha=0.6, color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='理想線')
    plt.xlabel('模型預測')
    plt.ylabel('實際值')
    plt.title('預測 vs. 實際')
    plt.legend()
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64
