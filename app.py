from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

app = Flask(__name__)
# 讓日誌直接輸出到控制台，由 Render 平台收集
logging.basicConfig(level=logging.INFO)

# 處理跨域請求
CORS(app)

# 設置頻率限制 (限制來自同一個 IP 的請求)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["500 per day", "100 per hour"]
)

# 注意：在免費的雲端平台上，簡單的記憶體快取和變數會在服務休眠後重置。
# 這裡保留基本架構，但要知道數據不是永久性的。
historical_data = []

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "success", "message": "AI Engine is running."})

@app.route('/predict', methods=['POST'])
@limiter.limit("20 per minute") # 限制每分鐘最多 20 次請求
def predict():
    try:
        data = request.get_json()
        
        if not data or 'roadmap' not in data:
            return jsonify({"error": "無效的請求，缺少 'roadmap' 數據"}), 400
        
        roadmap = data['roadmap']
        
        # 這裡應該放入您完整的 AI 預測邏輯
        # make_prediction(roadmap)
        # 為了演示，我們先回傳一個簡單的結果
        prediction = {
            "banker": 0.55,
            "player": 0.40,
            "tie": 0.05,
            "details": {
                "analysis": "這是一個示範回應"
            }
        }
        
        app.logger.info(f"成功預測，路紙長度: {len(roadmap)}")
        
        return jsonify(prediction)
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "伺服器內部錯誤"}), 500

if __name__ == '__main__':
    # 這個區塊僅供本地測試，在 Render 上會由 gunicorn 啟動
    app.run(host="0.0.0.0", port=8000, debug=True)
