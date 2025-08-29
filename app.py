from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler
import os

app = Flask(__name__)

# 更完善的 CORS 配置
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "https://your-frontend-domain.com"],  # 替換為您的前端域名
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# 設置日誌
if not os.path.exists('logs'):
    os.makedirs('logs')

log_handler = RotatingFileHandler('logs/app.log', maxBytes=10000, backupCount=3)
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
log_handler.setFormatter(formatter)
app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO)

# 存儲歷史數據
historical_data = []

# 添加根路徑路由
@app.route('/')
def home():
    app.logger.info("根路徑被訪問")
    return jsonify({
        "message": "預測服務已啟動",
        "endpoints": {
            "predict": "/predict (POST)"
        },
        "status": "active"
    })

# 健康檢查端點
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": "2025-08-29T07:00:00Z"})

# 處理 OPTIONS 請求（CORS 預檢請求）
@app.route('/predict', methods=['OPTIONS'])
def handle_options():
    app.logger.info("處理 OPTIONS 請求")
    response = jsonify({"status": "ok"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.info("收到預測請求")
        
        # 記錄請求頭部信息
        app.logger.info(f"請求頭: {dict(request.headers)}")
        
        # 檢查是否為預檢請求
        if request.method == 'OPTIONS':
            return handle_options()
            
        # 獲取 JSON 數據
        if not request.is_json:
            app.logger.warning("請求不是 JSON 格式")
            return jsonify({"error": "請求必須是 JSON 格式"}), 400
            
        data = request.get_json()
        app.logger.info(f"收到數據: {data}")
        
        if not data:
            app.logger.warning("未提供數據")
            return jsonify({"error": "未提供數據"}), 400
        
        # 檢查數據是否與歷史數據相同
        if is_data_identical(data, historical_data):
            app.logger.info("收到的路紙與現有歷史數據一致，使用快取結果。")
            return jsonify({
                "result": "cached_prediction", 
                "status": "unchanged",
                "timestamp": "2025-08-29T07:00:00Z"
            })
        
        # 更新歷史數據
        historical_data.clear()
        historical_data.extend(data)
        
        # 進行預測
        prediction = make_prediction(data)
        app.logger.info(f"生成預測結果: {prediction}")
        
        response = jsonify({
            "result": prediction, 
            "status": "new_prediction",
            "timestamp": "2025-08-29T07:00:00Z"
        })
        
        # 添加 CORS 頭部
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        
        return response
        
    except Exception as e:
        app.logger.error(f"預測錯誤: {str(e)}", exc_info=True)
        return jsonify({"error": "內部伺服器錯誤", "details": str(e)}), 500

def is_data_identical(new_data, existing_data):
    # 實現數據比較邏輯
    return new_data == existing_data

def make_prediction(data):
    # 實現您的預測邏輯
    # 這裡應該包含您的 XGBoost 和 HMM 模型預測代碼
    # 暫時返回模擬數據
    return {
        "prediction": "示例預測結果",
        "confidence": 0.85,
        "model_used": "XGBoost"
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.logger.info(f"啟動應用程式，端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
