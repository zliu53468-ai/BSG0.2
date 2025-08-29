from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
CORS(app)  # 處理跨域請求

# 設置頻率限制
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# 設置快取
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# 設置日誌
log_handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
log_handler.setLevel(logging.INFO)
app.logger.addHandler(log_handler)

# 存儲歷史數據
historical_data = []

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
@cache.cached(timeout=60)
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # 檢查數據是否與歷史數據相同
        if is_data_identical(data, historical_data):
            app.logger.info("收到的路紙與現有歷史數據一致，使用快取結果。")
            # 返回快取結果
            return jsonify({"result": "cached_prediction", "status": "unchanged"})
        
        # 更新歷史數據
        historical_data.clear()
        historical_data.extend(data)
        
        # 進行預測
        prediction = make_prediction(data)
        
        return jsonify({"result": prediction, "status": "new_prediction"})
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

def is_data_identical(new_data, existing_data):
    # 實現數據比較邏輯
    return new_data == existing_data

def make_prediction(data):
    # 實現您的預測邏輯
    return "prediction_result"

if __name__ == '__main__':
    app.run(debug=True)
