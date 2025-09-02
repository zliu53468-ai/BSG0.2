# -*- coding: utf-8 -*-
import os
import json
import logging
import time
import re
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from line_bot_webhook import linebot_bp
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Flask 應用程式設定
# =============================================================================
app = Flask(__name__)
app.register_blueprint(linebot_bp)

# =============================================================================
# 日誌設定
# =============================================================================
if not os.path.exists('logs'):
    os.makedirs('logs')
log_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=5)
log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
app.logger.handlers.clear()
app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO)
CORS(app, resources={r"/*": {"origins": "*"}})

# =============================================================================
# 全域變數與模型預載 (新增完整模型載入程式碼)
# =============================================================================
MODEL_DIR = 'models'
N_FEATURES_WINDOW = 20
LABEL_MAP = {'B': 0, 'P': 1}
REVERSE_MAP = {0: 'B', 1: 'P'}

try:
    # 載入 XGBoost 模型
    xgboost_model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    # 載入 HMM 模型 
    hmm_model = joblib.load(os.path.join(MODEL_DIR, 'hmm_model.pkl'))
    # 載入 LightGBM 模型
    lgb_model = joblib.load(os.path.join(MODEL_DIR, 'lgb_model.pkl'))
    app.logger.info("所有模型載入成功")
except Exception as e:
    app.logger.error(f"模型載入失敗: {str(e)}")
    raise RuntimeError(f"模型初始化失敗: {str(e)}")

# =============================================================================
# 預測路由 (新增完整預測功能)
# =============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        data = request.get_json()
        roadmap = data.get('roadmap', [])
        
        if len(roadmap) < N_FEATURES_WINDOW:
            return jsonify({"error": f"至少需要{N_FEATURES_WINDOW}筆歷史資料"}), 400

        # 特徵工程
        features = np.array([
            [1 if roadmap[-i-1] == 'B' else 0 for i in range(N_FEATURES_WINDOW)] +
            [sum(1 for x in roadmap[-N_FEATURES_WINDOW:] if x == 'B')]
        ])

        # 多模型預測
        xgb_proba = xgboost_model.predict_proba(features)[0]
        lgb_proba = lgb_model.predict_proba(features)[0]
        hmm_pred = hmm_model.predict(features.reshape(-1, 1))
        
        # 集成預測結果
        final_proba = (xgb_proba + lgb_proba) / 2
        prediction = np.argmax(final_proba)
        
        return jsonify({
            "prediction": REVERSE_MAP[prediction],
            "probability": float(final_proba[prediction]),
            "processing_time": f"{time.time()-start_time:.3f}s"
        })
        
    except Exception as e:
        app.logger.error(f"預測錯誤: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =============================================================================
# 主程式
# =============================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
```
