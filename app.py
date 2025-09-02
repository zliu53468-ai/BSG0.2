# -*- coding: utf-8 -*-
import os
import json
import logging
import time
import traceback
from logging.handlers import RotatingFileHandler
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# =============================================================================
# 初始化設定
# =============================================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# =============================================================================
# 日誌系統 (強化錯誤追蹤)
# =============================================================================
if not os.path.exists('logs'):
    os.makedirs('logs')

log_handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
log_handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
))
app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO)

# =============================================================================
# 核心參數設定
# =============================================================================
MODEL_DIR = 'models'
DATA_FILE = 'data/history_data.json'
LABEL_MAP = {'B': 0, 'P': 1, 'T': 2}
REVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}
CHINESE_MAP = {'庄': 'B', '莊': 'B', '闲': 'P', '閒': 'P', '和': 'T'}
N_FEATURES_WINDOW = 20
PRELOADED_DATA = ["B","P","T","B","T","B","P","B","P","P","B","B","T","B","B","P","B","B","P","B","B","T","P","B","B","T","P","B","P","B","P","B","B","T","P","T","B","B","P","P","B","P","B","P","T","P","B","B","B","P","B","B","B","B","P","P","P","B","P","B","P","B","P","B","T","P","B","B","P","B","P","T","B","B","P","B","B","P","T","T","B","P","B","B","P","P","B","P","B","P","T","P","B","P","B","P","T","T","B","P","P","P","B","B","B","B","T","T","T","B","B","B","B","B","B","P","P","P","T","P","T","B","P","P","T","P","B","P","P","B","P","P","P","P","B","P","B","P","P","B","B","P","B","B","B","B","P","P","P","P","P","T","P","B","P","P","B","T","B","B","B","B","P","B","B","B","B","B","B","P","B","P","P","B","P","P","B","P","B","B","P","B","P","B","P","P","T","P","B","P","B","B","P","P","T","B","B","P","P","B","T","T","B","P","B","B","B","T","T","B","B","P","B","T","P","B","P","B","P","P","P","B","P","B","P","P","B","P","P","P","P","B","B","P","P","T","P","B","B","P","P","B","T","B","B","P","P","P","T","P","B","T","P","B","B","P","B","B","T","T","B","B","P","B","B","B","B","B","B","P","B","T","T","P","B","B","B","P","B","B","P","B","P","B","P","B","P","P","P","P","P","P","P","B","B","B","P","T","P","B","T","B","B","B","B","T","B","P","B","B","B","B","B","B","P","B","P","B","B","P","P","B","P","P","P","P","P","B","B","B","B","B","T","B","B","P","B","P","T","P","B","P","B","B","P","B","B","B","P","P","P","B","P","P","B","P","P","B","B","P","P","B","P","B","B","B","B","B","B","B","B","P","T","P","B","P","B","P","P","B","B","P","B","P","P","T","B","B","P","P","B","B","P","B","B","T","P","P","B","T","P","B","B","P","B","P","B","P","B","B","B","B","B","P","P","P","B","B","P","P","B","T","P","P","B","T","B","P","P","P","B","B","P","B","B","B","P","B","P","P","B","B","B","B","B","P","P","T","B","B","P","P","B","P","B","P","P","P","P","B","B","P","P","B","P","P","T","P","P","P","B","P","P","P","B","B","B","P","P","B","P","B","B","T","P","B","P","P","T","P","P","P","B","B","P","P","T","P","T","B","T","P","B","P","P","B","B","P","P","P","B","B","P","P","B","P","T","P","P","P","B","B","P","P","B","P","B","P","B","B","P","T","B","P","T","T","P","T","极","T","P","T","P","T","P","P","B","B","P","P","P","P","P"]

# =============================================================================
# 模型管理模組
# =============================================================================
models = {}
models_loaded = False

def load_models():
    global models, models_loaded
    if models_loaded:
        return True
    
    try:
        app.logger.info("🚀 啟動 AI 預測引擎...")
        models = {
            'scaler': joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl')),
            'xgb': joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl')),
            'hmm': joblib.load(os.path.join(MODEL_DIR, 'hmm_model.pkl')),
            'lgbm': joblib.load(os.path.join(MODEL_DIR, 'lgbm_model.pkl'))
        }
        models_loaded = True
        app.logger.info("✅ 模型載入完成")
        return True
    except Exception as e:
        app.logger.error(f"❌ 模型載入失敗: {str(e)}")
        return False

# =============================================================================
# 數據管理模組 (強化版)
# =============================================================================
def load_historical_data():
    """動態混合預載數據與用戶數據"""
    try:
        with open(DATA_FILE, 'r') as f:
            user_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        user_data = []
    
    # 數據清洗流程
    valid_data = [
        CHINESE_MAP.get(item, item) 
        for item in user_data 
        if CHINESE_MAP.get(item, item) in LABEL_MAP
    ]
    return valid_data + PRELOADED_DATA  # 確保新數據在前

def save_new_data(new_data):
    """異步數據保存機制"""
    valid_chars = set(CHINESE_MAP.values())
    filtered = [d for d in new_data if d in valid_chars]
    
    if not filtered:
        app.logger.warning(f"🚫 無效數據被過濾: {new_data}")
        return 0
    
    try:
        if not os.path.exists(os.path.dirname(DATA_FILE)):
            os.makedirs(os.path.dirname(DATA_FILE))
            
        with open(DATA_FILE, 'a') as f:  # 追加模式
            json.dump(filtered, f)
            f.write('\n')  # 多行儲存
        return len(filtered)
    except Exception as e:
        app.logger.error(f"💥 數據保存失敗: {str(e)}")
        return 0

# =============================================================================
# 路單分析引擎
# =============================================================================
class RoadmapAnalyzer:
    def __init__(self, roadmap):
        self.roadmap = roadmap
        self.big_road = self._build_big_road()
    
    def _build_big_road(self):
        grid, current_col = [], []
        last_result = None
        for res in self.roadmap:
            if res != last_result and last_result is not None:
                grid.append(current_col)
                current_col = []
            current_col.append(res)
            last_result = res
        if current_col:
            grid.append(current_col)
        return grid
    
    def _get_col_length(self, index):
        return len(self.big_road[index]) if 0 <= index < len(self.big_road) else 0
    
    def get_derived_features(self):
        features = []
        for offset in [1, 2, 3]:  # 大眼路/小路/蟑螂路
            road = []
            for col_idx in range(offset, len(self.big_road)):
                start_row = 1 if self._get_col_length(col_idx-1) == 1 else 0
                for row in range(start_row, 6):
                    if row >= self._get_col_length(col_idx):
                        break
                    # 核心分析邏輯...
                    road.append('R' if (row % 2 == 0) else 'B')
            red_ratio = road[-10:].count('R')/10 if len(road)>=10 else 0.5
            features.append(red_ratio)
        return features

# =============================================================================
# API 路由
# =============================================================================
@app.route('/predict', methods=['POST'])
def predict_handler():
    start = time.time()
    try:
        if not load_models():
            return jsonify({"error": "AI 引擎初始化失敗"}), 500
        
        data = request.get_json()
        raw_roadmap = data.get('roadmap', [])
        
        # 數據預處理
        roadmap = [
            CHINESE_MAP.get(item, item) 
            for item in raw_roadmap
            if CHINESE_MAP.get(item, item) in LABEL_MAP
        ]
        
        if len(roadmap) < N_FEATURES_WINDOW:
            return jsonify({"error": f"需至少{N_FEATURES_WINDOW}筆有效路單"}), 400
        
        # 特徵工程
        analyzer = RoadmapAnalyzer(roadmap)
        latest_window = [LABEL_MAP[r] for r in roadmap[-N_FEATURES_WINDOW:]]
        features = np.array([latest_window + analyzer.get_derived_features()])
        
        # 多模型集成預測
        scaled = models['scaler'].transform(features)
        xgb_proba = models['xgb'].predict_proba(scaled)[0]
        lgbm_proba = models['lgbm'].predict_proba(scaled)[0]
        final_proba = (xgb_proba + lgbm_proba) / 2
        
        return jsonify({
            "prediction": REVERSE_MAP[np.argmax(final_proba)],
            "confidence": float(np.max(final_proba)),
            "processing_time": f"{time.time()-start:.3f}s"
        })
        
    except Exception as e:
        app.logger.error(f"🔥 預測異常: {traceback.format_exc()}")
        return jsonify({"error": "系統異常"}), 500

# =============================================================================
# 系統健康檢查
# =============================================================================
@app.route('/health')
def health_check():
    return jsonify({
        "status": "active",
        "model_status": "loaded" if models_loaded else "unloaded",
        "data_count": len(load_historical_data())
    })

# =============================================================================
# 啟動入口
# =============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
