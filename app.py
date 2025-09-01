# -*- coding: utf-8 -*-
import os
import json
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# =============================================================================
# Flask 應用程式與日誌設定
# =============================================================================
app = Flask(__name__)
if not os.path.exists('logs'): os.makedirs('logs')
log_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=5)
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
log_handler.setFormatter(formatter)
if app.logger.hasHandlers(): app.logger.handlers.clear()
app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO)
CORS(app, resources={r"/*": {"origins": "*"}})

# =============================================================================
# 全域變數與模型預載
# =============================================================================
MODEL_DIR = 'models'
LABEL_MAP = {'B': 0, 'P': 1}
REVERSE_MAP = {0: '莊', 1: '閒'}
LSTM_SEQUENCE_LENGTH = 15
models = {}

def load_all_models():
    """在伺服器啟動時預先載入所有模型到記憶體"""
    global models
    try:
        models['lstm'] = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'lstm_model.h5'))
        app.logger.info("✅ LSTM AI 核心已成功載入記憶體。")
    except Exception as e:
        app.logger.error(f"❌ 載入模型失敗: {e}", exc_info=True)

# =============================================================================
# 路單分析核心 (僅供前端繪圖使用)
# =============================================================================
class BaccaratAnalyzer:
    def __init__(self, roadmap):
        self.roadmap = [r for r in roadmap if r in ['B', 'P']]
        self.big_road_grid = self._generate_big_road_grid()
    def _generate_big_road_grid(self):
        grid = []
        if not self.roadmap: return grid
        current_col, last_result = [], None
        for result in self.roadmap:
            if result != last_result and last_result is not None:
                grid.append(current_col)
                current_col = []
            current_col.append(result)
            last_result = result
        if current_col: grid.append(current_col)
        return grid
    def _get_col_len(self, c):
        return len(self.big_road_grid[c]) if 0 <= c < len(self.big_road_grid) else 0
    def _get_derived_bead_color(self, c, r, offset):
        if c < offset: return None
        if r == 0: return 'R' if self._get_col_len(c - 1) == self._get_col_len(c - offset -1) else 'B'
        ref_bead_exists = r < self._get_col_len(c - offset)
        ref_bead_above_exists = (r - 1) < self._get_col_len(c - offset)
        if ref_bead_exists: return 'R'
        elif ref_bead_above_exists: return 'B'
        return 'B'
    def get_derived_roads_data(self):
        roads = {'big_eye': [], 'small': [], 'cockroach': []}
        offsets = {'big_eye': 1, 'small': 2, 'cockroach': 3}
        for name, offset in offsets.items():
            derived_road_flat = []
            if len(self.big_road_grid) < offset + 1: continue
            for c in range(offset, len(self.big_road_grid)):
                start_row = 1 if len(self.big_road_grid[c - 1]) == 1 else 0
                for r in range(start_row, 6):
                    if r >= self._get_col_len(c): break
                    color = self._get_derived_bead_color(c, r, offset)
                    if color: derived_road_flat.append(color)
            if derived_road_flat:
                grid, current_col, last_bead = [], [], None
                for bead in derived_road_flat:
                    if bead != last_bead and last_bead is not None:
                        grid.append(current_col); current_col = []
                    current_col.append(bead); last_bead = bead
                if current_col: grid.append(current_col)
                roads[name] = grid
        return roads

# =============================================================================
# 預測函式
# =============================================================================
def get_lstm_prediction(lstm_model, roadmap_numeric):
    if len(roadmap_numeric) < LSTM_SEQUENCE_LENGTH: 
        return "數據不足", 0.5, 0.5
    sequence = roadmap_numeric[-LSTM_SEQUENCE_LENGTH:].reshape(1, LSTM_SEQUENCE_LENGTH, 1)
    prediction_prob_p = lstm_model.predict(sequence, verbose=0)[0][0]
    prediction_prob_b = 1 - prediction_prob_p
    
    if abs(prediction_prob_p - 0.5) < 0.03: 
        suggestion = "觀望"
    else:
        suggestion = "閒" if prediction_prob_p > 0.5 else "莊"
    
    return suggestion, float(prediction_prob_b), float(prediction_prob_p)

# =============================================================================
# API Endpoint
# =============================================================================
@app.route("/", methods=["GET"])
def home(): return jsonify({"status": "online", "model_loaded": 'lstm' in models})

@app.route("/predict", methods=["POST"])
def predict():
    if 'lstm' not in models: return jsonify({"error": "模型尚未載入，請稍後重試。"}), 503
    try:
        data = request.get_json(); received_roadmap = data["roadmap"]
        filtered_roadmap = [r for r in received_roadmap if r in ["B", "P"]]
        
        analyzer = BaccaratAnalyzer(filtered_roadmap)
        derived_roads_data = analyzer.get_derived_roads_data()
        
        if len(filtered_roadmap) < LSTM_SEQUENCE_LENGTH:
             return jsonify({
                "banker": 0.5, "player": 0.5, "tie": 0.05,
                "details": {"suggestion": "數據不足", "derived_roads": derived_roads_data}
            })
        
        roadmap_numeric = np.array([LABEL_MAP[r] for r in filtered_roadmap]).reshape(-1, 1)

        suggestion, banker_prob, player_prob = get_lstm_prediction(models['lstm'], roadmap_numeric)

        tie_prob = 1.0 - (banker_prob + player_prob)

        return jsonify({
            "banker": round(banker_prob, 4), "player": round(player_prob, 4),
            "tie": round(tie_prob, 4) if tie_prob > 0 else 0.05,
            "details": {
                "suggestion": suggestion,
                "derived_roads": derived_roads_data
            }
        })
    except Exception as e:
        app.logger.error(f"預測時發生錯誤: {e}", exc_info=True)
        return jsonify({"error": "內部伺服器錯誤"}), 500

if __name__ == "__main__":
    load_all_models()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    load_all_models()

