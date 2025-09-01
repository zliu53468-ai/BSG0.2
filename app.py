# -*- coding: utf-8 -*-
import os
import json
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import lightgbm as lgb

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
N_FEATURES_WINDOW = 20
models = {}

def load_all_models():
    """在伺服器啟動時預先載入所有模型到記憶體"""
    global models
    try:
        models['scaler'] = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        models['xgb'] = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
        models['hmm'] = joblib.load(os.path.join(MODEL_DIR, 'hmm_model.pkl'))
        models['lgbm'] = joblib.load(os.path.join(MODEL_DIR, 'lgbm_model.pkl'))
        app.logger.info("✅ 所有 AI 專家模型已成功載入記憶體。")
    except Exception as e:
        app.logger.error(f"❌ 載入模型失敗: {e}", exc_info=True)

# =============================================================================
# 路單分析核心 (BaccaratAnalyzer)
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
    def get_derived_road_features(self):
        roads_data = self.get_derived_roads_data()
        features = []
        for name in ['big_eye', 'small', 'cockroach']:
            road = roads_data.get(name, [])
            flat_road = [bead for col in road for bead in col] if road else []
            if len(flat_road) > 5:
                window = flat_road[-10:]; red_count = window.count('R')
                features.append(red_count / len(window))
            else: features.append(0.5)
        return features

# =============================================================================
# 獨立預測函式
# =============================================================================
def get_hmm_prediction(hmm_model, roadmap_numeric):
    try:
        if len(roadmap_numeric) < 2: return "數據不足"
        hidden_states = hmm_model.predict(roadmap_numeric)
        last_state = hidden_states[-1]
        transition_probs = hmm_model.transmat_[last_state, :]
        emission_probs = hmm_model.emissionprob_
        prob_b = np.dot(transition_probs, emission_probs[:, 0])
        prob_p = np.dot(transition_probs, emission_probs[:, 1])
        total_prob = prob_b + prob_p
        if total_prob < 1e-9: return "觀望"
        prob_b /= total_prob; prob_p /= total_prob
        if abs(prob_b - prob_p) < 0.04: return "觀望"
        return "莊" if prob_b > prob_p else "閒"
    except Exception: return "觀望"

def get_ml_prediction(model, scaler, roadmap):
    window = roadmap[-N_FEATURES_WINDOW:]
    b_count=window.count('B'); p_count=window.count('P'); total=b_count+p_count
    b_ratio=b_count/total if total > 0 else 0.5; p_ratio=p_count/total if total > 0 else 0.5
    streak=0; last_result=None
    for item in reversed(window):
        if item in ['B','P']:
            if last_result is None: last_result=item; streak=1
            elif item==last_result: streak+=1
            else: break
    streak_type = LABEL_MAP.get(last_result, -1)
    prev_result = LABEL_MAP.get(window[-1], -1) if window else -1
    basic_features = [b_ratio, p_ratio, streak, streak_type, prev_result]
    analyzer = BaccaratAnalyzer(roadmap)
    derived_features = analyzer.get_derived_road_features()
    all_features = np.array(basic_features + derived_features).reshape(1, -1)
    features_scaled = scaler.transform(all_features)
    
    pred_prob = model.predict_proba(features_scaled)[0]
    prediction = REVERSE_MAP[np.argmax(pred_prob)]
    if float(np.max(pred_prob)) < 0.53: prediction = "觀望" # 信心閥值
        
    return prediction, float(pred_prob[0]), float(pred_prob[1])

# =============================================================================
# API Endpoint
# =============================================================================
@app.route("/", methods=["GET"])
def home(): return jsonify({"status": "online", "models_loaded": len(models)>0})

@app.route("/predict", methods=["POST"])
def predict():
    if not models: return jsonify({"error": "模型尚未載入，請稍後重試。"}), 503
    try:
        data = request.get_json(); received_roadmap = data["roadmap"]
        filtered_roadmap = [r for r in received_roadmap if r in ["B", "P"]]
        if len(filtered_roadmap) < N_FEATURES_WINDOW:
             return jsonify({
                "banker": 0.5, "player": 0.5, "tie": 0.05,
                "details": {"xgb": "數據不足", "hmm": "數據不足", "lgbm": "數據不足", "derived_roads": {}}
            })
        
        roadmap_numeric = np.array([LABEL_MAP[r] for r in filtered_roadmap]).reshape(-1, 1)

        xgb_suggestion, banker_prob, player_prob = get_ml_prediction(models['xgb'], models['scaler'], filtered_roadmap)
        lgbm_suggestion, _, _ = get_ml_prediction(models['lgbm'], models['scaler'], filtered_roadmap)
        hmm_suggestion = get_hmm_prediction(models['hmm'], roadmap_numeric)

        analyzer = BaccaratAnalyzer(filtered_roadmap)
        derived_roads_data = analyzer.get_derived_roads_data()
        tie_prob = 1.0 - (banker_prob + player_prob)

        return jsonify({
            "banker": round(banker_prob, 4), "player": round(player_prob, 4),
            "tie": round(tie_prob, 4) if tie_prob > 0 else 0.05,
            "details": {
                "xgb": xgb_suggestion, "hmm": hmm_suggestion, "lgbm": lgbm_suggestion,
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

