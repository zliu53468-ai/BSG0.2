# -*- coding: utf-8 -*-
import os
import json
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

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
# 全域變數與設定
# =============================================================================
MODEL_DIR = 'models'
LABEL_MAP = {'B': 0, 'P': 1}
REVERSE_MAP = {0: '莊', 1: '閒'}
N_FEATURES_WINDOW = 20

# =============================================================================
# 路單分析核心 (BaccaratAnalyzer) - 此版本與 train.py 完全一致
# =============================================================================
class BaccaratAnalyzer:
    def __init__(self, roadmap):
        self.roadmap = [r for r in roadmap if r in ['B', 'P', 'T']]
        self.big_road_grid = self._generate_big_road_grid()

    def _generate_big_road_grid(self):
        grid = []
        if not self.roadmap: return grid
        filtered_map = [r for r in self.roadmap if r in ['B', 'P']]
        if not filtered_map: return grid

        current_col = []
        last_result = None
        for result in filtered_map:
            if result != last_result and last_result is not None:
                grid.append(current_col)
                current_col = []
            current_col.append(result)
            last_result = result
        if current_col: grid.append(current_col)
        return grid

    def _get_derived_bead(self, c, r, offset):
        if c - offset < 0: return None 
        
        len_col_offset = len(self.big_road_grid[c - offset])
        len_col_prev = len(self.big_road_grid[c - 1])

        bead_at_pos = self.big_road_grid[c-offset][r] if r < len_col_offset else None
        bead_above = self.big_road_grid[c-offset][r-1] if r > 0 and r-1 < len_col_offset else None

        if bead_at_pos:
            return 'B' if not bead_above else 'R'
        else:
            if bead_above:
                len_col_prev_offset = len(self.big_road_grid[c-1-offset]) if c-1-offset >= 0 else 0
                return 'R' if len_col_prev == len_col_prev_offset else 'B'
            else:
                return 'B'

    def get_derived_roads(self):
        roads = {'big_eye': [], 'small': [], 'cockroach': []}
        if len(self.big_road_grid) < 2: return roads

        offsets = {'big_eye': 1, 'small': 2, 'cockroach': 3}

        for name, offset in offsets.items():
            road_data = []
            start_col = offset + 1
            if len(self.big_road_grid) < start_col: continue

            for c in range(start_col - 1, len(self.big_road_grid)):
                col_data = []
                start_row = 1 if len(self.big_road_grid[c-1]) == 1 and len(self.big_road_grid[c-offset]) > 1 else 0

                for r in range(start_row, 6): # 檢查固定深度
                    if r >= len(self.big_road_grid[c]): break
                    bead = self._get_derived_bead(c, r, offset)
                    if bead: col_data.append(bead)
                
                if col_data: road_data.append(col_data)
            roads[name] = road_data
        return roads

    def get_derived_road_predictions(self):
        predictions = {'big_eye': 0.5, 'small': 0.5, 'cockroach': 0.5}
        roads_data = self.get_derived_roads()
        for name, road in roads_data.items():
            if not road: continue
            flat_road = [bead for col in road for bead in col]
            if len(flat_road) > 2:
                window = flat_road[-10:]
                red_count = window.count('R')
                predictions[name] = red_count / len(window) if window else 0.5
        return [predictions['big_eye'], predictions['small'], predictions['cockroach']]

# =============================================================================
# 獨立預測函式
# =============================================================================
def get_hmm_prediction(hmm_model, roadmap):
    try:
        roadmap_numeric = np.array([LABEL_MAP[r] for r in roadmap if r in LABEL_MAP]).reshape(-1, 1)
        if len(roadmap_numeric) < 2: return "數據不足"
        
        hidden_states = hmm_model.predict(roadmap_numeric)
        last_state = hidden_states[-1]
        transition_probs = hmm_model.transmat_[last_state, :]
        emission_probs = hmm_model.emissionprob_
        prob_b = np.dot(transition_probs, emission_probs[:, LABEL_MAP['B']])
        prob_p = np.dot(transition_probs, emission_probs[:, LABEL_MAP['P']])
        
        total_prob = prob_b + prob_p
        if total_prob < 1e-9: return "觀望"

        prob_b /= total_prob; prob_p /= total_prob
        if abs(prob_b - prob_p) < 0.04: return "觀望"
        return "莊" if prob_b > prob_p else "閒"
    except Exception:
        return "觀望"

def get_xgb_prediction(scaler, xgb_model, roadmap):
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
    derived_features = analyzer.get_derived_road_predictions()
    
    all_features = np.array(basic_features + derived_features).reshape(1, -1)
    features_scaled = scaler.transform(all_features)
    
    xgb_pred_prob = xgb_model.predict_proba(features_scaled)[0]
    prediction = REVERSE_MAP[np.argmax(xgb_pred_prob)]
    probability = float(np.max(xgb_pred_prob))

    if probability < 0.52: prediction = "觀望"

    banker_prob = float(xgb_pred_prob[LABEL_MAP['B']])
    player_prob = float(xgb_pred_prob[LABEL_MAP['P']])
    
    return prediction, banker_prob, player_prob

# =============================================================================
# API Endpoint
# =============================================================================
@app.route("/", methods=["GET"])
def home(): return jsonify({"status": "online"})

@app.route('/health', methods=['GET'])
def health_check():
    try:
        joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    except FileNotFoundError: pass
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "roadmap" not in data: return jsonify({"error": "無效請求"}), 400

        received_roadmap = [r for r in data["roadmap"] if r in ["B", "P", "T"]]
        
        if len(received_roadmap) < N_FEATURES_WINDOW:
             return jsonify({
                "banker": 0.5, "player": 0.5, "tie": 0.05,
                "details": {"xgb_suggestion": "數據不足", "hmm_suggestion": "數據不足", "derived_roads": {}}
            })

        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))
        hmm_model = joblib.load(os.path.join(MODEL_DIR, 'hmm_model.pkl'))
        
        xgb_suggestion, banker_prob, player_prob = get_xgb_prediction(scaler, xgb_model, received_roadmap)
        hmm_suggestion = get_hmm_prediction(hmm_model, received_roadmap)
        
        analyzer = BaccaratAnalyzer(received_roadmap)
        derived_roads_data = analyzer.get_derived_roads()
        
        tie_prob = 1.0 - (banker_prob + player_prob)

        return jsonify({
            "banker": round(banker_prob, 4),
            "player": round(player_prob, 4),
            "tie": round(tie_prob, 4) if tie_prob > 0 else 0.05,
            "details": {
                "xgb_suggestion": xgb_suggestion,
                "hmm_suggestion": hmm_suggestion,
                "derived_roads": derived_roads_data
            }
        })
    except FileNotFoundError: return jsonify({"error": "模型檔案不存在"}), 500
    except Exception as e:
        app.logger.error(f"預測時發生錯誤: {e}", exc_info=True)
        return jsonify({"error": "內部伺服器錯誤"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

